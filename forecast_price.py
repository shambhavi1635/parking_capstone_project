"""
forecast_price.py - Model 4: forecast-driven dynamic pricing.

Models 1-3 all price reactively: they look at occupancy that has already
happened. The point of dynamic pricing is to move the price BEFORE a lot fills,
so this module learns to predict the next utilization reading and prices on the
prediction instead.

Target   : utilization (Occupancy / Capacity) at the next 30-minute reading
Features : lagged utilization, intra-day momentum, time of day, lot capacity,
           and the geospatial competition features already in the dataset
Baseline : persistence ("next utilization = current utilization"), which is the
           honest thing to beat - a model that cannot beat it has learned nothing

Used as a library by pipeline.py, or run directly to print the model metrics:

    python forecast_price.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Pricing constants, kept identical to capstone1.py so the models stay comparable
BASE_PRICE = 10.0
ALPHA4 = 1.0
PRICE_MIN = 0.5 * BASE_PRICE
PRICE_MAX = 2.0 * BASE_PRICE

LAGS = (1, 2, 3)          # previous readings used as features
TRAIN_FRACTION = 0.8      # chronological split, no shuffling
RANDOM_STATE = 42


def build_panel(df):
    """Per-lot, per-reading panel with a lot id and a utilization column."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # The stream files have no lot id column, but coordinates identify the 14 lots
    if "SystemCodeNumber" in df.columns:
        df["lot"] = df["SystemCodeNumber"].astype(str)
    elif {"Latitude", "Longitude"}.issubset(df.columns):
        df["lot"] = (df["Latitude"].round(6).astype(str) + "_"
                     + df["Longitude"].round(6).astype(str))
    else:
        df["lot"] = "all"

    df["util"] = (df["Occupancy"] / df["Capacity"]).clip(0, 1.5)
    df["day"] = df["Timestamp"].dt.normalize()
    return df.sort_values(["lot", "Timestamp"]).reset_index(drop=True)


def make_features(df):
    """Lag and rolling features computed inside (lot, day) so nothing leaks
    across the overnight gap or between lots, and the target never sees ahead."""
    df = build_panel(df)
    grouped = df.groupby(["lot", "day"], sort=False)["util"]

    for lag in LAGS:
        df[f"util_lag{lag}"] = grouped.shift(lag)
    # Mean of the three previous readings. Averaging the lag columns is
    # equivalent to a 3-period rolling mean of the shifted series, and it cannot
    # silently roll across a lot or day boundary the way a groupby-shift-rolling
    # chain can.
    df["util_roll3"] = df[[f"util_lag{lag}" for lag in LAGS]].mean(axis=1)
    df["util_delta1"] = df["util_lag1"] - df["util_lag2"]   # recent momentum
    df["hour"] = df["Timestamp"].dt.hour
    df["minute_of_day"] = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Target: the NEXT reading's utilization, within the same lot and day
    df["util_next"] = df.groupby(["lot", "day"], sort=False)["util"].shift(-1)

    feature_cols = [f"util_lag{lag}" for lag in LAGS] + [
        "util", "util_roll3", "util_delta1",
        "minute_of_day", "hour", "dayofweek", "is_weekend", "Capacity",
    ]
    for extra in ("nearest_distances", "competitor_counts"):
        if extra in df.columns:
            feature_cols.append(extra)

    df = df.dropna(subset=feature_cols + ["util_next"]).reset_index(drop=True)
    return df, feature_cols


def chronological_split(df, train_fraction=TRAIN_FRACTION):
    """Split on calendar day: train on earlier days, test on later ones.
    A random split would leak future readings into training."""
    days = np.sort(df["day"].unique())
    cutoff = days[int(len(days) * train_fraction)]
    return df[df["day"] < cutoff].copy(), df[df["day"] >= cutoff].copy(), cutoff


def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_forecasters(df, verbose=True):
    """Fit the candidate models, score them against the persistence baseline,
    and return the best one along with the scored test frame."""
    feat_df, feature_cols = make_features(df)
    train, test, cutoff = chronological_split(feat_df)

    X_train, y_train = train[feature_cols], train["util_next"]
    X_test, y_test = test[feature_cols], test["util_next"]

    # Baseline: assume nothing changes before the next reading
    results = {"Persistence (baseline)": evaluate(y_test, test["util"])}

    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1,
            random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        results[name] = evaluate(y_test, model.predict(X_test))

    table = pd.DataFrame(results).T
    base_mae = table.loc["Persistence (baseline)", "MAE"]
    table["MAE_vs_baseline_%"] = 100 * (base_mae - table["MAE"]) / base_mae

    best_name = table.drop(index="Persistence (baseline)")["MAE"].idxmin()

    if verbose:
        print(f"Train/test split at day {pd.Timestamp(cutoff).date()}  "
              f"({len(train):,} train rows / {len(test):,} test rows, "
              f"{feat_df['lot'].nunique()} lots)")
        print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
        print()
        print("=== MODEL 4: occupancy forecast accuracy (held-out days) ===")
        print(table.to_string(float_format=lambda v: f"{v:,.4f}"))
        print(f"\nBest forecaster: {best_name}")
        best = fitted[best_name]
        if hasattr(best, "feature_importances_"):
            importances = (pd.Series(best.feature_importances_, index=feature_cols)
                           .sort_values(ascending=False).head(6))
            print("\nTop feature importances:")
            print(importances.to_string(float_format=lambda v: f"{v:,.4f}"))

    test = test.assign(util_pred=fitted[best_name].predict(X_test))
    return best_name, fitted[best_name], feature_cols, table, test


def price_from_forecast(util_pred):
    """Same price law as Model 1, but applied to PREDICTED utilization so the
    price leads demand instead of trailing it."""
    raw = BASE_PRICE * (1.0 + ALPHA4 * np.asarray(util_pred, dtype=float))
    return np.clip(raw, PRICE_MIN, PRICE_MAX)


def daily_forecast_prices(df, verbose=True):
    """Model 4 daily price series, aligned with the daily windows of Models 1-3.

    Restricted to the held-out test days, which is the only stretch where the
    forecast is honest. Returns (daily_frame, accuracy_table, best_model_name).
    """
    best_name, _model, _cols, table, test = train_forecasters(df, verbose=verbose)
    test = test.assign(price=price_from_forecast(test["util_pred"]))

    daily = (test.groupby("day")
             .agg(price=("price", "mean"),
                  util_mean=("util_next", "mean"),
                  util_pred_mean=("util_pred", "mean"),
                  n_readings=("price", "size"))
             .reset_index()
             .rename(columns={"day": "t"}))
    return daily, table, best_name


def load_source(verbose=True):
    """Prefer the richest input available, mirroring pipeline.py's loader."""
    for path in ("dataset.csv", "parking_stream_model2.csv", "parking_stream.csv"):
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if {"LastUpdatedDate", "LastUpdatedTime"}.issubset(df.columns):
            df["Timestamp"] = pd.to_datetime(
                df["LastUpdatedDate"] + " " + df["LastUpdatedTime"],
                format="%d-%m-%Y %H:%M:%S")
        if verbose:
            print(f"Source: {path}  ({len(df):,} rows)\n")
        return df
    raise FileNotFoundError("no input CSV found in the project directory")


if __name__ == "__main__":
    source = load_source()
    daily, accuracy, best = daily_forecast_prices(source)

    print("\n=== MODEL 4: resulting daily price (held-out days only) ===")
    price = daily["price"]
    print(f"days       : {len(daily)}")
    print(f"mean price : ${price.mean():.3f}")
    print(f"range      : ${price.min():.3f} - ${price.max():.3f}")
    print(f"std        : {price.std(ddof=1):.3f}")
    print(f"corr(price, realised utilization) = "
          f"{np.corrcoef(price, daily['util_mean'])[0, 1]:+.4f}")
