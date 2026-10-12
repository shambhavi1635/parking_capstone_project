"""
pipeline.py - run all pricing models end to end and report their metrics.

capstone1.py is the canonical implementation: it runs the three models as real
Pathway streaming jobs in Colab. This file reproduces the same daily tumbling
windows and the same pricing formulas in pandas, so the whole thing runs
anywhere (Pathway is Linux/Colab-oriented and is not required here) and the
metrics can actually be measured rather than estimated.

The formulas below are copied from capstone1.py and must stay in sync:

  Model 1  rule-based         price = BASE * (1 + alpha * util_mean), clamped
  Model 2  demand function    demand = 1.0*occ_rate + 0.5*queue - 0.7*traffic
                                       + 1.5*special + 0.8*vehicle_weight
                              price  = BASE + 0.5 * demand
  Model 3  competition-aware  discount 10% when full and undercut,
                              premium 10% when competitors are >10% above
  Model 4  forecast-driven    forecast_price.py (optional, needs scikit-learn)

Usage:
    python pipeline.py              # all models + metrics
    python pipeline.py --no-model4  # skip the ML forecaster
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Make the sibling forecast_price.py importable even under `python -I`, which
# does not put the script's own directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- constants --
BASE_PRICE = 10.0
ALPHA1 = 1.0                      # Model 1 utilization sensitivity
PRICE_MIN = 0.5 * BASE_PRICE
PRICE_MAX = 2.0 * BASE_PRICE
FULL_UTIL_THRESHOLD = 0.9         # Model 3 "lot is full" peak-utilization cut
COMPETITOR_RADIUS_M = 500
EARTH_RADIUS_M = 6_371_000.0
RANDOM_SEED = 42

# Columns capstone1.py is known to have filled with np.random. When they come
# from one of the generated stream files we re-derive them from real signal
# instead of trusting the stored noise.
SIMULATED_IN_STREAM_FILES = [
    "QueueLength", "TrafficLevel", "IsSpecialDay", "VehicleTypeWeight",
]


# -------------------------------------------------------------------- input --
def load_source(verbose=True):
    """Prefer the real dataset; fall back to the generated stream files."""
    for path in ("dataset.csv", "parking_stream_model2.csv", "parking_stream.csv"):
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue

        if {"LastUpdatedDate", "LastUpdatedTime"}.issubset(df.columns):
            df["Timestamp"] = pd.to_datetime(
                df["LastUpdatedDate"] + " " + df["LastUpdatedTime"],
                format="%d-%m-%Y %H:%M:%S")
        else:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        if path != "dataset.csv":
            # These were np.random in the generated files; drop them so the
            # provenance logic below re-derives them from observed signal.
            dropped = [c for c in SIMULATED_IN_STREAM_FILES if c in df.columns]
            df = df.drop(columns=dropped)
            if verbose and dropped:
                print(f"Source: {path}  ({len(df):,} rows)")
                print(f"  dropped stored simulated columns: {', '.join(dropped)}")
        elif verbose:
            print(f"Source: {path}  ({len(df):,} rows)")

        return df.sort_values("Timestamp").reset_index(drop=True)

    raise FileNotFoundError("no input CSV found in the project directory")


# ----------------------------------------------------------------- features --
def add_demand_features(df, verbose=True):
    """The four demand features the problem statement requires.

    Real dataset columns are mapped whenever they exist; anything genuinely
    missing is derived from observed signal, and what remains unobservable is
    labelled SIMULATED so the report can say so.
    """
    np.random.seed(RANDOM_SEED)
    provenance = {}

    util = (df["Occupancy"] / df["Capacity"]).clip(0, 1.5)
    hour = df["Timestamp"].dt.hour

    # QueueLength: cars waiting (0-10)
    if "QueueLength" in df.columns and df["QueueLength"].notna().any():
        df["QueueLength"] = pd.to_numeric(df["QueueLength"], errors="coerce").fillna(0).clip(0, 10)
        provenance["QueueLength"] = "dataset column"
    else:
        pressure = ((util - 0.7) / 0.3).clip(0, 1)      # queues form near capacity
        df["QueueLength"] = np.rint(
            pressure * 10 + np.random.normal(0, 0.8, size=len(df))
        ).clip(0, 10).astype(int)
        provenance["QueueLength"] = "derived from utilization (>70% occupancy)"

    # TrafficLevel: 0 low, 1 medium, 2 high
    if "TrafficConditionNearby" in df.columns and df["TrafficConditionNearby"].notna().any():
        mapping = {"low": 0, "average": 1, "medium": 1, "high": 2}
        df["TrafficLevel"] = (df["TrafficConditionNearby"].astype(str).str.strip()
                              .str.lower().map(mapping).fillna(1).astype(int))
        provenance["TrafficLevel"] = "dataset column TrafficConditionNearby"
    elif "TrafficLevel" in df.columns and df["TrafficLevel"].notna().any():
        df["TrafficLevel"] = pd.to_numeric(
            df["TrafficLevel"], errors="coerce").fillna(1).clip(0, 2).astype(int)
        provenance["TrafficLevel"] = "dataset column"
    else:
        df["TrafficLevel"] = np.select(
            [hour.isin([8, 9, 17, 18, 19]), hour.between(10, 16)], [2, 1], default=0
        ).astype(int)
        provenance["TrafficLevel"] = "derived from hour-of-day rush-hour profile"

    # IsSpecialDay: events or holidays
    if "IsSpecialDay" in df.columns and df["IsSpecialDay"].notna().any():
        df["IsSpecialDay"] = pd.to_numeric(
            df["IsSpecialDay"], errors="coerce").fillna(0).clip(0, 1).astype(int)
        provenance["IsSpecialDay"] = "dataset column"
    else:
        # Not inferable from a timestamp. A weekend flag was tried and rejected:
        # weekends are quieter here, so the PS's +1.5 coefficient would raise the
        # price exactly when demand fell.
        df["IsSpecialDay"] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        provenance["IsSpecialDay"] = "SIMULATED - not inferable from timestamps (~5%)"

    # VehicleTypeWeight
    if "VehicleType" in df.columns and df["VehicleType"].notna().any():
        weights = {"cycle": 0.5, "bike": 0.5, "bicycle": 0.5,
                   "car": 1.0, "small": 1.0,
                   "suv": 1.5, "van": 1.5,
                   "truck": 2.0, "ev": 2.0, "electric": 2.0}
        df["VehicleTypeWeight"] = (df["VehicleType"].astype(str).str.strip()
                                   .str.lower().map(weights).fillna(1.0).astype(float))
        provenance["VehicleTypeWeight"] = "dataset column VehicleType"
    elif "VehicleTypeWeight" in df.columns and df["VehicleTypeWeight"].notna().any():
        df["VehicleTypeWeight"] = pd.to_numeric(
            df["VehicleTypeWeight"], errors="coerce").fillna(1.0)
        provenance["VehicleTypeWeight"] = "dataset column"
    else:
        df["VehicleTypeWeight"] = np.random.choice(
            [1.0, 1.5, 2.0], size=len(df), p=[0.7, 0.2, 0.1])
        provenance["VehicleTypeWeight"] = "SIMULATED - no vehicle-type column in source"

    df["util"] = util
    df["day"] = df["Timestamp"].dt.normalize()

    if verbose:
        print("\nDemand feature provenance:")
        for key, value in provenance.items():
            print(f"  {key:20s} <- {value}")
        print("\nCorrelation of each demand feature with utilization:")
        for col in SIMULATED_IN_STREAM_FILES:
            r = np.corrcoef(df[col].astype(float), util)[0, 1]
            print(f"  {col:20s} r = {r:+.4f}")

    return df, provenance


def add_proximity_features(df, radius_m=COMPETITOR_RADIUS_M):
    """Nearest-competitor distance and competitor count within the radius.
    Same BallTree/haversine approach as capstone1.py."""
    if not {"Latitude", "Longitude"}.issubset(df.columns):
        return df

    coords = np.radians(df[["Latitude", "Longitude"]].astype(float).to_numpy())
    tree = BallTree(coords, metric="haversine")
    radius_rad = radius_m / EARTH_RADIUS_M

    counts = tree.query_radius(coords, r=radius_rad, count_only=True) - 1  # drop self
    # Nearest other lot: k=2 so the self-match at distance 0 can be discarded
    dist, _idx = tree.query(coords, k=2)
    df["competitor_counts"] = np.maximum(counts, 0)
    df["nearest_distances"] = dist[:, 1] * EARTH_RADIUS_M
    return df


# ------------------------------------------------------------------ model 1 --
def run_model1(df):
    """Rule-based: price rises linearly with mean daily utilization, clamped."""
    daily = df.groupby("day").agg(
        occ_max=("Occupancy", "max"),
        occ_min=("Occupancy", "min"),
        cap=("Capacity", "max"),
        util_mean=("util", "mean"),
        util_peak=("util", "max"),
    ).reset_index().rename(columns={"day": "t"})

    raw = BASE_PRICE * (1.0 + ALPHA1 * daily["util_mean"])
    daily["raw_price"] = raw
    daily["price"] = raw.clip(PRICE_MIN, PRICE_MAX)
    daily["clamped"] = (raw < PRICE_MIN) | (raw > PRICE_MAX)
    return daily


# ------------------------------------------------------------------ model 2 --
def run_model2(df):
    """Demand function over the daily window, exactly as in capstone1.py."""
    daily = df.groupby("day").agg(
        occ_sum=("Occupancy", "sum"),
        cap_sum=("Capacity", "sum"),
        avg_queue_length=("QueueLength", "mean"),
        avg_traffic_level=("TrafficLevel", "mean"),
        avg_is_special_day=("IsSpecialDay", "mean"),
        avg_vehicle_type_weight=("VehicleTypeWeight", "mean"),
        util_mean=("util", "mean"),
    ).reset_index().rename(columns={"day": "t"})

    daily["occupancy_rate"] = daily["occ_sum"] / (daily["cap_sum"] + 0.1)
    daily["demand"] = (
        1.0 * daily["occupancy_rate"]
        + 0.5 * daily["avg_queue_length"]
        - 0.7 * daily["avg_traffic_level"]
        + 1.5 * daily["avg_is_special_day"]
        + 0.8 * daily["avg_vehicle_type_weight"]
    )
    daily["price"] = BASE_PRICE + 0.5 * daily["demand"]
    return daily


def demand_variance_shares(daily):
    """How much of the demand score's variance each term contributes - the check
    that caught ~63% of Model 2's price signal coming from np.random."""
    terms = {
        "occupancy_rate (w=1.0)": 1.0 * daily["occupancy_rate"],
        "avg_queue_length (w=0.5)": 0.5 * daily["avg_queue_length"],
        "avg_traffic_level (w=-0.7)": -0.7 * daily["avg_traffic_level"],
        "avg_is_special_day (w=1.5)": 1.5 * daily["avg_is_special_day"],
        "avg_vehicle_wt (w=0.8)": 0.8 * daily["avg_vehicle_type_weight"],
    }
    var = pd.Series({k: v.var(ddof=1) for k, v in terms.items()})
    return pd.DataFrame({"variance": var, "share_%": 100 * var / var.sum()}) \
        .sort_values("share_%", ascending=False)


# ------------------------------------------------------------------ model 3 --
def run_model3(df, model2_daily):
    """Competition-aware: nudge Model 2's price against nearby lots' prices."""
    rows = df.copy()

    # Attach Model 2's daily price back onto each reading (capstone1.py does the
    # same with merge_asof) so a per-lot competitor price can be computed.
    rows = pd.merge_asof(
        rows.sort_values("Timestamp"),
        model2_daily[["t", "price"]].sort_values("t"),
        left_on="Timestamp", right_on="t", direction="nearest",
    )

    if {"Latitude", "Longitude"}.issubset(rows.columns):
        coords = np.radians(rows[["Latitude", "Longitude"]].astype(float).to_numpy())
        tree = BallTree(coords, metric="haversine")
        radius_rad = COMPETITOR_RADIUS_M / EARTH_RADIUS_M
        neighbours = tree.query_radius(coords, r=radius_rad)
        prices = rows["price"].to_numpy(dtype=float)
        mean_competitor = np.empty(len(rows))
        for i, idx in enumerate(neighbours):
            others = idx[idx != i]
            mean_competitor[i] = prices[others].mean() if others.size else prices[i]
        rows["mean_competitor_price"] = mean_competitor
    else:
        rows["mean_competitor_price"] = rows["price"]

    daily = rows.groupby("day").agg(
        price=("price", "mean"),
        competitor_price=("mean_competitor_price", "mean"),
        util_peak=("util", "max"),
        util_mean=("util", "mean"),
    ).reset_index().rename(columns={"day": "t"})

    daily["is_full"] = daily["util_peak"] >= FULL_UTIL_THRESHOLD
    discount = daily["is_full"] & (daily["competitor_price"] < daily["price"])
    premium = ~discount & (daily["competitor_price"] > daily["price"] * 1.1)

    daily["adjusted_price"] = np.where(
        discount, daily["price"] * 0.9,
        np.where(premium, daily["price"] * 1.1, daily["price"]))
    daily["branch"] = np.where(discount, "discount",
                               np.where(premium, "premium", "unchanged"))
    return daily


# ------------------------------------------------------------------ metrics --
def price_metrics(prices, utilization):
    """Behavioural diagnostics. There is no ground-truth price in this dataset,
    so these describe how a price series behaves rather than its accuracy."""
    p = np.asarray(prices, dtype=float)
    changes = np.diff(p)
    return {
        "days": len(p),
        "mean_price": p.mean(),
        "min_price": p.min(),
        "max_price": p.max(),
        "std": p.std(ddof=1),
        "cv_%": 100 * p.std(ddof=1) / p.mean(),
        "mean_abs_daily_change": np.abs(changes).mean() if changes.size else 0.0,
        "max_abs_daily_change": np.abs(changes).max() if changes.size else 0.0,
        "within_bounds_%": 100 * np.mean((p >= PRICE_MIN) & (p <= PRICE_MAX)),
        "max_x_base": p.max() / BASE_PRICE,
        "corr_with_utilization": np.corrcoef(p, np.asarray(utilization, float))[0, 1],
    }


def main(run_model4=True):
    df = load_source()
    df, _provenance = add_demand_features(df)
    df = add_proximity_features(df)

    m1 = run_model1(df)
    m2 = run_model2(df)
    m3 = run_model3(df, m2)

    series = {
        "Model 1 (rule-based)": (m1["price"], m1["util_mean"]),
        "Model 2 (demand function)": (m2["price"], m2["util_mean"]),
        "Model 3 (competition-aware)": (m3["adjusted_price"], m3["util_mean"]),
    }

    forecast_table = None
    if run_model4:
        try:
            from forecast_price import daily_forecast_prices
            print()
            m4, forecast_table, best = daily_forecast_prices(df, verbose=True)
            series[f"Model 4 (forecast, {best})"] = (m4["price"], m4["util_mean"])
        except ImportError as exc:
            print(f"\nModel 4 skipped ({exc})")

    metrics = pd.DataFrame({name: price_metrics(p, u)
                            for name, (p, u) in series.items()}).T

    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print("PRICE BEHAVIOUR METRICS (daily windows, base price $10)")
    print("=" * 78)
    print(metrics.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n=== MODEL 2: where the demand signal comes from ===")
    shares = demand_variance_shares(m2)
    print(shares.to_string(float_format=lambda v: f"{v:,.3f}"))
    sim = shares.loc[[i for i in shares.index
                      if "special_day" in i or "vehicle" in i], "share_%"].sum()
    print(f"\nshare of demand variance from still-simulated features: {sim:.1f}%")

    print("\n=== MODEL 3: competition branch activity ===")
    counts = m3["branch"].value_counts()
    for branch in ("discount", "premium", "unchanged"):
        n = int(counts.get(branch, 0))
        print(f"  {branch:10s} {n:3d} / {len(m3)} days  ({100 * n / len(m3):5.1f}%)")
    moved = m3["branch"] != "unchanged"
    print(f"  days flagged is_full : {int(m3['is_full'].sum())} / {len(m3)}")
    print(f"  mean |adjustment|    : "
          f"${np.abs(m3['adjusted_price'] - m3['price']).mean():.4f}")
    print(f"  adjustment rate      : {100 * moved.mean():.1f}%")
    busy = m3.loc[m3["branch"] == "discount", "util_mean"].mean()
    calm = m3.loc[m3["branch"] == "unchanged", "util_mean"].mean()
    print(f"  mean utilization     : discount days {busy:.3f} vs unchanged {calm:.3f}")
    print("  NOTE: Model 3's negative utilization correlation is the PS rule working")
    print("  as written - 'full and undercut -> cut the price' deliberately sheds")
    print("  demand toward cheaper neighbours, so demand correlation is the wrong")
    print("  success metric here. Judge Model 3 on adjustment rate and bound")
    print("  compliance instead. The premium branch stays at 0% because Model 2")
    print("  emits one city-wide price per day, so a lot and its neighbours are")
    print("  never 10% apart; per-lot pricing is what would activate it.")

    print("\n=== MODEL 1: clamp activity ===")
    print(f"  price clamped to [{PRICE_MIN:.0f}, {PRICE_MAX:.0f}] on "
          f"{int(m1['clamped'].sum())} / {len(m1)} days")

    if forecast_table is not None:
        print("\n=== MODEL 4: forecast accuracy (held-out days) ===")
        print(forecast_table.to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\nNote: Models 1-3 price reactively on observed occupancy; Model 4 "
          "prices on predicted\noccupancy and is scored only on held-out days, "
          "so its day count is lower by design.")
    return metrics


if __name__ == "__main__":
    main(run_model4="--no-model4" not in sys.argv)
