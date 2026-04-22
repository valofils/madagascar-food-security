import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PROCESSED


def load_clean(scenario: str = "cs") -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED, f"ipcphase_{scenario}.csv")
    df = pd.read_csv(path, parse_dates=["projection_start", "projection_end", "reporting_date"])
    print(f"Loaded {scenario}: {len(df)} rows")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling features — all shift(1) so no current-period leakage."""
    df = df.sort_values(["fnid", "projection_start"]).copy()

    df["lag_1"]          = df.groupby("fnid")["ipc_phase"].shift(1)
    df["lag_2"]          = df.groupby("fnid")["ipc_phase"].shift(2)
    df["lag_3"]          = df.groupby("fnid")["ipc_phase"].shift(3)
    df["rolling_mean_3"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_max_3"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).max())
    )
    df["phase_trend"] = df["lag_1"] - df["lag_2"]

    # Expanding historical max per unit — past observations only
    df["unit_hist_max"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).max())
    )

    # Rolling count of Crisis periods in past 4 observations (crisis momentum)
    df["crisis_momentum"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: (x.shift(1) >= 3).rolling(4, min_periods=1).sum())
    )

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features targeting cold-start Crisis escalation —
    the pattern the base model missed (lag < 2.5 but lean season + worsening trend).
    """
    # Cold-start flag: previous phase below Crisis threshold
    df["is_cold_start"] = (df["lag_1"] < 2.5).astype(int)

    # Lean season amplifies escalation risk — interaction with recent phase level
    df["lean_x_lag1"]   = df["is_lean_season"] * df["lag_1"]

    # Acceleration: is the trend worsening AND in lean season?
    df["lean_x_trend"]  = df["is_lean_season"] * df["phase_trend"].clip(lower=0)

    # How far below Crisis was the previous phase? (escalation distance)
    df["gap_to_crisis"]  = (3.0 - df["lag_1"]).clip(lower=0)

    # Combined risk: moderate phase + lean season + worsening trend
    df["escalation_risk"] = (
        df["rolling_mean_3"] * df["is_lean_season"] * (1 + df["phase_trend"].clip(lower=0))
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["is_ipc2"] = (df["scale"] == "IPC2").astype(int)
    return df


# ── Feature columns ────────────────────────────────────────────────────────────
# Base temporal + lag features (leak-free)
# + interaction features targeting cold-start escalation
FEATURE_COLS = [
    # Temporal
    "year", "month", "quarter", "is_lean_season", "period_days",
    # Lag / rolling
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_max_3",
    "phase_trend",
    # Unit history (expanding, past-only)
    "unit_hist_max", "crisis_momentum",
    # Interaction / derived
    "is_cold_start", "lean_x_lag1", "lean_x_trend",
    "gap_to_crisis", "escalation_risk",
    # Categorical
    "is_ipc2", "preference_rating",
]

TARGET_COL = "ipc_phase"


def build_feature_matrix(df: pd.DataFrame, drop_nulls: bool = True) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_interaction_features(df)
    df = encode_categoricals(df)
    if drop_nulls:
        before = len(df)
        subset = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
        df = df.dropna(subset=subset).copy()
        print(f"Dropped {before - len(df)} rows with nulls in features")
    return df


def save(df: pd.DataFrame, filename: str):
    path = os.path.join(DATA_PROCESSED, filename)
    df.to_csv(path, index=False)
    print(f"  [ok] Saved -> {path}  ({len(df)} rows, {len(df.columns)} cols)")


def run_features(scenario: str = "cs"):
    print(f"\n=== Feature Engineering ({scenario.upper()}) ===\n")
    df = load_clean(scenario)
    df = build_feature_matrix(df)
    print(f"\nFeature matrix shape: {df.shape}")
    print(f"\nTarget distribution:")
    print(df[TARGET_COL].value_counts().sort_index())
    print(f"\nNull counts in features:")
    null_counts = df[FEATURE_COLS].isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.any() else "  none")
    save(df, f"features_{scenario}.csv")
    print("\n=== Feature engineering complete ===")
    return df


if __name__ == "__main__":
    run_features("cs")
