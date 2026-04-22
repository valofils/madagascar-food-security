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
    """
    Lag and rolling features computed strictly from past observations.
    All shifts use shift(1) so no current-period leakage.
    """
    df = df.sort_values(["fnid", "projection_start"]).copy()

    df["lag_1"]          = df.groupby("fnid")["ipc_phase"].shift(1)
    df["lag_2"]          = df.groupby("fnid")["ipc_phase"].shift(2)
    df["rolling_mean_3"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_max_3"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).max())
    )
    df["phase_trend"] = df["lag_1"] - df["lag_2"]

    # Expanding historical max per unit — uses only past rows (shift applied above)
    df["unit_hist_max"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).max())
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["is_ipc2"] = (df["scale"] == "IPC2").astype(int)
    return df


# ── Features used for training ─────────────────────────────────────────────────
# Removed: unit_mean_phase, unit_pct_crisis  — computed on full dataset (leakage)
#          unit_code                          — geographic identity, not conditions
#          unit_max_phase                     — replaced by unit_hist_max (lag-safe)
FEATURE_COLS = [
    "year", "month", "quarter", "is_lean_season", "period_days",
    "lag_1", "lag_2", "rolling_mean_3", "rolling_max_3", "phase_trend",
    "unit_hist_max",
    "is_ipc2", "preference_rating",
]

TARGET_COL = "ipc_phase"


def build_feature_matrix(df: pd.DataFrame, drop_nulls: bool = True) -> pd.DataFrame:
    df = add_lag_features(df)
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
    print(f"\nSample row:")
    print(df[FEATURE_COLS + [TARGET_COL]].iloc[10].to_dict())
    save(df, f"features_{scenario}.csv")
    print("\n=== Feature engineering complete ===")
    return df


if __name__ == "__main__":
    run_features("cs")
