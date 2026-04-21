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
    df = df.sort_values(["fnid", "projection_start"]).copy()
    df["lag_1"]          = df.groupby("fnid")["ipc_phase"].shift(1)
    df["lag_2"]          = df.groupby("fnid")["ipc_phase"].shift(2)
    df["rolling_mean_3"] = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_max_3"]  = (
        df.groupby("fnid")["ipc_phase"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).max())
    )
    df["phase_trend"]  = df["lag_1"] - df["lag_2"]
    df["phase_change"] = df["ipc_phase"] - df["lag_1"]
    return df


def add_unit_features(df: pd.DataFrame) -> pd.DataFrame:
    unit_stats = (
        df.groupby("fnid")["ipc_phase"]
        .agg(unit_mean_phase="mean", unit_max_phase="max")
        .reset_index()
    )
    crisis_rate = (
        df.assign(is_crisis=(df["ipc_phase"] >= 3).astype(int))
        .groupby("fnid")["is_crisis"]
        .mean()
        .reset_index()
        .rename(columns={"is_crisis": "unit_pct_crisis"})
    )
    unit_stats = unit_stats.merge(crisis_rate, on="fnid")
    df = df.merge(unit_stats, on="fnid", how="left")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["unit_code"] = pd.Categorical(df["fnid"]).codes
    df["is_ipc2"]   = (df["scale"] == "IPC2").astype(int)
    return df


FEATURE_COLS = [
    "year", "month", "quarter", "is_lean_season", "period_days",
    "lag_1", "lag_2", "rolling_mean_3", "rolling_max_3", "phase_trend",
    "unit_mean_phase", "unit_max_phase", "unit_pct_crisis",
    "unit_code", "is_ipc2", "preference_rating",
]

TARGET_COL = "ipc_phase"


def build_feature_matrix(df: pd.DataFrame, drop_nulls: bool = True) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_unit_features(df)
    df = encode_categoricals(df)

    if drop_nulls:
        before = len(df)
        # only drop on cols that actually exist
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
