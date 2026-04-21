import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_RAW, DATA_PROCESSED, IPC_PHASES

# ── loaders ───────────────────────────────────────────────────────────────────

def load_ipcphase() -> pd.DataFrame:
    path = os.path.join(DATA_RAW, "fewsnet_ipcphase_mdg.json")
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded ipcphase: {len(df)} rows, {df.columns.tolist()}")
    return df


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. keep only Madagascar records (API sometimes bleeds other countries)
    df = df[df["country_code"] == "MG"].copy()
    print(f"After country filter: {len(df)} rows")

    # 2. drop rows with null phase value (our target)
    df = df[df["value"].notna()].copy()
    print(f"After dropping null phase: {len(df)} rows")

    # 3. parse dates
    df["projection_start"] = pd.to_datetime(df["projection_start"], errors="coerce")
    df["projection_end"]   = pd.to_datetime(df["projection_end"],   errors="coerce")
    df["reporting_date"]   = pd.to_datetime(df["reporting_date"],   errors="coerce")
    df = df[df["projection_start"].notna()].copy()
    print(f"After dropping null dates: {len(df)} rows")

    # 4. cast phase to int
    df["ipc_phase"] = df["value"].astype(int)

    # 5. harmonise IPC scale labels
    scale_map = {
        "IPC 2.0": "IPC2",
        "IPC 3.0": "IPC3",
        "IPC 3.1": "IPC3",
        "IPC Highest Household": "IPC3",
    }
    df["scale"] = df["classification_scale"].map(scale_map).fillna("IPC3")

    # 6. drop IPC Highest Household (different methodology)
    df = df[df["classification_scale"] != "IPC Highest Household"].copy()
    print(f"After dropping Highest Household: {len(df)} rows")

    # 7. extract temporal features
    df["year"]        = df["projection_start"].dt.year
    df["month"]       = df["projection_start"].dt.month
    df["quarter"]     = df["projection_start"].dt.quarter
    df["period_days"] = (df["projection_end"] - df["projection_start"]).dt.days

    # 8. lean season flag (Oct-Mar in Madagascar)
    df["is_lean_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)

    # 9. select and rename columns
    keep = [
        "fnid", "geographic_unit_name", "geographic_unit_full_name",
        "unit_type", "scenario", "scenario_name",
        "scale", "classification_scale",
        "projection_start", "projection_end", "reporting_date",
        "year", "month", "quarter", "period_days", "is_lean_season",
        "ipc_phase", "pct_phase3", "pct_phase4", "pct_phase5",
        "preference_rating", "is_allowing_for_assistance",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    return df.sort_values(["fnid", "projection_start"]).reset_index(drop=True)


# ── split by scenario ─────────────────────────────────────────────────────────

def split_scenarios(df: pd.DataFrame) -> dict:
    return {
        "cs":  df[df["scenario"] == "CS"].copy(),
        "ml1": df[df["scenario"] == "ML1"].copy(),
        "ml2": df[df["scenario"] == "ML2"].copy(),
    }


# ── save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, filename: str):
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    path = os.path.join(DATA_PROCESSED, filename)
    df.to_csv(path, index=False)
    print(f"  [ok] Saved -> {path}  ({len(df)} rows)")


# ── main ──────────────────────────────────────────────────────────────────────

def run_preprocessing():
    print("\n=== Preprocessing ===\n")

    df_raw = load_ipcphase()
    df     = clean(df_raw)

    print(f"\nFinal cleaned shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nPhase distribution:")
    print(df["ipc_phase"].value_counts().sort_index())
    print(f"\nScenario counts:")
    print(df["scenario"].value_counts())
    print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique geographic units: {df['fnid'].nunique()}")

    # save full cleaned dataset
    save(df, "ipcphase_clean.csv")

    # save scenario splits
    splits = split_scenarios(df)
    for name, split_df in splits.items():
        save(split_df, f"ipcphase_{name}.csv")

    print("\n=== Preprocessing complete ===")
    return df


if __name__ == "__main__":
    run_preprocessing()
