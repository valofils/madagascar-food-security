"""
src/lstm_features.py
--------------------
Builds 3D sequence arrays (samples, timesteps, features) from the
preprocessed IPC feature matrix, suitable for LSTM input.

Each sample is one livelihood zone at one point in time.
The sequence window contains the T previous observations for that zone.

Usage:
    from src.lstm_features import build_sequences, SEQUENCE_FEATURES

    X_train, y_train, X_test, y_test = build_sequences(df, seq_len=4)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── features used in the sequence (subset of FEATURE_COLS) ───────────────────
# We drop unit-level aggregates (unit_mean_phase etc.) because they are
# constant across all timesteps for a given unit — they add no temporal signal.
# The LSTM learns temporal dynamics from the raw and lagged phase values.

SEQUENCE_FEATURES = [
    "ipc_phase",          # raw phase (target is next-step phase)
    "is_lean_season",     # seasonal signal
    "period_days",        # window length
    "preference_rating",  # data quality
    "unit_pct_crisis",    # unit-level crisis rate (static context)
    "unit_mean_phase",    # unit-level mean (static context)
]

# binary crisis target: Phase 3+
CRISIS_THRESHOLD = 3


def make_crisis_target(series: pd.Series) -> pd.Series:
    return (series >= CRISIS_THRESHOLD).astype(int)


def build_sequences(
    df: pd.DataFrame,
    seq_len: int = 4,
    test_year: int = 2024,
    features: list = None,
    scale: bool = True,
) -> tuple:
    """
    Build (X_train, y_train, X_test, y_test, scaler) for LSTM training.

    Parameters
    ----------
    df       : preprocessed feature DataFrame (output of features.py)
    seq_len  : number of past timesteps per sample
    test_year: temporal split — train < test_year, test >= test_year
    features : list of column names to use (defaults to SEQUENCE_FEATURES)
    scale    : whether to StandardScale the features

    Returns
    -------
    X_train  : np.ndarray shape (n_train, seq_len, n_features)
    y_train  : np.ndarray shape (n_train,)  — binary crisis label
    X_test   : np.ndarray shape (n_test,  seq_len, n_features)
    y_test   : np.ndarray shape (n_test,)
    scaler   : fitted StandardScaler (or None if scale=False)
    """
    if features is None:
        features = SEQUENCE_FEATURES

    # sort chronologically within each unit
    df = df.sort_values(["fnid", "projection_start"]).copy()

    # fill any remaining nulls in sequence features with forward fill per unit
    df[features] = (
        df.groupby("fnid")[features]
        .transform(lambda x: x.ffill().bfill())
    )
    df = df.dropna(subset=features + ["ipc_phase"]).copy()

    X_seqs, y_labels, years = [], [], []

    for fnid, group in df.groupby("fnid"):
        group = group.reset_index(drop=True)
        if len(group) <= seq_len:
            continue  # not enough history for even one sample

        feat_arr = group[features].values.astype(np.float32)
        phase_arr = group["ipc_phase"].values
        year_arr = group["year"].values

        for i in range(seq_len, len(group)):
            X_seqs.append(feat_arr[i - seq_len : i])   # past seq_len steps
            y_labels.append(int(phase_arr[i] >= CRISIS_THRESHOLD))  # next step
            years.append(int(year_arr[i]))

    X = np.array(X_seqs, dtype=np.float32)   # (N, seq_len, n_features)
    y = np.array(y_labels, dtype=np.int32)   # (N,)
    years = np.array(years, dtype=np.int32)

    # temporal split
    train_mask = years < test_year
    test_mask  = years >= test_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    # scale on train, apply to test
    scaler = None
    if scale:
        n_train, t, f = X_train.shape
        scaler = StandardScaler()
        X_train = scaler.fit_transform(
            X_train.reshape(-1, f)
        ).reshape(n_train, t, f)

        n_test = X_test.shape[0]
        X_test = scaler.transform(
            X_test.reshape(-1, f)
        ).reshape(n_test, t, f)

    print(f"Sequences built:")
    print(f"  seq_len    : {seq_len}")
    print(f"  features   : {len(features)}  {features}")
    print(f"  X_train    : {X_train.shape}  |  Crisis rate: {y_train.mean():.3f}")
    print(f"  X_test     : {X_test.shape}   |  Crisis rate: {y_test.mean():.3f}")

    return X_train, y_train, X_test, y_test, scaler


if __name__ == "__main__":
    from config import DATA_PROCESSED

    df = pd.read_csv(
        os.path.join(DATA_PROCESSED, "features_cs.csv"),
        parse_dates=["projection_start"],
    )
    X_train, y_train, X_test, y_test, scaler = build_sequences(df, seq_len=4)
    print(f"\nReady for LSTM input.")
    print(f"  Sample X[0] shape : {X_train[0].shape}")
    print(f"  Sample y[0]       : {y_train[0]}")
