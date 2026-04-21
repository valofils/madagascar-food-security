import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PROCESSED, MODELS_DIR
from src.features import FEATURE_COLS, TARGET_COL

def load_features(scenario: str = "cs") -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED, f"features_{scenario}.csv")
    df = pd.read_csv(path)
    print(f"Loaded features ({scenario}): {df.shape}")
    return df


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    """Phase 3+ = 1 (Crisis or worse), Phase 1-2 = 0."""
    return (df[TARGET_COL] >= 3).astype(int)


def make_multiclass_target(df: pd.DataFrame) -> pd.Series:
    """
    Merge Phase 4 into Phase 3, then remap to 0-indexed for XGBoost:
      Phase 1 -> 0 (Minimal/Stressed-low)
      Phase 2 -> 1 (Stressed)
      Phase 3+ -> 2 (Crisis or worse)
    """
    return df[TARGET_COL].clip(upper=3).astype(int) - 1


PHASE_LABELS = {0: "Minimal (P1)", 1: "Stressed (P2)", 2: "Crisis+ (P3+)"}


def temporal_split(df: pd.DataFrame, test_year: int = 2024):
    train = df[df["year"] < test_year].copy()
    test  = df[df["year"] >= test_year].copy()
    print(f"Train: {len(train)} rows ({train.year.min():.0f}-{train.year.max():.0f})")
    print(f"Test:  {len(test)} rows  ({test.year.min():.0f}-{test.year.max():.0f})")
    return train, test


def train_binary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("\n--- Binary model: Crisis (3+) vs Not Crisis ---")
    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]
    y_train, y_test = make_binary_target(train), make_binary_target(test)

    print(f"Train positives (crisis): {y_train.sum()} / {len(y_train)}")
    print(f"Test  positives (crisis): {y_test.sum()} / {len(y_test)}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test  accuracy: {model.score(X_test,  y_test):.4f}")
    return {"model": model, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "type": "binary"}


def train_multiclass(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("\n--- Multiclass model: Phase 1 / 2 / 3+ (0-indexed) ---")
    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]
    y_train, y_test = make_multiclass_target(train), make_multiclass_target(test)

    print(f"Train distribution: { {PHASE_LABELS[k]: v for k,v in y_train.value_counts().sort_index().items()} }")
    print(f"Test  distribution: { {PHASE_LABELS[k]: v for k,v in y_test.value_counts().sort_index().items()} }")

    weights = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        objective="multi:softmax", num_class=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=weights,
              eval_set=[(X_test, y_test)], verbose=False)

    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test  accuracy: {model.score(X_test,  y_test):.4f}")
    return {"model": model, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "type": "multiclass"}


def save_model(result: dict, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(result["model"], f)
    print(f"  [ok] Model -> {path}")

    meta = {
        "model_name": name,
        "model_type": result["type"],
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "train_accuracy": float(result["model"].score(result["X_train"], result["y_train"])),
        "test_accuracy":  float(result["model"].score(result["X_test"],  result["y_test"])),
    }
    meta_path = os.path.join(MODELS_DIR, f"{name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [ok] Meta  -> {meta_path}")


def run_training():
    print("\n=== Model Training ===\n")
    df = load_features("cs")
    train, test = temporal_split(df, test_year=2024)

    binary_result     = train_binary(train, test)
    multiclass_result = train_multiclass(train, test)

    save_model(binary_result,     "ipc_binary_classifier")
    save_model(multiclass_result, "ipc_multiclass_classifier")

    print("\n=== Training complete ===")


if __name__ == "__main__":
    run_training()
