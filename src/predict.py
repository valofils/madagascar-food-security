import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODELS_DIR
from src.features import FEATURE_COLS

IPC_PHASE_LABELS = {
    0: "Minimal",
    1: "Stressed",
    2: "Crisis",
    3: "Emergency",
    4: "Catastrophe",
}

CRISIS_LABEL = {
    0: "Not Crisis (Phase 1-2)",
    1: "Crisis or worse (Phase 3+)",
}


def load_artifact(name: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_df(features: dict) -> pd.DataFrame:
    """Convert a feature dict to a single-row DataFrame aligned to FEATURE_COLS."""
    row = {col: features.get(col, np.nan) for col in FEATURE_COLS}
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_binary(features: dict) -> dict:
    """
    Predict whether a geographic unit × period is Crisis (Phase 3+) or not.
    Returns label, probability, and threshold used.
    """
    artifact  = load_artifact("ipc_binary_classifier")
    model     = artifact["model"]
    threshold = artifact["threshold"]

    X        = _to_df(features)
    prob     = float(model.predict_proba(X)[0, 1])
    label    = int(prob >= threshold)

    return {
        "prediction":  label,
        "label":       CRISIS_LABEL[label],
        "probability": round(prob, 4),
        "threshold":   round(threshold, 4),
        "is_crisis":   bool(label == 1),
    }


def predict_multiclass(features: dict) -> dict:
    """
    Predict IPC phase class (0=P1, 1=P2, 2=P3+).
    Returns predicted class, label, and per-class probabilities.
    """
    artifact = load_artifact("ipc_multiclass_classifier")
    model    = artifact["model"]

    X        = _to_df(features)
    pred     = int(model.predict(X)[0])
    probs    = model.predict_proba(X)[0]

    return {
        "prediction": pred,
        "label":      ["Minimal (P1)", "Stressed (P2)", "Crisis+ (P3+)"][pred],
        "probabilities": {
            "Minimal (P1)":  round(float(probs[0]), 4),
            "Stressed (P2)": round(float(probs[1]), 4),
            "Crisis+ (P3+)": round(float(probs[2]), 4),
        },
    }


def predict_combined(features: dict) -> dict:
    """
    Run both models and return a combined prediction with confidence signal.
    This is the main entry point for the API.
    """
    binary     = predict_binary(features)
    multiclass = predict_multiclass(features)

    # agreement signal: both models agree on crisis
    both_crisis = binary["is_crisis"] and multiclass["prediction"] == 2
    alert_level = "HIGH" if both_crisis else ("MODERATE" if binary["is_crisis"] else "LOW")

    return {
        "alert_level":  alert_level,
        "binary":       binary,
        "multiclass":   multiclass,
        "features_used": {k: features.get(k) for k in FEATURE_COLS},
    }


if __name__ == "__main__":
    # smoke test with a sample Grand Sud feature vector
    sample = {
        "year": 2024, "month": 2, "quarter": 1,
        "is_lean_season": 1, "period_days": 29,
        "lag_1": 3.0, "lag_2": 3.0,
        "rolling_mean_3": 3.0, "rolling_max_3": 3.0,
        "phase_trend": 0.0,
        "unit_mean_phase": 2.8, "unit_max_phase": 4.0, "unit_pct_crisis": 0.65,
        "unit_code": 42, "is_ipc2": 0, "preference_rating": 90,
    }

    print("\n=== Predict: Grand Sud (high-risk profile) ===")
    result = predict_combined(sample)
    print(f"Alert level : {result['alert_level']}")
    print(f"Binary      : {result['binary']['label']} (prob={result['binary']['probability']})")
    print(f"Multiclass  : {result['multiclass']['label']}")
    print(f"Probabilities: {result['multiclass']['probabilities']}")

    print("\n=== Predict: low-risk profile ===")
    low_risk = {**sample, "lag_1": 1.0, "lag_2": 1.0, "rolling_mean_3": 1.0,
                "rolling_max_3": 1.0, "unit_mean_phase": 1.1,
                "unit_max_phase": 2.0, "unit_pct_crisis": 0.02}
    result2 = predict_combined(low_risk)
    print(f"Alert level : {result2['alert_level']}")
    print(f"Binary      : {result2['binary']['label']} (prob={result2['binary']['probability']})")
    print(f"Multiclass  : {result2['multiclass']['label']}")
