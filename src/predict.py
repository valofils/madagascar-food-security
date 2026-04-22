import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODELS_DIR
from src.features import FEATURE_COLS

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
    artifact  = load_artifact("ipc_binary_classifier")
    model     = artifact["model"]
    threshold = artifact["threshold"]

    X    = _to_df(features)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)

    return {
        "prediction":  pred,
        "label":       CRISIS_LABEL[pred],
        "probability": round(prob, 4),
        "threshold":   round(threshold, 4),
        "is_crisis":   bool(pred == 1),
    }


def predict_multiclass(features: dict) -> dict:
    artifact = load_artifact("ipc_multiclass_classifier")
    model    = artifact["model"]

    X     = _to_df(features)
    pred  = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

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
    """Run both models and return a combined prediction. Main API entry point."""
    binary     = predict_binary(features)
    multiclass = predict_multiclass(features)

    both_crisis = binary["is_crisis"] and multiclass["prediction"] == 2
    alert_level = "HIGH" if both_crisis else ("MODERATE" if binary["is_crisis"] else "LOW")

    return {
        "alert_level": alert_level,
        "binary":      binary,
        "multiclass":  multiclass,
        "features_used": {k: features.get(k) for k in FEATURE_COLS},
    }


if __name__ == "__main__":
    # Smoke test — v1.1 feature schema
    HIGH_RISK = {
        "year": 2024, "month": 1, "quarter": 1,
        "is_lean_season": 1, "period_days": 90,
        "lag_1": 2.5, "lag_2": 2.2, "lag_3": 2.0,
        "rolling_mean_3": 2.3, "rolling_max_3": 2.5,
        "phase_trend": 0.3,
        "unit_hist_max": 3.0, "crisis_momentum": 1.0,
        "is_cold_start": 1, "lean_x_lag1": 2.5, "lean_x_trend": 0.3,
        "gap_to_crisis": 0.5, "escalation_risk": 3.45,
        "is_ipc2": 0, "preference_rating": 1.0,
    }

    LOW_RISK = {
        "year": 2024, "month": 6, "quarter": 2,
        "is_lean_season": 0, "period_days": 91,
        "lag_1": 1.5, "lag_2": 1.6, "lag_3": 1.7,
        "rolling_mean_3": 1.6, "rolling_max_3": 1.8,
        "phase_trend": -0.1,
        "unit_hist_max": 2.0, "crisis_momentum": 0.0,
        "is_cold_start": 0, "lean_x_lag1": 0.0, "lean_x_trend": 0.0,
        "gap_to_crisis": 1.5, "escalation_risk": 0.0,
        "is_ipc2": 1, "preference_rating": 0.8,
    }

    for label, sample in [("High-risk (lean season, cold-start)", HIGH_RISK),
                           ("Low-risk  (harvest season, stable)",  LOW_RISK)]:
        print(f"\n=== {label} ===")
        result = predict_combined(sample)
        print(f"Alert level  : {result['alert_level']}")
        print(f"Binary       : {result['binary']['label']} "
              f"(p={result['binary']['probability']}, thr={result['binary']['threshold']})")
        print(f"Multiclass   : {result['multiclass']['label']}")
        print(f"Probabilities: {result['multiclass']['probabilities']}")
