"""
AWS Lambda handler — IPC Food Security Binary Classifier v1.1
20 features: temporal + lag + cold-start interaction features.

Input schema:
{
    "year":             2024,
    "month":            1,
    "quarter":          1,
    "is_lean_season":   1,
    "period_days":      90,
    "lag_1":            2.1,
    "lag_2":            1.9,
    "lag_3":            2.0,
    "rolling_mean_3":   2.0,
    "rolling_max_3":    2.3,
    "phase_trend":      0.2,
    "unit_hist_max":    3.0,
    "crisis_momentum":  0.0,
    "is_cold_start":    1,
    "lean_x_lag1":      2.1,
    "lean_x_trend":     0.2,
    "gap_to_crisis":    0.9,
    "escalation_risk":  2.52,
    "is_ipc2":          0,
    "preference_rating": 1.0
}
"""

import json
import os
import pickle
import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model/ipc_binary_classifier.pkl")
META_PATH  = os.environ.get("META_PATH",  "/opt/ml/model/ipc_binary_classifier_meta.json")

_model     = None
_meta      = None
_threshold = 0.5

FEATURE_NAMES = [
    "year", "month", "quarter", "is_lean_season", "period_days",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_max_3",
    "phase_trend",
    "unit_hist_max", "crisis_momentum",
    "is_cold_start", "lean_x_lag1", "lean_x_trend",
    "gap_to_crisis", "escalation_risk",
    "is_ipc2", "preference_rating",
]


def _load_model():
    global _model, _meta, _threshold
    if _model is None:
        logger.info("Cold start — loading model from %s", MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            estimator = raw.get("model") or raw.get("classifier")
            if estimator is None:
                for v in raw.values():
                    if hasattr(v, "predict"):
                        estimator = v; break
            if estimator is None:
                raise RuntimeError(f"No estimator in pickle. Keys: {list(raw.keys())}")
            _model     = estimator
            _threshold = float(raw.get("threshold", 0.5))
        else:
            _model = raw; _threshold = 0.5
        with open(META_PATH, "r") as f:
            _meta = json.load(f)
        logger.info("Model ready — %s | threshold=%.4f", type(_model).__name__, _threshold)
    return _model, _meta, _threshold


def _build_feature_vector(payload: dict) -> pd.DataFrame:
    missing = [f for f in FEATURE_NAMES if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    try:
        row = {f: float(payload[f]) for f in FEATURE_NAMES}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric value: {exc}") from exc
    if not (1 <= row["month"] <= 12):
        raise ValueError(f"month must be 1-12, got {row['month']}")
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def _ok(body, status=200):
    return {"statusCode": status,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(body)}

def _error(message, status=400):
    return {"statusCode": status,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": message})}


def lambda_handler(event, context):
    logger.info("Event: %s", json.dumps(event))
    try:
        if "body" in event:
            raw = event["body"]
            payload = json.loads(raw) if isinstance(raw, str) else raw
        else:
            payload = event
    except (json.JSONDecodeError, TypeError) as exc:
        return _error(f"Invalid JSON: {exc}")

    try:
        features = _build_feature_vector(payload)
    except ValueError as exc:
        logger.error("Feature error: %s", exc)
        return _error(str(exc))

    try:
        model, meta, threshold = _load_model()
        prob_crisis = float(model.predict_proba(features)[0][1])
        prediction  = int(prob_crisis >= threshold)
    except Exception as exc:
        logger.exception("Prediction failed")
        return _error(f"Prediction error: {exc}", status=500)

    label = "crisis_or_above" if prediction == 1 else "no_crisis"
    return _ok({
        "prediction":         prediction,
        "label":              label,
        "probability_crisis": round(prob_crisis, 4),
        "threshold_used":     round(threshold, 4),
        "model_version":      meta.get("model_version", "1.1.0"),
    })
