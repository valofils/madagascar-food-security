import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.predict import predict_combined, FEATURE_COLS

app = FastAPI(
    title="Madagascar IPC Food Security Classifier",
    description=(
        "Predicts IPC acute food insecurity phase for Madagascar "
        "livelihood zones using FEWS NET historical data (2016-2026). "
        "Returns binary Crisis/Not-Crisis and multiclass P1/P2/P3+ predictions."
    ),
    version="1.0.0",
)


# ── request schema ────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    year:             int   = Field(..., ge=2016, le=2030, example=2024)
    month:            int   = Field(..., ge=1,    le=12,   example=2)
    quarter:          int   = Field(..., ge=1,    le=4,    example=1)
    is_lean_season:   int   = Field(..., ge=0,    le=1,    example=1)
    period_days:      int   = Field(..., ge=1,    le=366,  example=29)
    lag_1:            float = Field(..., ge=1,    le=5,    example=3.0)
    lag_2:            float = Field(..., ge=1,    le=5,    example=3.0)
    rolling_mean_3:   float = Field(..., ge=1,    le=5,    example=3.0)
    rolling_max_3:    float = Field(..., ge=1,    le=5,    example=3.0)
    phase_trend:      float = Field(...,                   example=0.0)
    unit_mean_phase:  float = Field(..., ge=1,    le=5,    example=2.8)
    unit_max_phase:   float = Field(..., ge=1,    le=5,    example=4.0)
    unit_pct_crisis:  float = Field(..., ge=0,    le=1,    example=0.65)
    unit_code:        int   = Field(..., ge=0,             example=42)
    is_ipc2:          int   = Field(..., ge=0,    le=1,    example=0)
    preference_rating: float = Field(..., ge=0,   le=100,  example=90.0)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "service": "Madagascar IPC Food Security Classifier",
        "version": "1.0.0",
        "status":  "ok",
        "docs":    "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/features", tags=["info"])
def list_features():
    return {"features": FEATURE_COLS, "count": len(FEATURE_COLS)}


@app.get("/example", tags=["info"])
def example_request():
    return {
        "year": 2024, "month": 2, "quarter": 1,
        "is_lean_season": 1, "period_days": 29,
        "lag_1": 3.0, "lag_2": 3.0,
        "rolling_mean_3": 3.0, "rolling_max_3": 3.0,
        "phase_trend": 0.0,
        "unit_mean_phase": 2.8, "unit_max_phase": 4.0,
        "unit_pct_crisis": 0.65, "unit_code": 42,
        "is_ipc2": 0, "preference_rating": 90.0,
    }


@app.post("/predict", tags=["prediction"])
def predict(request: PredictionRequest):
    """
    Predict IPC food insecurity phase for a Madagascar livelihood zone.

    Returns:
    - **alert_level**: HIGH / MODERATE / LOW
    - **binary**: Crisis (Phase 3+) vs Not Crisis with probability
    - **multiclass**: Phase 1 / 2 / 3+ with per-class probabilities
    """
    try:
        features = request.model_dump()
        result   = predict_combined(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
