import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_combined, FEATURE_COLS

app = FastAPI(
    title="Madagascar IPC Food Security Classifier",
    description=(
        "Predicts IPC acute food insecurity phase for Madagascar livelihood zones "
        "using FEWS NET historical data (2016–2026). Returns binary Crisis/Not-Crisis "
        "and multiclass P1/P2/P3+ predictions. Model v1.1 — 20 leak-free features."
    ),
    version="1.1.0",
)


# ── Request schema (v1.1 — 20 features) ───────────────────────────────────────
class PredictionRequest(BaseModel):
    # Temporal
    year:             int   = Field(..., ge=2016, le=2030, example=2024)
    month:            int   = Field(..., ge=1,    le=12,   example=1)
    quarter:          int   = Field(..., ge=1,    le=4,    example=1)
    is_lean_season:   int   = Field(..., ge=0,    le=1,    example=1)
    period_days:      int   = Field(..., ge=1,    le=366,  example=90)

    # Lag / rolling (shift(1) — no current-period leakage)
    lag_1:            float = Field(..., ge=1, le=5, example=2.5)
    lag_2:            float = Field(..., ge=1, le=5, example=2.2)
    lag_3:            float = Field(..., ge=1, le=5, example=2.0)
    rolling_mean_3:   float = Field(..., ge=1, le=5, example=2.3)
    rolling_max_3:    float = Field(..., ge=1, le=5, example=2.5)
    phase_trend:      float = Field(...,             example=0.3)
    unit_hist_max:    float = Field(..., ge=1, le=5, example=3.0,
                          description="Expanding historical max phase for this unit (past obs only)")
    crisis_momentum:  float = Field(..., ge=0, le=4, example=1.0,
                          description="Count of Crisis periods in previous 4 observations")

    # Cold-start interaction features
    is_cold_start:    int   = Field(..., ge=0, le=1,  example=1,
                          description="1 if lag_1 < 2.5 (previous period below Crisis threshold)")
    lean_x_lag1:      float = Field(..., ge=0,        example=2.5,
                          description="is_lean_season × lag_1")
    lean_x_trend:     float = Field(..., ge=0,        example=0.3,
                          description="is_lean_season × max(phase_trend, 0)")
    gap_to_crisis:    float = Field(..., ge=0,        example=0.5,
                          description="max(3.0 - lag_1, 0) — distance below Crisis threshold")
    escalation_risk:  float = Field(..., ge=0,        example=3.45,
                          description="rolling_mean_3 × is_lean_season × (1 + max(phase_trend,0))")

    # Categorical
    is_ipc2:          int   = Field(..., ge=0, le=1,   example=0)
    preference_rating: float = Field(..., ge=0, le=100, example=1.0)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {
        "service": "Madagascar IPC Food Security Classifier",
        "version": "1.1.0",
        "status":  "ok",
        "docs":    "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/features", tags=["info"])
def list_features():
    """Return the list of required input features and their count."""
    return {"features": FEATURE_COLS, "count": len(FEATURE_COLS)}


@app.get("/example", tags=["info"])
def example_request():
    """
    Return an example request body for a high-risk Grand Sud profile
    during lean season with cold-start escalation signal.
    """
    return {
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


@app.post("/predict", tags=["prediction"])
def predict(request: PredictionRequest):
    """
    Predict IPC food insecurity phase for a Madagascar livelihood zone.

    Returns:
    - **alert_level**: HIGH / MODERATE / LOW
    - **binary**: Crisis (Phase 3+) vs Not Crisis with probability and threshold
    - **multiclass**: Phase 1 / 2 / 3+ with per-class probabilities

    Alert level logic:
    - `HIGH` — both models agree on Crisis
    - `MODERATE` — binary model flags Crisis, multiclass disagrees
    - `LOW` — neither model flags Crisis

    Note: The classification threshold is optimised for recall ≥ 70% on the
    validation set (humanitarian context — missing Crisis is worse than false alarm).
    The model is known to underperform on cold-start Crisis in units with no prior
    Crisis history. See model card for full failure analysis.
    """
    try:
        features = request.model_dump()
        result   = predict_combined(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
