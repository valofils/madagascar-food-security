# Madagascar Food Security Early Warning System

> Predicting acute food insecurity phases for Madagascar livelihood zones using
> FEWS NET IPC data and XGBoost — deployed as a serverless AWS Lambda function
> behind API Gateway, with a FastAPI local service and Docker support.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project builds an end-to-end machine learning pipeline that ingests IPC
(Integrated Food Security Phase Classification) data from the FEWS NET API,
engineers temporal lag and cold-start interaction features, trains binary and
multiclass XGBoost classifiers with walk-forward cross-validation, and serves
predictions via both a local FastAPI service and a serverless AWS Lambda function.

The system outputs a crisis probability and binary label for any Madagascar
livelihood zone based on its historical food security trajectory.

**Domain context**: Madagascar experiences chronic food insecurity, particularly
in the Grand Sud (Androy, Atsimo Andrefana) and Grand Sud-Est (Befotaka,
Farafangana, Ikongo) regions. The lean season runs October–March. This system
is designed to support humanitarian early warning, not replace IPC technical
working group processes.

---

## Results

### Walk-forward cross-validation (2019–2023)

| Fold | n Crisis | ROC-AUC | Precision | Recall | F1 |
|------|----------|---------|-----------|--------|----|
| 2019 | 11 | 0.990 | 0.556 | 0.909 | 0.690 |
| 2020 | 23 | 0.975 | 0.759 | 0.957 | 0.846 |
| 2021 | 63 | 0.977 | 0.920 | 0.730 | 0.814 |
| 2022 | 79 | 0.987 | 0.716 | 0.987 | 0.830 |
| 2023 | 4  | 0.978 | 0.125 | 1.000 | 0.222 |
| **Mean** | — | **0.981** | **0.615** | **0.917** | **0.680** |

### Held-out test set (2024–2026)

| Model | ROC-AUC | Crisis Recall | Crisis F1 |
|-------|---------|---------------|-----------|
| Binary (Crisis vs Not) | 0.801 | 0.00 | 0.00 |
| Multiclass (P1 / P2 / P3+) | — | 0.00 | 0.00 |

**The model fails on the 2024–2026 test period due to distributional shift** —
Crisis events in 2024/2026 arrive from lower baseline phases (cold-start pattern)
in geographic units not previously seen in Crisis. Walk-forward CV confirms the
model generalises well within its training distribution (recall 0.917). Full
failure analysis with quantified root causes and recommended mitigations is in
[`models/model_card.md`](models/model_card.md).

- **Train**: 2,948 rows, 2016–2023 | **Test**: 759 rows, 2024–2026
- **Features**: 20 (temporal, lag/rolling, cold-start interactions)
- **Threshold**: recall-constrained (≥70%) — missing Crisis is worse than false alarm

---

## Project Structure

```
madagascar-food-security/
├── src/
│   ├── data_ingestion.py   # FEWS NET API fetcher with pagination + retry
│   ├── preprocessing.py    # clean, harmonise IPC scales, scenario split
│   ├── features.py         # lag/rolling/cold-start features (20 total, leak-free)
│   ├── train.py            # XGBoost + SMOTE + walk-forward CV + threshold tuning
│   ├── evaluate.py         # classification report, ROC-AUC, confusion matrix
│   └── predict.py          # predict_combined() → alert_level
├── api/
│   └── main.py             # FastAPI: /health /features /example /predict
├── serverless/
│   ├── lambda_function.py  # AWS Lambda handler (cold-start caching, threshold)
│   ├── Dockerfile          # Lambda container image (python:3.11)
│   ├── requirements.txt    # pinned deps for Lambda layer
│   ├── test_lambda_local.py# 5-case local test suite (no AWS needed)
│   └── DEPLOY.md           # 8-step AWS deployment runbook
├── models/
│   ├── ipc_binary_classifier.pkl       # gitignored
│   ├── ipc_multiclass_classifier.pkl   # gitignored
│   ├── model_card.md                   # full eval + failure analysis
│   └── evaluation_metrics.json
├── tests/
│   ├── test_ingestion.py   # 13 tests — mocked API, pagination, file saving
│   └── test_features.py    # 26 tests — lag, rolling, encoding
├── data/
│   ├── raw/                # FEWS NET API responses (gitignored)
│   └── processed/          # cleaned CSVs + evaluation plots
├── Dockerfile              # FastAPI container
├── run_pipeline.py         # pipeline orchestrator (CLI)
├── config.py               # paths, API URLs, constants
└── requirements.txt
```

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/valofils/madagascar-food-security.git
cd madagascar-food-security

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Full pipeline: ingest → preprocess → features → train → evaluate
python run_pipeline.py

# Or run individual stages
python run_pipeline.py --steps train evaluate

# Skip ingestion if data is already fresh
python run_pipeline.py --skip ingest
```

### 3. Serve the API locally

```bash
PYTHONPATH=$(pwd) uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 4. Run with Docker

```bash
docker build -t madagascar-food-security .
docker run -p 8000:8000 madagascar-food-security
```

### 5. Run tests

```bash
pytest tests/ -v
# 39 passed in ~0.8s
```

### 6. Test the Lambda handler locally

```bash
python serverless/test_lambda_local.py
# 5 passed — no AWS account needed
```

---

## API Reference

### Local FastAPI (`http://localhost:8000`)

#### `POST /predict`

**Request body** (20 features required):

```json
{
  "year": 2024, "month": 1, "quarter": 1,
  "is_lean_season": 1, "period_days": 90,
  "lag_1": 2.5, "lag_2": 2.2, "lag_3": 2.0,
  "rolling_mean_3": 2.3, "rolling_max_3": 2.5,
  "phase_trend": 0.3,
  "unit_hist_max": 3.0, "crisis_momentum": 1.0,
  "is_cold_start": 1, "lean_x_lag1": 2.5, "lean_x_trend": 0.3,
  "gap_to_crisis": 0.5, "escalation_risk": 3.45,
  "is_ipc2": 0, "preference_rating": 1.0
}
```

**Response**:

```json
{
  "alert_level": "HIGH",
  "binary": {
    "prediction": 1,
    "label": "Crisis or worse (Phase 3+)",
    "probability": 0.82,
    "threshold": 0.45,
    "is_crisis": true
  },
  "multiclass": {
    "prediction": 2,
    "label": "Crisis+ (P3+)",
    "probabilities": {
      "Minimal (P1)": 0.06,
      "Stressed (P2)": 0.12,
      "Crisis+ (P3+)": 0.82
    }
  }
}
```

### Serverless Lambda (`POST /predict` via API Gateway)

Same request body and response schema as above, minus `alert_level`.
See [`serverless/DEPLOY.md`](serverless/DEPLOY.md) for deployment instructions.

---

## Feature Engineering

20 leak-free features in four groups:

| Group | Features |
|-------|---------|
| Temporal | `year`, `month`, `quarter`, `is_lean_season`, `period_days` |
| Lag / rolling | `lag_1`, `lag_2`, `lag_3`, `rolling_mean_3`, `rolling_max_3`, `phase_trend`, `unit_hist_max`, `crisis_momentum` |
| Cold-start interactions | `is_cold_start`, `lean_x_lag1`, `lean_x_trend`, `gap_to_crisis`, `escalation_risk` |
| Categorical | `is_ipc2`, `preference_rating` |

Features removed in v1.1: `unit_mean_phase`, `unit_pct_crisis`, `unit_max_phase`,
`unit_code` — all computed on the full dataset, causing target leakage into training rows.

---

## Data

**Source**: [FEWS NET Data Warehouse API](https://fdw.fews.net/api/)  
**Endpoint**: `/api/ipcphase/?country_code=MG`  
**Auth**: none required for public data

| Attribute | Value |
|-----------|-------|
| Country | Madagascar (MG) |
| Date range | Feb 2016 – Jul 2026 |
| Geographic units | 312 livelihood zones (fnids) |
| Scenarios | CS / ML1 / ML2 |
| IPC scales | IPC 2.0, 3.0, 3.1 → harmonised |

Raw data files are gitignored. Run `python run_pipeline.py --steps ingest` to
fetch fresh data from the API.

---

## Zoomcamp Module Mapping

Built following the [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) curriculum:

| Module | Topic | Status |
|--------|-------|--------|
| 1 | Intro & problem framing | ✅ |
| 2 | Regression / EDA | ✅ via preprocessing |
| 3 | Classification (logistic baseline) | ⏭ skipped → went straight to XGBoost |
| 4 | Evaluation metrics | ✅ |
| 5 | Deployment (FastAPI + Docker) | ✅ |
| 6 | Decision trees & XGBoost | ✅ |
| 8 | Deep learning (LSTM) | ✅ |
| 9 | Serverless (AWS Lambda + API Gateway) | ✅ |
| 10 | Kubernetes | ⏳ future |

---

## Author

**Mariel**  
MSc Mathematics, AIMS Ghana (2026)  
IPC Level I Analyst | M&E Specialist | Data Scientist  
GitHub: [@valofils](https://github.com/valofils)