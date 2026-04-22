# Madagascar Food Security Early Warning System

> Predicting acute food insecurity phases for Madagascar livelihood zones using
> FEWS NET IPC data and XGBoost — deployed as a FastAPI service with Docker.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project builds an end-to-end machine learning pipeline that ingests IPC
(Integrated Food Security Phase Classification) data from the FEWS NET API,
engineers temporal and geographic features, trains binary and multiclass XGBoost
classifiers, and serves predictions via a REST API.

The system outputs an `alert_level` — **HIGH**, **MODERATE**, or **LOW** — for
any Madagascar livelihood zone based on its historical food security trajectory.

**Domain context**: Madagascar experiences chronic food insecurity, particularly
in the Grand Sud (Androy, Atsimo Andrefana) and Grand Sud-Est (Befotaka,
Farafangana, Ikongo) regions. The lean season runs October–March. This system
is designed to support humanitarian early warning, not replace IPC technical
working group processes.

---

## Results

| Model | Accuracy | ROC-AUC | Crisis F1 |
|-------|----------|---------|-----------|
| Binary (Crisis vs Not) | 0.95 | **0.916** | 0.10 |
| Multiclass (P1 / P2 / P3+) | 0.93 | — | 0.10 |

- **Train**: 3,211 rows, 2016–2023
- **Test**: 759 rows, 2024–2026 (temporal split, no leakage)
- Top features: `unit_mean_phase`, `unit_max_phase`, `unit_pct_crisis`, `lag_1`
- Crisis recall is low due to sparse test cases (37/759); ROC-AUC of 0.916
  confirms strong discriminative ability

See [`models/model_card.md`](models/model_card.md) for full evaluation details.

---

## Project Structure

```
madagascar-food-security/
├── src/
│   ├── data_ingestion.py   # FEWS NET API fetcher with pagination + retry
│   ├── preprocessing.py    # clean, harmonise IPC scales, scenario split
│   ├── features.py         # lag/rolling/unit features + categorical encoding
│   ├── train.py            # XGBoost + SMOTE + threshold tuning
│   ├── evaluate.py         # classification report, ROC-AUC, confusion matrix
│   └── predict.py          # predict_combined() → alert_level
├── api/
│   └── main.py             # FastAPI: /health /features /example /predict
├── models/
│   ├── ipc_binary_classifier.pkl
│   ├── ipc_multiclass_classifier.pkl
│   ├── model_card.md
│   └── evaluation_metrics.json
├── tests/
│   ├── test_ingestion.py   # 13 tests — mocked API, pagination, file saving
│   └── test_features.py    # 26 tests — lag, rolling, unit stats, encoding
├── data/
│   ├── raw/                # FEWS NET API responses (gitignored)
│   └── processed/          # cleaned CSVs + evaluation plots
├── Dockerfile
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

# List all available stages
python run_pipeline.py --list
```

### 3. Serve the API

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

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /health`
```json
{"status": "ok"}
```

### `GET /features`
Returns the list of 16 required input features.

### `GET /example`
Returns an example request body for a high-risk Grand Sud profile.

### `POST /predict`

**Request body** (all fields required):

```json
{
  "year": 2024,
  "month": 2,
  "quarter": 1,
  "is_lean_season": 1,
  "period_days": 29,
  "lag_1": 3.0,
  "lag_2": 3.0,
  "rolling_mean_3": 3.0,
  "rolling_max_3": 3.0,
  "phase_trend": 0.0,
  "unit_mean_phase": 2.8,
  "unit_max_phase": 4.0,
  "unit_pct_crisis": 0.65,
  "unit_code": 42,
  "is_ipc2": 0,
  "preference_rating": 90.0
}
```

**Response**:

```json
{
  "alert_level": "HIGH",
  "binary": {
    "prediction": 1,
    "label": "Crisis or worse (Phase 3+)",
    "probability": 0.1274,
    "threshold": 0.0016,
    "is_crisis": true
  },
  "multiclass": {
    "prediction": 2,
    "label": "Crisis+ (P3+)",
    "probabilities": {
      "Minimal (P1)": 0.0006,
      "Stressed (P2)": 0.3919,
      "Crisis+ (P3+)": 0.6075
    }
  },
  "features_used": { "...": "..." }
}
```

**Alert level logic**:
- `HIGH` — both models agree on crisis
- `MODERATE` — binary model flags crisis, multiclass disagrees
- `LOW` — neither model flags crisis

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

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| `ingest` | `src/data_ingestion.py` | Fetch raw data from FEWS NET API |
| `preprocess` | `src/preprocessing.py` | Clean, harmonise, split by scenario |
| `features` | `src/features.py` | Lag, rolling, unit features + encoding |
| `train` | `src/train.py` | XGBoost + SMOTE + threshold tuning |
| `evaluate` | `src/evaluate.py` | Metrics, plots, feature importance |

---

## Zoomcamp Module Mapping

This project was built following the
[ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) curriculum:

| Module | Topic | Status |
|--------|-------|--------|
| 1 | Intro & problem framing | ✅ |
| 2 | Regression / EDA | ✅ via preprocessing |
| 3 | Classification (logistic baseline) | ⏭ skipped → went straight to XGBoost |
| 4 | Evaluation metrics | ✅ |
| 5 | Deployment (FastAPI + Docker) | ✅ |
| 6 | Decision trees & XGBoost | ✅ |
| 8 | Deep learning | ⏳ future |
| 9 | Serverless (AWS Lambda) | ⏳ future |
| 10 | Kubernetes | ⏳ future |

---

## Author

**Mariel Valosimbazafy**  
MSc Mathematics, AIMS Ghana (2026)  
IPC Level I Analyst | M&E Specialist | Data Scientist  
GitHub: [@valofils](https://github.com/valofils)
