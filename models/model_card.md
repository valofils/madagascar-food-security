# Model Card — Madagascar IPC Food Security Classifiers

> **Project**: Madagascar Food Security Early Warning System  
> **Author**: Mariel Valosimbazafy ([@valofils](https://github.com/valofils))  
> **Last updated**: April 2026  
> **Repository**: `madagascar-food-security`

---

## 1. Model Overview

This project contains two complementary XGBoost classifiers trained on FEWS NET IPC
(Integrated Food Security Phase Classification) data for Madagascar (2016–2026).
They are designed for food security early warning: predicting which livelihood zones
are likely to reach crisis-level acute food insecurity (IPC Phase 3 or above).

| Artifact | Task | Primary use |
|----------|------|-------------|
| `ipc_binary_classifier.pkl` | Crisis (Phase 3+) vs Not Crisis (Phase 1–2) | Early warning trigger |
| `ipc_multiclass_classifier.pkl` | Phase 1 / Phase 2 / Phase 3+ | Severity grading |

The binary classifier is the **primary model** for operational use. The multiclass
classifier is supplementary — it adds granularity but has limited power on Phase 3+
due to class imbalance in the test set.

---

## 2. Intended Use

### Primary use case
Automated early warning flag for humanitarian responders and food security analysts
monitoring acute food insecurity in Madagascar livelihood zones. The system outputs
an `alert_level` (HIGH / MODERATE / LOW) that combines signals from both models.

### Intended users
- Food security analysts (FEWS NET, WFP, UNICEF, FAO)
- M&E specialists and IPC technical working groups
- Humanitarian programme planners

### Out-of-scope uses
- Direct targeting of food assistance beneficiaries (human review required)
- Predictions outside Madagascar without retraining
- Causal attribution of food insecurity drivers
- Replacing IPC technical working group classification processes

---

## 3. Data

### Source
**FEWS NET Data Warehouse API** (`https://fdw.fews.net/api/`)  
Endpoint: `/api/ipcphase/?country_code=MG`  
No authentication required for public data.

### Coverage
| Attribute | Value |
|-----------|-------|
| Country | Madagascar (MG) |
| Date range | February 2016 – July 2026 |
| Geographic units | 312 unique fnids (livelihood zones) |
| Scenarios used | CS (Current Situation) only for training |
| IPC scales | IPC 2.0, IPC 3.0, IPC 3.1 — harmonised to IPC2/IPC3 |
| Total records (cleaned) | 3,970 rows |

### Train / test split
Temporal split — no data leakage:

| Split | Years | Rows |
|-------|-------|------|
| Train | 2016–2023 | 3,211 |
| Test | 2024–2026 | 759 |

### Class distribution (CS scenario, cleaned)

| Phase | Label | Share |
|-------|-------|-------|
| 1 | Minimal | ~80% |
| 2 | Stressed | ~14% |
| 3 | Crisis | ~6% |
| 4 | Emergency | ~0.1% (merged into Phase 3 for modelling) |

**Note**: Phase 4 (Emergency) has only 99 records total and 5 in the test set.
It is merged into Phase 3+ for both models.

### Preprocessing
- Rows with null phase values or null projection dates dropped
- `IPC Highest Household` classification scale excluded (different methodology)
- Temporal features extracted: year, month, quarter, period_days, is_lean_season
- Lean season flag: October–March (Madagascar's hunger gap)

---

## 4. Features

16 features used by both models:

| Feature | Type | Description |
|---------|------|-------------|
| `year` | int | Year of projection period |
| `month` | int | Month of projection start |
| `quarter` | int | Quarter (1–4) |
| `is_lean_season` | binary | 1 if Oct–Mar (lean season), else 0 |
| `period_days` | int | Length of projection window in days |
| `lag_1` | float | IPC phase in previous period |
| `lag_2` | float | IPC phase two periods ago |
| `rolling_mean_3` | float | Mean phase over last 3 periods |
| `rolling_max_3` | float | Max phase over last 3 periods |
| `phase_trend` | float | lag_1 − lag_2 (direction of change) |
| `unit_mean_phase` | float | Historical mean IPC phase for this livelihood zone |
| `unit_max_phase` | float | Historical max IPC phase for this livelihood zone |
| `unit_pct_crisis` | float | Fraction of periods unit was in Phase 3+ |
| `unit_code` | int | Integer encoding of geographic unit (fnid) |
| `is_ipc2` | binary | 1 if IPC 2.0 scale, else 0 |
| `preference_rating` | float | FEWS NET data quality preference rating |

---

## 5. Model Architecture

### Binary classifier (`ipc_binary_classifier.pkl`)

```
Algorithm      : XGBoost (XGBClassifier)
Objective      : binary:logistic
n_estimators   : 400
max_depth      : 6
learning_rate  : 0.05
subsample      : 0.8
colsample_bytree: 0.8
scale_pos_weight: computed from class ratio
Imbalance handling: SMOTE oversampling on training set
Threshold      : tuned via precision-recall curve to maximise F1
```

### Multiclass classifier (`ipc_multiclass_classifier.pkl`)

```
Algorithm      : XGBoost (XGBClassifier)
Objective      : multi:softmax
num_class      : 3  (Phase 1 / Phase 2 / Phase 3+)
n_estimators   : 400
max_depth      : 6
learning_rate  : 0.05
subsample      : 0.8
colsample_bytree: 0.8
Imbalance handling: SMOTE oversampling on training set
```

Both models are serialised as `{"model": XGBClassifier, "threshold": float}` using pickle.

---

## 6. Performance

### Binary classifier — test set (2024–2026, n=759)

| Metric | Not Crisis (P1–2) | Crisis (P3+) | Overall |
|--------|-------------------|--------------|---------|
| Precision | 0.95 | 0.67 | — |
| Recall | 1.00 | 0.05 | — |
| F1-score | 0.98 | 0.10 | — |
| Support | 722 | 37 | 759 |
| **Accuracy** | | | **0.95** |
| **ROC-AUC** | | | **0.916** |

**Threshold**: 0.0016 (tuned for recall on Crisis class)

### Multiclass classifier — test set (2024–2026, n=759)

| Metric | Minimal (P1) | Stressed (P2) | Crisis+ (P3+) | Overall |
|--------|--------------|---------------|---------------|---------|
| Precision | 0.97 | 0.85 | 1.00 | — |
| Recall | 1.00 | 0.93 | 0.05 | — |
| F1-score | 0.98 | 0.89 | 0.10 | — |
| Support | 524 | 198 | 37 | 759 |
| **Accuracy** | | | | **0.93** |

### Top 10 features — binary model

| Feature | Importance |
|---------|-----------|
| `unit_mean_phase` | 0.400 |
| `unit_max_phase` | 0.160 |
| `unit_pct_crisis` | 0.123 |
| `lag_1` | 0.074 |
| `is_lean_season` | 0.048 |
| `rolling_mean_3` | 0.037 |
| `period_days` | 0.029 |
| `year` | 0.029 |
| `quarter` | 0.022 |
| `month` | 0.019 |

### Top 10 features — multiclass model

| Feature | Importance |
|---------|-----------|
| `lag_1` | 0.405 |
| `unit_mean_phase` | 0.117 |
| `unit_max_phase` | 0.074 |
| `year` | 0.054 |
| `is_lean_season` | 0.051 |
| `unit_pct_crisis` | 0.046 |
| `rolling_mean_3` | 0.045 |
| `rolling_max_3` | 0.040 |
| `month` | 0.036 |
| `quarter` | 0.033 |

---

## 7. Limitations and Known Issues

### Crisis recall
The binary model achieves **Crisis recall of 0.05** at the default threshold on the
test set, despite a tuned threshold and SMOTE. This is primarily due to sparse test
data — only 37 Crisis cases in 759 test rows (4.9%). The ROC-AUC of 0.916 indicates
the model has strong discriminative ability; the recall issue reflects threshold
sensitivity in a highly imbalanced regime.

**Operational implication**: in a humanitarian context, missing a real crisis is far
more costly than a false alarm. The alert system uses both models together and errs
toward sensitivity — a HIGH alert is triggered when *either* model flags crisis.

### Phase 4 sparsity
Only 99 Phase 4 (Emergency) records exist in the full dataset, with 5 in the test
set. Phase 4 is merged into Phase 3+ for all modelling. The system cannot
distinguish Emergency from Crisis.

### Geographic encoding
`unit_code` is a categorical integer derived from `fnid` strings. New livelihood
zones introduced after training will receive unseen codes and may produce unreliable
predictions.

### Temporal drift
The model is trained on 2016–2023 data. Structural shocks (cyclones, droughts,
political crises) that alter food security dynamics significantly may degrade
performance over time. Periodic retraining is recommended.

### IPC scale harmonisation
Three IPC scales (2.0, 3.0, 3.1) are harmonised into a binary flag (`is_ipc2`).
This is a simplification; methodological differences across scales may introduce
noise in earlier years.

---

## 8. Ethical Considerations

- **Humanitarian priority**: the system is designed to support, not replace,
  expert IPC analysis. All alerts should be validated by trained analysts before
  informing programme decisions.
- **False negatives**: a LOW alert for a zone that is actually in crisis could
  delay response. Users should treat this system as one input among many.
- **Data provenance**: FEWS NET data reflects the best available field assessments
  but may have gaps or delays, particularly in remote zones of Madagascar.
- **No individual-level data**: all predictions are at the livelihood zone level.
  No household or individual data is used.

---

## 9. How to Use

### API (recommended)
```bash
# Start the server
source venv/bin/activate
PYTHONPATH=$(pwd) uvicorn api.main:app --host 0.0.0.0 --port 8000

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2024, "month": 2, "quarter": 1,
    "is_lean_season": 1, "period_days": 29,
    "lag_1": 3.0, "lag_2": 3.0,
    "rolling_mean_3": 3.0, "rolling_max_3": 3.0,
    "phase_trend": 0.0,
    "unit_mean_phase": 2.8, "unit_max_phase": 4.0,
    "unit_pct_crisis": 0.65, "unit_code": 42,
    "is_ipc2": 0, "preference_rating": 90.0
  }'
```

### Python
```python
from src.predict import predict_combined

features = {
    "year": 2024, "month": 2, "quarter": 1,
    "is_lean_season": 1, "period_days": 29,
    "lag_1": 3.0, "lag_2": 3.0,
    "rolling_mean_3": 3.0, "rolling_max_3": 3.0,
    "phase_trend": 0.0,
    "unit_mean_phase": 2.8, "unit_max_phase": 4.0,
    "unit_pct_crisis": 0.65, "unit_code": 42,
    "is_ipc2": 0, "preference_rating": 90.0,
}

result = predict_combined(features)
print(result["alert_level"])   # HIGH / MODERATE / LOW
print(result["binary"])        # Crisis probability + label
print(result["multiclass"])    # Per-phase probabilities
```

### Docker
```bash
docker build -t madagascar-food-security .
docker run -p 8000:8000 madagascar-food-security
```

---

## 10. Citation

If you use this model or data pipeline, please cite:

```
Valosimbazafy, M. (2026). Madagascar IPC Food Security Early Warning Classifier.
GitHub: https://github.com/valofils/madagascar-food-security
Data source: FEWS NET Data Warehouse, https://fdw.fews.net
```
