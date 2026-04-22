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
a predicted IPC phase and crisis probability that informs — but does not replace —
analyst judgement.

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
FEWS NET IPC phase data for Madagascar, accessed via the FEWS NET Data Warehouse API.
Covers current situation (CS) assessments from 2016 to early 2026.

### Coverage
- **Geography**: Madagascar livelihood zones (fnid-level)
- **Period**: 2016–2026 (current situation assessments only)
- **Rows after feature engineering**: 3,707 observations
- **Class distribution**: Phase 1: 80%, Phase 2: 14%, Phase 3+: 6%

### Train / test split
Temporal split: train on 2016–2023, test on 2024–2026. This mirrors operational
deployment — the model always predicts forward in time from its training window.

---

## 4. Features

The model uses 20 leak-free features grouped into four categories:

### Temporal context
| Feature | Description |
|---------|-------------|
| `year`, `month`, `quarter` | Observation date components |
| `is_lean_season` | Binary flag for lean season months |
| `period_days` | Length of IPC assessment period |

### Lag and rolling features *(shift(1) — no current-period leakage)*
| Feature | Description |
|---------|-------------|
| `lag_1`, `lag_2`, `lag_3` | IPC phase in previous 1, 2, 3 periods |
| `rolling_mean_3` | Mean phase over previous 3 periods |
| `rolling_max_3` | Max phase over previous 3 periods |
| `phase_trend` | `lag_1 - lag_2` (direction of change) |
| `unit_hist_max` | Expanding historical max phase per unit (past only) |
| `crisis_momentum` | Count of Crisis periods in previous 4 observations |

### Cold-start interaction features *(targeting sudden escalation)*
| Feature | Description |
|---------|-------------|
| `is_cold_start` | 1 if `lag_1 < 2.5` (previous period below Crisis) |
| `lean_x_lag1` | Lean season × lag_1 (amplified risk) |
| `lean_x_trend` | Lean season × positive trend |
| `gap_to_crisis` | Distance of previous phase below Phase 3 threshold |
| `escalation_risk` | Composite: `rolling_mean × lean_season × (1 + positive_trend)` |

### Categorical
| Feature | Description |
|---------|-------------|
| `is_ipc2` | 1 if IPC Phase 2 scale area |
| `preference_rating` | FEWS NET data quality/preference score |

### Features deliberately excluded
| Feature | Reason |
|---------|--------|
| `unit_mean_phase` | Computed on full dataset — target leakage |
| `unit_pct_crisis` | Computed on full dataset — target leakage |
| `unit_max_phase` | Computed on full dataset — target leakage |
| `unit_code` | Geographic identity, not conditions; fails on unseen units |

---

## 5. Training Pipeline

### Class imbalance
Crisis cases represent ~7% of training data. SMOTE (Synthetic Minority Oversampling)
is applied to the training set only — never to validation or test data.

### Threshold selection
The default 0.5 classification threshold is replaced by a recall-constrained
optimum: the highest-F1 threshold where recall ≥ 70% on a stratified validation
split (20% of training data). This reflects the humanitarian principle that
**missing a Crisis is worse than a false alarm**.

### Validation strategy
Walk-forward cross-validation across 5 test years (2019–2023), training on all
prior years each time. This provides a realistic estimate of generalisation across
different Crisis patterns.

---

## 6. Performance

### Walk-forward cross-validation (2019–2023)

| Fold (test year) | n Crisis | ROC-AUC | Precision | Recall | F1 |
|-----------------|----------|---------|-----------|--------|----|
| 2019 | 11 | 0.990 | 0.556 | 0.909 | 0.690 |
| 2020 | 23 | 0.975 | 0.759 | 0.957 | 0.846 |
| 2021 | 63 | 0.977 | 0.920 | 0.730 | 0.814 |
| 2022 | 79 | 0.987 | 0.716 | 0.987 | 0.830 |
| 2023 | 4  | 0.978 | 0.125 | 1.000 | 0.222 |
| **Mean** | — | **0.981** | **0.615** | **0.917** | **0.680** |

Within the training distribution the model generalises well, achieving 91.7% recall
on Crisis cases at a mean precision of 61.5%.

### Held-out test set (2024–2026)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.801 |
| Crisis precision | 0.00 |
| Crisis recall | 0.00 |
| Crisis F1 | 0.00 |
| Non-crisis accuracy | 95% |

**The model fails to detect Crisis cases in the 2024–2026 test period.**
See Section 7 for root cause analysis.

---

## 7. Failure Analysis — Distributional Shift (2024–2026)

### What went wrong
The binary classifier assigns near-zero probabilities to 75% of actual Crisis cases
in the test set (median p = 0.0006). No threshold adjustment can recover from this —
the model has not learned the 2024/2026 Crisis signature.

### Root cause: temporal distributional shift
Crisis events in the training period (peak: 2021–2022) and test period (2024–2026)
have structurally different feature profiles:

| Feature | Train Crisis mean | Test Crisis mean | Δ |
|---------|-----------------|-----------------|---|
| `lag_1` | 2.77 | 2.08 | −0.69 |
| `rolling_mean_3` | 2.64 | 1.97 | −0.67 |
| `rolling_max_3` | 2.84 | 2.14 | −0.70 |
| `is_lean_season` | 0.78 | 1.00 | +0.22 |

Training Crisis events followed prolonged high-phase periods (persistent escalation).
Test Crisis events arrive from lower baselines — a cold-start pattern — and all occur
during lean season. The model learned "Crisis follows Crisis" and cannot generalise
to first-occurrence escalation.

### Geographic novelty
6 of 22 test Crisis units (27%) never appeared in Crisis during training. The model
has no phase history for these units in a crisis state.

### Year 2023 anomaly
2023 had only 4 Crisis cases (0.9% rate) — a recovery period following the 2022 peak.
This created a gap in crisis examples that the walk-forward CV flagged (F1=0.222 for
the 2023 fold) but could not compensate for.

### What this is not
This failure was not caused by:
- Data leakage (removed in v1.1)
- Threshold miscalibration (tested exhaustively)
- Feature engineering errors (walk-forward CV confirms within-sample generalisation)

### Recommended mitigations
1. **Annual retraining** as new IPC cycles complete — the 2024/2026 data should be
   incorporated into training once a subsequent test year is available
2. **External covariates** — NDVI anomaly, rainfall departure, market price indices,
   and conflict events are known drivers of cold-start Crisis that are independent
   of phase history
3. **Regional sub-models** — separate models per livelihood zone cluster may better
   capture heterogeneous escalation patterns
4. **Ensemble with rule-based triggers** — a simple rule (lean season AND lag_1 > 1.8
   AND phase_trend > 0) could catch cold-start cases the ML model misses

---

## 8. Deployment

The binary classifier is deployed as an AWS Lambda function behind API Gateway.
See `serverless/DEPLOY.md` for the full deployment runbook.

### API input (20 features)
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

### API output
```json
{
  "prediction": 1,
  "label": "crisis_or_above",
  "probability_crisis": 0.78,
  "threshold_used": 0.45,
  "model_version": "1.1.0"
}
```

### Operational guidance
- `probability_crisis` should be presented to analysts alongside the binary label
- The threshold was optimised for recall ≥ 70% on the validation set
- **Do not use predictions alone for programme decisions** — always combine with
  IPC technical working group assessment and field knowledge
- The model is known to underperform on cold-start Crisis in new geographic units;
  apply additional scrutiny to units with no prior Crisis history

---

## 9. Ethical Considerations

- **Consequential decisions**: Food security predictions influence resource allocation
  affecting vulnerable populations. False negatives (missed Crisis) may delay response;
  false positives may divert resources from genuine crises elsewhere.
- **Accountability**: This system is designed as a decision-support tool, not a
  replacement for IPC technical working group processes.
- **Transparency**: Known failure modes are documented in Section 7. Users should
  be informed of the model's limitations before operational use.
- **Equity**: The model may perform differently across livelihood zones depending on
  data density. Zones with sparse historical data warrant additional caution.

---

## 10. Model Versioning

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | April 2025 | Initial release — 16 features including leaky unit aggregates |
| v1.1.0 | April 2026 | Removed leaky features; added cold-start interactions; walk-forward CV; recall-constrained threshold; full failure analysis |
---

## 11. Experiment Log

### Regional spatial covariates (attempted, reverted — April 2026)

**Hypothesis**: Regional crisis rate from the previous period is a proxy for
rainfall deficit. When neighboring units are in Crisis, drought is likely
affecting the whole region, which should predict escalation in the focal unit.

**Features added**:
- `region_mean_phase_lag1` — mean IPC phase of other units in same region, lagged
- `region_pct_crisis_lag1` — fraction of regional units in Crisis, lagged
- `region_crisis_count_lag1` — count of Crisis units in region, lagged
- `region_crisis_trend` — change in regional crisis rate over two periods
- `cold_start_regional` — interaction: unit below Crisis × regional crisis rate

All features computed leave-one-out (excluding focal unit) and lagged by one
period to prevent leakage.

**Result**: ROC-AUC dropped from 0.801 to 0.665. Crisis recall unchanged at 0%.
Regional features had zero importance in the final model.

**Root cause**: The 2024/2026 Crisis events are spatially isolated.
`cold_start_regional` was exactly 0.0 for all 37 test Crisis cases —
neighboring units were NOT in crisis. The regional spillover signal that
characterised 2021/2022 (a regional drought) does not exist in 2024/2026,
where individual units escalated independently.

**Conclusion**: The 2021/2022 vs 2024/2026 distributional shift operates at
both the unit level (lower lag values) and the regional level (isolated vs
clustered Crisis). No feature constructed from phase history alone can bridge
this gap. True external covariates (CHIRPS rainfall, NDVI anomaly, market
price indices) are required to capture the drivers of isolated cold-start
escalation.
