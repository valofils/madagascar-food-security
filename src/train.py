import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_recall_curve, classification_report,
    roc_auc_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PROCESSED, MODELS_DIR
from src.features import FEATURE_COLS, TARGET_COL

PHASE_LABELS = {0: "Minimal (P1)", 1: "Stressed (P2)", 2: "Crisis+ (P3+)"}


def load_features(scenario: str = "cs") -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED, f"features_{scenario}.csv")
    df = pd.read_csv(path)
    print(f"Loaded features ({scenario}): {df.shape}")
    return df


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    return (df[TARGET_COL] >= 3).astype(int)


def make_multiclass_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL].clip(upper=3).astype(int) - 1


def temporal_split(df: pd.DataFrame, test_year: int = 2024):
    train = df[df["year"] < test_year].copy()
    test  = df[df["year"] >= test_year].copy()
    print(f"Train: {len(train)} rows ({train.year.min():.0f}–{train.year.max():.0f})")
    print(f"Test:  {len(test)} rows  ({test.year.min():.0f}–{test.year.max():.0f})")
    return train, test


def find_best_threshold(model, X_val, y_val,
                        min_recall: float = 0.55) -> tuple[float, dict]:
    """
    Find threshold that maximises F1 subject to recall >= min_recall.

    In a humanitarian context, missing a Crisis is worse than a false alarm,
    so we enforce a minimum recall floor before optimising precision.
    Falls back to pure F1 if no threshold meets the recall constraint.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)

    # Try recall-constrained F1 first
    mask = rec[:-1] >= min_recall   # prec/rec have one more element than thresholds
    if mask.any():
        best_idx = np.argmax(f1_scores[:-1][mask])
        best_thresh = float(thresholds[mask][best_idx])
        strategy = f"recall≥{min_recall:.0%}"
    else:
        # Fall back to pure F1
        best_idx   = np.argmax(f1_scores)
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        strategy   = "pure F1 (recall constraint not achievable)"

    # Metrics at chosen threshold
    y_pred = (y_prob >= best_thresh).astype(int)
    metrics = {
        "threshold":     round(best_thresh, 6),
        "val_precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "val_recall":    round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "val_f1":        round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "strategy":      strategy,
    }
    print(f"  Threshold ({strategy}): {best_thresh:.4f}  "
          f"precision={metrics['val_precision']:.3f}  "
          f"recall={metrics['val_recall']:.3f}  "
          f"F1={metrics['val_f1']:.3f}")
    return best_thresh, metrics


def walk_forward_cv(df: pd.DataFrame,
                    test_years: list[int],
                    min_recall: float = 0.55) -> dict:
    """
    Walk-forward cross-validation across multiple test years.
    Each fold trains on all years before test_year and evaluates on test_year.
    Returns averaged metrics across folds.
    """
    print("\n── Walk-forward cross-validation ──")
    fold_results = []

    for test_year in test_years:
        train_fold = df[df["year"] < test_year]
        test_fold  = df[df["year"] == test_year]

        y_train_raw = make_binary_target(train_fold)
        y_test      = make_binary_target(test_fold)

        if y_test.sum() == 0:
            print(f"  Fold {test_year}: no Crisis cases — skipping")
            continue
        if y_train_raw.sum() < 6:
            print(f"  Fold {test_year}: too few Crisis cases in train — skipping")
            continue

        X_train_raw = train_fold[FEATURE_COLS]
        X_test      = test_fold[FEATURE_COLS]

        # Stratified val split for threshold tuning
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_raw, y_train_raw,
            test_size=0.2, random_state=42, stratify=y_train_raw,
        )

        sm = SMOTE(random_state=42, k_neighbors=min(5, y_tr.sum() - 1))
        X_sm, y_sm = sm.fit_resample(X_train_raw, y_train_raw)

        model = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        model.fit(X_sm, y_sm, eval_set=[(X_val, y_val)], verbose=False)

        threshold, _ = find_best_threshold(model, X_val, y_val, min_recall)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        fold_roc   = roc_auc_score(y_test, y_prob)
        fold_prec  = precision_score(y_test, y_pred, zero_division=0)
        fold_rec   = recall_score(y_test, y_pred, zero_division=0)
        fold_f1    = f1_score(y_test, y_pred, zero_division=0)
        n_crisis   = y_test.sum()

        print(f"  Fold {test_year}: n_crisis={n_crisis}  "
              f"threshold={threshold:.3f}  "
              f"ROC={fold_roc:.3f}  prec={fold_prec:.3f}  "
              f"rec={fold_rec:.3f}  F1={fold_f1:.3f}")

        fold_results.append({
            "year": test_year, "n_crisis": int(n_crisis),
            "threshold": threshold, "roc_auc": fold_roc,
            "precision": fold_prec, "recall": fold_rec, "f1": fold_f1,
        })

    if not fold_results:
        return {}

    results_df = pd.DataFrame(fold_results)
    print(f"\n  CV averages across {len(results_df)} folds:")
    print(f"    ROC-AUC:   {results_df['roc_auc'].mean():.3f}")
    print(f"    Precision: {results_df['precision'].mean():.3f}")
    print(f"    Recall:    {results_df['recall'].mean():.3f}")
    print(f"    F1:        {results_df['f1'].mean():.3f}")

    return {
        "folds":         fold_results,
        "mean_roc_auc":  round(float(results_df["roc_auc"].mean()), 4),
        "mean_precision":round(float(results_df["precision"].mean()), 4),
        "mean_recall":   round(float(results_df["recall"].mean()), 4),
        "mean_f1":       round(float(results_df["f1"].mean()), 4),
    }


def train_binary(train: pd.DataFrame, test: pd.DataFrame,
                 min_recall: float = 0.55) -> dict:
    print("\n--- Binary model: Crisis (3+) vs Not Crisis ---")

    X_train_raw = train[FEATURE_COLS]
    X_test      = test[FEATURE_COLS]
    y_train_raw = make_binary_target(train)
    y_test      = make_binary_target(test)

    print(f"Class balance — train Crisis: {y_train_raw.sum()}/{len(y_train_raw)} "
          f"({100*y_train_raw.mean():.1f}%)")
    print(f"Class balance — test  Crisis: {y_test.sum()}/{len(y_test)} "
          f"({100*y_test.mean():.1f}%)")

    # Stratified val split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=0.2, random_state=42, stratify=y_train_raw,
    )
    print(f"Val set: n={len(y_val)}, Crisis={y_val.sum()} ({100*y_val.mean():.1f}%)")

    # SMOTE on full training set
    print(f"Before SMOTE — Crisis: {y_train_raw.sum()} / {len(y_train_raw)}")
    sm = SMOTE(random_state=42, k_neighbors=min(5, y_train_raw.sum() - 1))
    X_sm, y_sm = sm.fit_resample(X_train_raw, y_train_raw)
    print(f"After SMOTE  — Crisis: {y_sm.sum()} / {len(y_sm)}")

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_sm, y_sm, eval_set=[(X_val, y_val)], verbose=False)

    # Threshold: recall-constrained F1
    threshold, val_metrics = find_best_threshold(model, X_val, y_val, min_recall)

    # Final test evaluation
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)
    test_roc    = roc_auc_score(y_test, y_prob_test)
    test_report = classification_report(
        y_test, y_pred_test,
        target_names=["Not Crisis (P1-2)", "Crisis (P3+)"],
        output_dict=True, zero_division=0,
    )
    crisis_m = test_report.get("Crisis (P3+)", {})

    print("\nTest-set performance at tuned threshold:")
    print(classification_report(
        y_test, y_pred_test,
        target_names=["Not Crisis (P1-2)", "Crisis (P3+)"],
        zero_division=0,
    ))
    print(f"ROC-AUC: {test_roc:.4f}")

    # Feature importances
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(importances.head(10).round(4).to_string())

    return {
        "model":     model,
        "threshold": threshold,
        "X_train":   X_train_raw,
        "X_test":    X_test,
        "y_train":   y_train_raw,
        "y_test":    y_test,
        "type":      "binary",
        "eval_metrics": {
            "roc_auc":        round(test_roc, 4),
            "threshold":      round(threshold, 6),
            "threshold_strategy": val_metrics["strategy"],
            "val_f1":         val_metrics["val_f1"],
            "val_precision":  val_metrics["val_precision"],
            "val_recall":     val_metrics["val_recall"],
            "test_precision": round(float(crisis_m.get("precision", 0)), 4),
            "test_recall":    round(float(crisis_m.get("recall", 0)), 4),
            "test_f1":        round(float(crisis_m.get("f1-score", 0)), 4),
            "test_support":   int(crisis_m.get("support", 0)),
        },
    }


def train_multiclass(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("\n--- Multiclass model: Phase 1 / 2 / 3+ ---")
    X_train_raw = train[FEATURE_COLS]
    X_test      = test[FEATURE_COLS]
    y_train_raw = make_multiclass_target(train)
    y_test      = make_multiclass_target(test)

    print(f"Before SMOTE: {y_train_raw.value_counts().sort_index().to_dict()}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_sm, y_sm = sm.fit_resample(X_train_raw, y_train_raw)
    print(f"After SMOTE:  {pd.Series(y_sm).value_counts().sort_index().to_dict()}")

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        objective="multi:softmax", num_class=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_sm, y_sm, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    print("\nWith SMOTE:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(PHASE_LABELS.values()),
        zero_division=0,
    ))

    return {
        "model":     model,
        "threshold": 0.5,
        "X_train":   X_train_raw,
        "X_test":    X_test,
        "y_train":   y_train_raw,
        "y_test":    y_test,
        "type":      "multiclass",
    }


def save_model(result: dict, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)

    artifact = {"model": result["model"], "threshold": result["threshold"]}
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  [ok] Model -> {path}")

    meta = {
        "model_name":     name,
        "model_version":  "1.1.0",
        "model_type":     result["type"],
        "features":       FEATURE_COLS,
        "n_features":     len(FEATURE_COLS),
        "target":         TARGET_COL,
        "threshold":      result["threshold"],
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "train_accuracy": float(result["model"].score(result["X_train"], result["y_train"])),
        "test_accuracy":  float(result["model"].score(result["X_test"],  result["y_test"])),
    }
    meta_path = os.path.join(MODELS_DIR, f"{name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [ok] Meta  -> {meta_path}")

    if "eval_metrics" in result:
        metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(result["eval_metrics"], f, indent=2)
        print(f"  [ok] Metrics -> {metrics_path}")


def run_training():
    print("\n=== Model Training v1.1 ===\n")
    df = load_features("cs")

    # Walk-forward CV to understand generalisation across years
    cv_years = [y for y in [2019, 2020, 2021, 2022, 2023]
                if (df[df["year"] == y]["ipc_phase"] >= 3).sum() >= 3]
    cv_results = walk_forward_cv(df, test_years=cv_years, min_recall=0.70)

    # Final model trained on all pre-2024 data
    train, test = temporal_split(df, test_year=2024)
    print(f"\nFinal model train: {len(train)} rows | test: {len(test)} rows")

    binary_result     = train_binary(train, test, min_recall=0.70)
    multiclass_result = train_multiclass(train, test)

    # Attach CV results to metrics
    if cv_results and "eval_metrics" in binary_result:
        binary_result["eval_metrics"]["cv"] = cv_results

    save_model(binary_result,     "ipc_binary_classifier")
    save_model(multiclass_result, "ipc_multiclass_classifier")

    print("\n=== Training complete ===")


if __name__ == "__main__":
    run_training()
