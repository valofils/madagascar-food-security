import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, roc_auc_score
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
    print(f"Train: {len(train)} rows ({train.year.min():.0f}-{train.year.max():.0f})")
    print(f"Test:  {len(test)} rows  ({test.year.min():.0f}-{test.year.max():.0f})")
    return train, test


def find_best_threshold(model, X_val, y_val) -> tuple[float, dict]:
    """
    Find threshold that maximises F1 for the positive (Crisis) class.
    Called on a stratified validation set carved from training data.
    Returns (threshold, metrics_at_threshold).
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx  = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    metrics = {
        "threshold":     round(best_thresh, 6),
        "val_precision": round(float(prec[best_idx]), 4),
        "val_recall":    round(float(rec[best_idx]), 4),
        "val_f1":        round(float(f1_scores[best_idx]), 4),
    }
    print(f"  Best threshold (val): {best_thresh:.4f}  "
          f"precision={metrics['val_precision']:.3f}  "
          f"recall={metrics['val_recall']:.3f}  "
          f"F1={metrics['val_f1']:.3f}")
    return best_thresh, metrics


def train_binary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("\n--- Binary model: Crisis (3+) vs Not Crisis ---")

    X_train_raw = train[FEATURE_COLS]
    X_test      = test[FEATURE_COLS]
    y_train_raw = make_binary_target(train)
    y_test      = make_binary_target(test)

    print(f"Class balance — train Crisis: {y_train_raw.sum()}/{len(y_train_raw)} "
          f"({100*y_train_raw.mean():.1f}%)")
    print(f"Class balance — test  Crisis: {y_test.sum()}/{len(y_test)} "
          f"({100*y_test.mean():.1f}%)")

    # ── Stratified validation split for threshold tuning ──────────────────────
    # Random stratified split — keeps Crisis cases proportional in both halves.
    # This is only for threshold tuning; the model trains on the full train set.
    # We use stratify= to guarantee enough Crisis cases in the validation set.
    X_tr_thresh, X_val, y_tr_thresh, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_train_raw,
    )
    print(f"Threshold-tuning val set: n={len(y_val)}, "
          f"Crisis={y_val.sum()} ({100*y_val.mean():.1f}%)")

    # ── SMOTE on full training set (model trains on everything pre-2024) ───────
    print(f"Before SMOTE — Crisis: {y_train_raw.sum()} / {len(y_train_raw)}")
    sm = SMOTE(random_state=42, k_neighbors=min(5, y_train_raw.sum() - 1))
    X_train_smote, y_train_smote = sm.fit_resample(X_train_raw, y_train_raw)
    print(f"After SMOTE  — Crisis: {y_train_smote.sum()} / {len(y_train_smote)}")

    # ── XGBoost — no scale_pos_weight after SMOTE (classes already balanced) ──
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    # Eval set uses the stratified val split (unsmoted, real distribution)
    model.fit(
        X_train_smote, y_train_smote,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Tune threshold on stratified val set ──────────────────────────────────
    threshold, val_metrics = find_best_threshold(model, X_val, y_val)

    # ── Final evaluation on held-out test set (touched once) ──────────────────
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)

    test_roc_auc = roc_auc_score(y_test, y_prob_test)
    test_report  = classification_report(
        y_test, y_pred_test,
        target_names=["Not Crisis (P1-2)", "Crisis (P3+)"],
        output_dict=True,
        zero_division=0,
    )
    crisis_metrics = test_report.get("Crisis (P3+)", {})

    print("\nTest-set performance at tuned threshold:")
    print(classification_report(
        y_test, y_pred_test,
        target_names=["Not Crisis (P1-2)", "Crisis (P3+)"],
        zero_division=0,
    ))
    print(f"ROC-AUC: {test_roc_auc:.4f}")

    return {
        "model":     model,
        "threshold": threshold,
        "X_train":   X_train_raw,
        "X_test":    X_test,
        "y_train":   y_train_raw,
        "y_test":    y_test,
        "type":      "binary",
        "eval_metrics": {
            "roc_auc":        round(test_roc_auc, 4),
            "threshold":      round(threshold, 6),
            "val_f1":         val_metrics["val_f1"],
            "val_precision":  val_metrics["val_precision"],
            "val_recall":     val_metrics["val_recall"],
            "test_precision": round(float(crisis_metrics.get("precision", 0)), 4),
            "test_recall":    round(float(crisis_metrics.get("recall", 0)), 4),
            "test_f1":        round(float(crisis_metrics.get("f1-score", 0)), 4),
            "test_support":   int(crisis_metrics.get("support", 0)),
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
    X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)
    print(f"After SMOTE:  {pd.Series(y_train).value_counts().sort_index().to_dict()}")

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        objective="multi:softmax", num_class=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

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
        "model_version":  "1.0.0",
        "model_type":     result["type"],
        "features":       FEATURE_COLS,
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
    print("\n=== Model Training (with SMOTE + threshold tuning) ===\n")
    df = load_features("cs")
    train, test = temporal_split(df, test_year=2024)

    binary_result     = train_binary(train, test)
    multiclass_result = train_multiclass(train, test)

    save_model(binary_result,     "ipc_binary_classifier")
    save_model(multiclass_result, "ipc_multiclass_classifier")

    print("\n=== Training complete ===")


if __name__ == "__main__":
    run_training()
