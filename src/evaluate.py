import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PROCESSED, MODELS_DIR
from src.features import FEATURE_COLS, TARGET_COL
from src.train import (
    load_features, temporal_split,
    make_binary_target, make_multiclass_target,
    PHASE_LABELS,
)

PLOTS_DIR = "data/processed/plots"


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ── binary evaluation ─────────────────────────────────────────────────────────

def evaluate_binary(model, X_test, y_test):
    print("\n=== Binary Model: Crisis (3+) vs Not Crisis ===\n")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(
        y_test, y_pred,
        target_names=["Not Crisis (P1-2)", "Crisis (P3+)"]
    ))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Crisis", "Crisis"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix — Binary")

    # precision-recall curve
    prec, rec, thresh = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, color="steelblue", lw=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve (AUC={roc_auc:.3f})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "binary_evaluation.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [ok] Plot -> {path}")

    return {"roc_auc": roc_auc}


# ── multiclass evaluation ─────────────────────────────────────────────────────

def evaluate_multiclass(model, X_test, y_test):
    print("\n=== Multiclass Model: P1 / P2 / P3+ ===\n")

    y_pred = model.predict(X_test)

    print(classification_report(
        y_test, y_pred,
        target_names=list(PHASE_LABELS.values())
    ))

    # confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=list(PHASE_LABELS.values())).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix — Multiclass")
    axes[0].tick_params(axis="x", rotation=15)

    # feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fi.tail(12).plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Top 12 Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "multiclass_evaluation.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [ok] Plot -> {path}")


# ── feature importance summary ────────────────────────────────────────────────

def print_feature_importance(model, label: str):
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    fi = fi.sort_values(ascending=False)
    print(f"\nTop 10 features ({label}):")
    for feat, score in fi.head(10).items():
        bar = "#" * int(score * 200)
        print(f"  {feat:<22} {score:.4f}  {bar}")


# ── save metrics ──────────────────────────────────────────────────────────────

def save_metrics(binary_metrics: dict, name: str = "evaluation_metrics"):
    path = os.path.join(MODELS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(binary_metrics, f, indent=2)
    print(f"  [ok] Metrics -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n=== Model Evaluation ===")
    ensure_plots_dir()

    df = load_features("cs")
    _, test = temporal_split(df, test_year=2024)

    X_test = test[FEATURE_COLS]

    # binary
    binary_model = load_model("ipc_binary_classifier")
    y_test_bin   = make_binary_target(test)
    bin_metrics  = evaluate_binary(binary_model, X_test, y_test_bin)
    print_feature_importance(binary_model, "binary")

    # multiclass
    multi_model  = load_model("ipc_multiclass_classifier")
    y_test_multi = make_multiclass_target(test)
    evaluate_multiclass(multi_model, X_test, y_test_multi)
    print_feature_importance(multi_model, "multiclass")

    save_metrics(bin_metrics)
    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    run_evaluation()
