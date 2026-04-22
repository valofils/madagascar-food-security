"""
src/train_lstm.py
-----------------
Module 8 — Deep Learning: LSTM for IPC Food Security Forecasting

Trains a stacked LSTM binary classifier (Crisis Phase 3+ vs Not Crisis)
on IPC time series sequences per Madagascar livelihood zone.

Mirrors ML Zoomcamp Module 8 concepts:
  - TensorFlow / Keras Sequential API
  - Dense baseline vs LSTM comparison
  - Learning rate range test
  - ModelCheckpoint + EarlyStopping + ReduceLROnPlateau callbacks
  - Threshold tuning via precision-recall curve
  - Model saved in .keras format

Usage:
    python src/train_lstm.py
    python src/train_lstm.py --seq-len 6 --epochs 150
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)
from datetime import datetime, timezone

from config import DATA_PROCESSED, MODELS_DIR
from src.lstm_features import build_sequences, SEQUENCE_FEATURES

PLOTS_DIR = os.path.join(DATA_PROCESSED, "plots")
tf.random.set_seed(42)
np.random.seed(42)


# ── model builders ────────────────────────────────────────────────────────────

def build_dense_baseline(seq_len: int, n_features: int, lr: float = 1e-3):
    """Flatten sequence → Dense layers. No temporal memory."""
    model = keras.Sequential([
        keras.Input(shape=(seq_len, n_features)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="dense_baseline")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_lstm(seq_len: int, n_features: int, lr: float = 1e-3):
    """
    Stacked LSTM for binary crisis classification.

    Input → LSTM(64, return_sequences) → Dropout
          → LSTM(32)                   → Dropout
          → Dense(16, relu)            → Dense(1, sigmoid)
    """
    model = keras.Sequential([
        keras.Input(shape=(seq_len, n_features)),
        layers.LSTM(64, return_sequences=True, name="lstm_1"),
        layers.Dropout(0.3,                   name="dropout_1"),
        layers.LSTM(32,                        name="lstm_2"),
        layers.Dropout(0.2,                   name="dropout_2"),
        layers.Dense(16, activation="relu",   name="dense_head"),
        layers.Dense(1,  activation="sigmoid", name="output"),
    ], name="ipc_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ── learning rate sweep ───────────────────────────────────────────────────────

def lr_sweep(seq_len, n_features, X_tr, y_tr, class_weight,
             min_lr=1e-5, max_lr=1e-1, steps=80):
    """Exponential LR range test. Returns (lrs, losses, best_lr)."""
    print("\n--- Learning Rate Range Test ---")
    model = keras.Sequential([
        keras.Input(shape=(seq_len, n_features)),
        layers.LSTM(32),
        layers.Dense(1, activation="sigmoid"),
    ])
    lrs, losses = [], []

    for lr in np.geomspace(min_lr, max_lr, steps):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
            loss="binary_crossentropy",
        )
        h = model.fit(X_tr, y_tr, epochs=1, batch_size=32,
                      class_weight=class_weight, verbose=0)
        lrs.append(float(lr))
        losses.append(h.history["loss"][0])

    best_lr = float(lrs[int(np.argmin(losses))])
    print(f"  Suggested LR: {best_lr:.2e}")

    # plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.semilogx(lrs, losses, color="steelblue", lw=2)
    plt.axvline(best_lr, color="red", linestyle="--",
                label=f"Best LR ≈ {best_lr:.1e}")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Range Test — LSTM")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "lstm_lr_sweep.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [ok] Plot -> {path}")

    return lrs, losses, best_lr


# ── training ──────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_test, y_test,
                class_weight, epochs, checkpoint_path):
    """Train with ModelCheckpoint + EarlyStopping + ReduceLROnPlateau."""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── evaluation ────────────────────────────────────────────────────────────────

def find_best_threshold(y_test, y_prob) -> float:
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1))
    best_t = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    print(f"  Best threshold: {best_t:.4f}  "
          f"(P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}, F1={f1[best_idx]:.3f})")
    return best_t


def evaluate_model(model, X_test, y_test, name, threshold=0.5):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n=== {name} (threshold={threshold:.4f}) ===")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Crisis (P1-2)", "Crisis (P3+)"],
    ))
    print(f"ROC-AUC: {auc:.4f}")
    return y_prob, auc


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(history, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, label in zip(
        axes, ["loss", "accuracy", "auc"], ["Loss", "Accuracy", "ROC-AUC"]
    ):
        ax.plot(history.history[metric],            label="Train", color="steelblue", lw=2)
        ax.plot(history.history[f"val_{metric}"],   label="Val",   color="coral",     lw=2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [ok] Plot -> {save_path}")


def plot_confusion_matrices(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, y_test, y_pred) in zip(axes, results):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Not Crisis", "Crisis"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    plt.suptitle("Confusion Matrices", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [ok] Plot -> {save_path}")


def plot_pr_curves(curve_data, save_path):
    plt.figure(figsize=(7, 5))
    for name, y_test, y_prob, color in curve_data:
        p, r, _ = precision_recall_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(r, p, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Deep Learning Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [ok] Plot -> {save_path}")


# ── save ──────────────────────────────────────────────────────────────────────

def save_lstm_artifacts(model, scaler, best_thresh, auc,
                        seq_len, n_features, epochs_run):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # model
    model_path = os.path.join(MODELS_DIR, "ipc_lstm.keras")
    model.save(model_path)
    print(f"  [ok] Model  -> {model_path}")

    # scaler
    scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [ok] Scaler -> {scaler_path}")

    # metadata
    meta = {
        "model_name":    "ipc_lstm",
        "model_type":    "LSTM binary classifier (Crisis Phase 3+)",
        "framework":     f"TensorFlow {tf.__version__} / Keras {keras.__version__}",
        "seq_len":       seq_len,
        "n_features":    n_features,
        "features":      SEQUENCE_FEATURES,
        "threshold":     round(best_thresh, 4),
        "val_auc":       round(auc, 4),
        "epochs_run":    epochs_run,
        "trained_at":    datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(MODELS_DIR, "ipc_lstm_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [ok] Meta   -> {meta_path}")

    # verify reload
    reloaded = keras.models.load_model(model_path)
    print(f"  [ok] Reload check passed")
    return model_path


# ── main ──────────────────────────────────────────────────────────────────────

def run_lstm_training(seq_len: int = 4, epochs: int = 100):
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Module 8 — LSTM IPC Food Security Classifier          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  Keras      : {keras.__version__}")

    # ── 1. load data ──────────────────────────────────────────────────────────
    print("\n--- Loading data ---")
    df = pd.read_csv(
        os.path.join(DATA_PROCESSED, "features_cs.csv"),
        parse_dates=["projection_start"],
    )
    print(f"  Loaded: {df.shape}  |  {df.fnid.nunique()} unique units")

    # ── 2. build sequences ────────────────────────────────────────────────────
    print("\n--- Building sequences ---")
    X_train, y_train, X_test, y_test, scaler = build_sequences(
        df, seq_len=seq_len, test_year=2024
    )
    N_FEATURES = X_train.shape[2]

    # class weights (mirrors SMOTE intent from Module 6)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}
    print(f"  Class weight for Crisis: {class_weight[1]:.1f}")

    # ── 3. dense baseline ─────────────────────────────────────────────────────
    print("\n--- Dense Baseline ---")
    dense_model = build_dense_baseline(seq_len, N_FEATURES)
    dense_model.summary()

    dense_history = train_model(
        dense_model, X_train, y_train, X_test, y_test,
        class_weight, epochs=min(epochs, 50),
        checkpoint_path=os.path.join(MODELS_DIR, "ipc_dense_best.keras"),
    )
    dense_prob, dense_auc = evaluate_model(
        dense_model, X_test, y_test, "Dense Baseline"
    )

    # ── 4. LR sweep ───────────────────────────────────────────────────────────
    _, _, best_lr = lr_sweep(seq_len, N_FEATURES, X_train, y_train, class_weight)

    # ── 5. LSTM ───────────────────────────────────────────────────────────────
    print("\n--- LSTM Model ---")
    lstm_model = build_lstm(seq_len, N_FEATURES, lr=best_lr)
    lstm_model.summary()

    checkpoint_path = os.path.join(MODELS_DIR, "ipc_lstm_best.keras")
    lstm_history = train_model(
        lstm_model, X_train, y_train, X_test, y_test,
        class_weight, epochs=epochs,
        checkpoint_path=checkpoint_path,
    )

    # ── 6. evaluate ───────────────────────────────────────────────────────────
    print("\n--- Evaluation ---")
    lstm_prob, lstm_auc = evaluate_model(lstm_model, X_test, y_test, "LSTM (default threshold)")

    best_thresh = find_best_threshold(y_test, lstm_prob)
    evaluate_model(lstm_model, X_test, y_test,
                   f"LSTM (tuned threshold={best_thresh:.4f})", threshold=best_thresh)

    # ── 7. plots ──────────────────────────────────────────────────────────────
    print("\n--- Plots ---")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_training_curves(
        lstm_history, "LSTM — Training Curves",
        os.path.join(PLOTS_DIR, "lstm_training_curves.png"),
    )
    plot_training_curves(
        dense_history, "Dense Baseline — Training Curves",
        os.path.join(PLOTS_DIR, "dense_training_curves.png"),
    )

    lstm_pred  = (lstm_prob  >= best_thresh).astype(int)
    dense_pred = (dense_prob >= 0.5).astype(int)
    plot_confusion_matrices(
        [
            (f"LSTM (t={best_thresh:.3f})", y_test, lstm_pred),
            ("Dense Baseline (t=0.5)",      y_test, dense_pred),
        ],
        os.path.join(PLOTS_DIR, "lstm_confusion_matrices.png"),
    )

    plot_pr_curves(
        [
            ("LSTM",           y_test, lstm_prob,  "steelblue"),
            ("Dense Baseline", y_test, dense_prob, "coral"),
        ],
        os.path.join(PLOTS_DIR, "lstm_pr_curves.png"),
    )

    # ── 8. save ───────────────────────────────────────────────────────────────
    print("\n--- Saving artifacts ---")
    epochs_run = len(lstm_history.history["loss"])
    save_lstm_artifacts(
        lstm_model, scaler, best_thresh, lstm_auc,
        seq_len, N_FEATURES, epochs_run,
    )

    # ── 9. summary ────────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                     MODULE 8 SUMMARY                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Dense baseline ROC-AUC : {dense_auc:.4f}                        ║")
    print(f"║  LSTM ROC-AUC           : {lstm_auc:.4f}                        ║")
    print(f"║  XGBoost ROC-AUC        : 0.9160  (Module 6)                ║")
    print(f"║  LSTM threshold         : {best_thresh:.4f}                        ║")
    print(f"║  Epochs trained         : {epochs_run:<4}                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Artifacts saved:                                        ║")
    print("║    models/ipc_lstm.keras                                 ║")
    print("║    models/lstm_scaler.pkl                                ║")
    print("║    models/ipc_lstm_meta.json                             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM for IPC food security")
    parser.add_argument("--seq-len", type=int, default=4,
                        help="Sequence length (default: 4)")
    parser.add_argument("--epochs",  type=int, default=100,
                        help="Max epochs (default: 100)")
    args = parser.parse_args()

    run_lstm_training(seq_len=args.seq_len, epochs=args.epochs)
