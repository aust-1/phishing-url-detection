"""Metrics and figures used to compare every model with the baseline."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from phishing.risk import risk_level, risk_score

LEVEL_ORDER = ["low", "medium", "high"]


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_proba: ArrayLike | None = None
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, false positive rate and, with probabilities,
    ROC-AUC and Brier score. Class 1 (phishing) is the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["brier"] = brier_score_loss(y_true, y_proba)
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}


def risk_level_table(y_true: ArrayLike, y_proba: ArrayLike) -> pd.DataFrame:
    """Number of URLs and real phishing rate for each risk level.

    With a reliable score the phishing rate must go up from low to high.
    """
    df = pd.DataFrame({"level": risk_level(risk_score(y_proba)), "phishing": np.asarray(y_true)})
    table = df.groupby("level")["phishing"].agg(n_urls="size", phishing_rate="mean")
    return table.reindex(LEVEL_ORDER).fillna({"n_urls": 0}).astype({"n_urls": int})


def plot_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["legitimate", "phishing"], ax=ax, colorbar=False
    )
    _save(fig, path)


def plot_calibration(y_true: ArrayLike, y_proba: ArrayLike, path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=n_bins, ax=ax)
    ax.set_title("Calibration curve")
    _save(fig, path)


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
