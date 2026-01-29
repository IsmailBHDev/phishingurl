# src/training/train.py

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
)

from src.features.url_features import urls_to_feature_df

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

URL_COL = "url"
LABEL_COL = "is_spam"


def pick_threshold_precision_or_f1(y_true, y_proba, min_precision=0.50):
    """
    Try to find a threshold with precision >= min_precision.
    If none exists, fallback to threshold that maximizes F1.
    """
    best_prec = None
    best_f1 = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}

    for t in [i / 100 for i in range(1, 100)]:
        y_pred = (y_proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # Track best F1 (fallback)
        if f1 > best_f1["f1"]:
            best_f1 = {
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }

        # Track best threshold that meets precision constraint (maximize recall)
        if p >= min_precision:
            if best_prec is None or r > best_prec["recall"]:
                best_prec = {
                    "threshold": float(t),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f1),
                }

    if best_prec is not None:
        best_prec["mode"] = f"precision>= {min_precision}"
        return best_prec

    best_f1["mode"] = "best_f1_fallback"
    return best_f1


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    X_train = urls_to_feature_df(train_df[URL_COL].tolist())
    y_train = train_df[LABEL_COL].astype(int)

    X_val = urls_to_feature_df(val_df[URL_COL].tolist())
    y_val = val_df[LABEL_COL].astype(int)

    # Imbalance-aware baseline
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]

    # PR-AUC is informative for imbalanced problems
    pr_auc = average_precision_score(y_val, val_proba)

    # Threshold selection (precision-first with fallback)
    thresh_info = pick_threshold_precision_or_f1(y_val, val_proba, min_precision=0.50)
    t = float(thresh_info["threshold"])

    val_pred = (val_proba >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val, val_pred, average="binary", zero_division=0
    )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_DIR / "url_model.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(list(X_train.columns), indent=2))
    (MODEL_DIR / "threshold.json").write_text(json.dumps({"threshold": t}, indent=2))

    report = {
        "val_precision": float(p),
        "val_recall": float(r),
        "val_f1": float(f1),
        "val_pr_auc": float(pr_auc),
        "chosen_threshold": float(t),
        "threshold_target_precision": 0.50,
        "threshold_mode": thresh_info["mode"],
        "threshold_selected_precision": float(thresh_info["precision"]),
        "threshold_selected_recall": float(thresh_info["recall"]),
        "threshold_selected_f1": float(thresh_info["f1"]),
    }

    (REPORTS_DIR / "val_metrics.json").write_text(json.dumps(report, indent=2))

    print("Saved model -> models/url_model.joblib")
    print("Validation:", report)


if __name__ == "__main__":
    main()
