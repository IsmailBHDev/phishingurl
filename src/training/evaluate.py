# src/training/evaluate.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

from src.features.url_features import urls_to_feature_df

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

URL_COL = "url"
LABEL_COL = "is_spam"


def load_threshold():
    p = MODEL_DIR / "threshold.json"
    if p.exists():
        return float(json.loads(p.read_text())["threshold"])
    return 0.5


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    model = joblib.load(MODEL_DIR / "url_model.joblib")
    threshold = load_threshold()

    X_test = urls_to_feature_df(test_df[URL_COL].tolist())
    y_test = test_df[LABEL_COL].astype(int).to_numpy()

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, pred).tolist()

    # Useful for imbalance:
    pr_auc = float(average_precision_score(y_test, proba))

    # ROC-AUC can fail if only one class is present (rare but possible)
    try:
        roc_auc = float(roc_auc_score(y_test, proba))
    except Exception:
        roc_auc = None

    report = {
        "threshold": float(threshold),
        "test_precision": float(p),
        "test_recall": float(r),
        "test_f1": float(f1),
        "test_pr_auc": pr_auc,
        "test_roc_auc": roc_auc,
        "confusion_matrix": {
            "format": "[[tn, fp], [fn, tp]]",
            "values": cm,
        },
        "base_rate_spam": float(np.mean(y_test)),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2))

    print("Test metrics saved -> reports/test_metrics.json")
    print(report)


if __name__ == "__main__":
    main()
