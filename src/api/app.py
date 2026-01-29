# src/api/app.py

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.url_features import extract_url_features

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "url_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

app = FastAPI(title="PhishGuard URL Detector", version="1.0")


class URLRequest(BaseModel):
    url: str


def load_threshold() -> float:
    if THRESHOLD_PATH.exists():
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    return 0.5


def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature schema file: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_columns: List[str] = json.loads(FEATURES_PATH.read_text())
    threshold = load_threshold()
    return model, feature_columns, threshold


model, feature_columns, threshold = load_assets()


def features_for_model(url: str) -> pd.DataFrame:
    feats = extract_url_features(url)
    df = pd.DataFrame([feats])

    # Ensure all expected columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct order
    return df[feature_columns].fillna(0)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold, "num_features": len(feature_columns)}


@app.post("/predict/url")
def predict_url(req: URLRequest) -> Dict[str, Any]:
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty")

    X = features_for_model(url)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= threshold)
    label = "phishing" if pred == 1 else "legit"

    # Explainability (Logistic Regression contributions)
    top_signals = []
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        row = X.iloc[0].to_dict()
        contribs = [(feature_columns[i], float(coefs[i] * row[feature_columns[i]])) for i in range(len(feature_columns))]
        contribs.sort(key=lambda x: x[1], reverse=True)
        top_signals = [{"feature": f, "contribution": c} for f, c in contribs[:8]]

    return {
        "url": url,
        "label": label,
        "probability": proba,
        "threshold": threshold,
        "top_signals": top_signals,
    }
