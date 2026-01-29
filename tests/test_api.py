from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "threshold" in body

def test_predict_url():
    r = client.post("/predict/url", json={"url": "http://secure-login-paypal.xyz/verify"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"legit", "phishing"}
    assert 0.0 <= body["probability"] <= 1.0
