"""
FastAPI monitoring service for the DDoS/intrusion detector.

Loads the trained RandomForest model plus the exact StandardScaler and PCA
that were FIT during preprocessing (src/preprocess.py) so a live flow record
gets the identical transform the training data went through:

    raw 78 features -> scaler.transform -> pca.transform -> model.predict

Endpoints:
    GET  /health           service + model status
    GET  /features          the ordered list of raw feature names a request
                             must supply (this is feature_columns.json)
    POST /predict            score ONE flow record
    POST /predict/batch       score a LIST of flow records in one call

Run:
    uvicorn src.service:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import time
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "model/artifacts")
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "frontend")

app = FastAPI(
    title="Cloud DDoS Detector",
    description="ML-based DDoS/intrusion detection monitoring service "
                 "(CICIDS2017-trained Random Forest classifier).",
    version="1.0.0",
)

# --- Load artifacts once at startup ---------------------------------------
_model = None
_scaler = None
_pca = None
_feature_columns: List[str] = []


def _load_artifacts():
    global _model, _scaler, _pca, _feature_columns
    _model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
    _scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    _pca = joblib.load(os.path.join(ARTIFACTS_DIR, "pca.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as f:
        _feature_columns = json.load(f)


@app.on_event("startup")
def startup_event():
    _load_artifacts()


# --- Request / response schemas -------------------------------------------
class FlowRecord(BaseModel):
    """
    A single network flow, keyed by the RAW CICIDS2017/CICFlowMeter column
    names (e.g. "Flow Duration", "Flow Bytes/s", "SYN Flag Count", ...).
    Use GET /features to see the exact required keys and their order.
    Any missing key defaults to 0.0; unknown extra keys are ignored.
    """
    features: Dict[str, float] = Field(..., description="raw feature name -> value")


class PredictResponse(BaseModel):
    prediction: str
    is_attack: bool
    attack_probability: float
    latency_ms: float


class BatchPredictRequest(BaseModel):
    records: List[FlowRecord]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    attack_count: int
    benign_count: int


# --- Core scoring -----------------------------------------------------------
def _vectorize(record: FlowRecord) -> np.ndarray:
    row = [record.features.get(col, 0.0) for col in _feature_columns]
    return np.array(row, dtype=float).reshape(1, -1)


def _score(record: FlowRecord) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.time()
    x = _vectorize(record)
    x_scaled = _scaler.transform(x)
    x_pca = _pca.transform(x_scaled)
    pred = int(_model.predict(x_pca)[0])
    proba = float(_model.predict_proba(x_pca)[0, 1])
    latency_ms = (time.time() - t0) * 1000
    return PredictResponse(
        prediction="ATTACK" if pred == 1 else "BENIGN",
        is_attack=bool(pred),
        attack_probability=proba,
        latency_ms=latency_ms,
    )


# --- Routes ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_type": type(_model).__name__ if _model is not None else None,
        "n_features_expected": len(_feature_columns),
        "pca_components": getattr(_pca, "n_components_", None),
    }


@app.get("/features")
def features():
    return {"feature_columns": _feature_columns, "count": len(_feature_columns)}


@app.get("/metrics")
def metrics():
    path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No metrics.json yet — run src/train.py first")
    with open(path) as f:
        return json.load(f)


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard/")


@app.post("/predict", response_model=PredictResponse)
def predict(record: FlowRecord):
    return _score(record)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    results = [_score(r) for r in req.records]
    attack_count = sum(1 for r in results if r.is_attack)
    return BatchPredictResponse(
        results=results,
        attack_count=attack_count,
        benign_count=len(results) - attack_count,
    )


# --- Dashboard (static frontend) -------------------------------------------
# Mounted last so it doesn't shadow the API routes above. Visiting "/" redirects
# here; the dashboard's JS calls the API routes on this same origin.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn

    _load_artifacts()
    uvicorn.run(app, host="0.0.0.0", port=8000)
