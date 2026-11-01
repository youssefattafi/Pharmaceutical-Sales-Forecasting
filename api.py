"""Local FastAPI service for trusted saved pharmaceutical forecasting model.

Run python save_model.py first. Start with:
uvicorn api:app --reload
"""
from pathlib import Path
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
app = FastAPI(title="Pharmaceutical Demand Forecast API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(description="Lagged sales and calendar features; names must match /model-info")

@app.get("/health")
def health():
    return {"status": "ready" if MODEL_PATH.exists() and METADATA_PATH.exists() else "model_not_saved"}

@app.get("/model-info")
def model_info():
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metadata missing. Run python save_model.py")
    return json.loads(METADATA_PATH.read_text())

@app.post("/predict")
def predict(request: PredictionRequest):
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not available. Run python save_model.py")
    metadata = json.loads(METADATA_PATH.read_text())
    expected = metadata["feature_names"]
    provided = request.features
    missing = sorted(set(expected) - set(provided))
    extra = sorted(set(provided) - set(expected))
    if missing or extra:
        raise HTTPException(status_code=422, detail={"missing_features": missing, "unexpected_features": extra})
    values = np.array([[provided[name] for name in expected]], dtype=float)
    if not np.isfinite(values).all():
        raise HTTPException(status_code=422, detail="All feature values must be finite numbers")
    # joblib/pickle is unsafe for untrusted artifacts: load only files produced locally by save_model.py.
    model = joblib.load(MODEL_PATH)
    result = model.predict(values)[0]
    return {
        "model_version": metadata["version"],
        "forecast_type": metadata["forecast_type"],
        "predictions": {cat: round(float(max(0, value)), 3) for cat, value in zip(metadata["categories"], result)},
        "note": "One-week-ahead prediction using previously observed weekly sales. Not medical guidance.",
    }
