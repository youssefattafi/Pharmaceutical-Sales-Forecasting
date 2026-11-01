"""FastAPI one-week-ahead forecasts from 12+ observed weekly sales records.

Run from repository root after generating models/random_forest.joblib and
models/metadata.json:
    uvicorn api:app --reload

Only load model artifacts that you trained yourself and trust: joblib uses pickle.
"""
from datetime import date, timedelta
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / 'models' / 'random_forest.joblib'
META_FILE = ROOT / 'models' / 'metadata.json'
CATEGORIES = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

app = FastAPI(title='Pharma Sales Forecast API', version='1.0')

class WeekSales(BaseModel):
    date: date
    quantities: dict[str, float]

class ForecastRequest(BaseModel):
    history: list[WeekSales] = Field(min_length=12)

@app.get('/health')
def health():
    return {'status': 'ready' if MODEL_FILE.exists() and META_FILE.exists() else 'model_missing'}

@app.post('/predict')
def predict(payload: ForecastRequest):
    if not MODEL_FILE.exists() or not META_FILE.exists():
        raise HTTPException(503, 'Missing model files; run python save_model.py')
    history = sorted(payload.history, key=lambda row: row.date)
    dates = [row.date for row in history]
    if len(set(dates)) != len(dates) or any((b - a).days != 7 for a, b in zip(dates, dates[1:])):
        raise HTTPException(422, 'History dates must be unique and exactly 7 days apart')
    for row in history:
        if set(row.quantities) != set(CATEGORIES):
            raise HTTPException(422, 'Every week must contain exactly the eight expected categories')
        if any(not np.isfinite(value) or value < 0 for value in row.quantities.values()):
            raise HTTPException(422, 'Quantities must be finite nonnegative numbers')
    next_week = dates[-1] + timedelta(days=7)
    values = {c: np.array([w.quantities[c] for w in history], dtype=float) for c in CATEGORIES}
    features = {}
    for category in CATEGORIES:
        v = values[category]
        for lag in (1, 2, 4, 8, 12):
            features[f'{category}_lag_{lag}'] = float(v[-lag])
        features[f'{category}_avg4'] = float(v[-4:].mean())
    week = next_week.isocalendar().week
    features['week_sin'] = float(np.sin(2 * np.pi * week / 52.18))
    features['week_cos'] = float(np.cos(2 * np.pi * week / 52.18))
    metadata = json.loads(META_FILE.read_text())
    if metadata.get('categories') != CATEGORIES:
        raise HTTPException(409, 'Model category metadata mismatch')
    expected = metadata['feature_names']
    if set(features) != set(expected):
        raise HTTPException(409, 'Model feature metadata mismatch')
    # SECURITY: only load your own locally generated joblib file.
    model = joblib.load(MODEL_FILE)
    predictions = model.predict(pd.DataFrame([features], columns=expected))[0]
    return {'forecast_week': next_week.isoformat(),
            'model_version': metadata.get('version'),
            'predicted_weekly_quantities': {c: round(max(0, float(p)), 2) for c, p in zip(CATEGORIES, predictions)},
            'note': 'Experimental one-week-ahead forecast; historical data ends in 2019.'}
