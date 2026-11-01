"""Train and persist a versioned one-step-ahead forecasting model.

Run: python save_model.py
Only load joblib artifacts created by trusted code.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from backtest import CATEGORIES, load_features

ROOT = Path(__file__).resolve().parent

def main():
    data, features = load_features()
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1)
    model.fit(features, data[CATEGORIES])
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    path = folder / "random_forest.joblib"
    joblib.dump(model, path)
    metadata = {
        "model": "RandomForestRegressor",
        "version": "1.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "categories": CATEGORIES,
        "feature_names": features.columns.tolist(),
        "training_rows": len(features),
        "last_training_week": str(data["datum"].iloc[-1].date()),
        "forecast_type": "one-week-ahead with observed lagged demand",
        "notes": "Refitted on all available history after separate development backtesting. No future observations used as input.",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {path}")
    print(f"Saved {folder / 'metadata.json'}")
    print(f"Training weeks: {len(features)}")

if __name__ == "__main__":
    main()
