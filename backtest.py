"""Walk-forward, one-week-ahead backtesting for eight drug categories.

Each test row uses actual sales from earlier weeks (not recursively predicted sales).
Run: python backtest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent
CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]

def load_features():
    data = pd.read_csv(ROOT / "data" / "salesweekly.csv", parse_dates=["datum"])
    data = data.sort_values("datum").reset_index(drop=True)
    if data["datum"].duplicated().any():
        raise ValueError("Duplicate weekly dates found")
    features = pd.DataFrame(index=data.index)
    for category in CATEGORIES:
        for lag in (1, 2, 4, 8, 12):
            features[f"{category}_lag_{lag}"] = data[category].shift(lag)
        features[f"{category}_avg4"] = data[category].shift(1).rolling(4).mean()
    week = data["datum"].dt.isocalendar().week.astype(int)
    features["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    features["week_cos"] = np.cos(2 * np.pi * week / 52.18)
    valid = features.notna().all(axis=1)
    return data.loc[valid].reset_index(drop=True), features.loc[valid].reset_index(drop=True)

def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.divide(2 * np.abs(actual - predicted), denominator,
                              out=np.zeros_like(actual, dtype=float), where=denominator != 0)) * 100
    return dict(MAE=mean_absolute_error(actual, predicted),
                RMSE=np.sqrt(mean_squared_error(actual, predicted)),
                sMAPE_pct=smape)

def main():
    data, features = load_features()
    # Reserve the latest 58 weeks for untouched final evaluation in train.py.
    development_end = len(data) - 58
    if development_end < 150:
        raise ValueError("Insufficient development history for backtesting")
    splitter = TimeSeriesSplit(n_splits=4, test_size=20)
    rows = []
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(features.iloc[:development_end]), 1):
        x_train, x_val = features.iloc[train_idx], features.iloc[validation_idx]
        for category in CATEGORIES:
            y_train = data.loc[train_idx, category]
            y_val = data.loc[validation_idx, category]
            baseline = data.loc[validation_idx, category].shift(0) * 0 + features.loc[validation_idx, f"{category}_lag_1"]
            models = {
                "Previous-week baseline": baseline.to_numpy(),
                "Ridge": Ridge(alpha=100.0),
                "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1),
                "Histogram Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, l2_regularization=10.0, random_state=42),
            }
            for name, model in models.items():
                if name == "Previous-week baseline":
                    prediction = model
                else:
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_val)
                rows.append(dict(fold=fold, category=category, model=name,
                                 train_end=str(data.loc[train_idx[-1], "datum"].date()),
                                 validation_start=str(data.loc[validation_idx[0], "datum"].date()),
                                 validation_end=str(data.loc[validation_idx[-1], "datum"].date()),
                                 **metrics(y_val, prediction)))
    result = pd.DataFrame(rows)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    result.to_csv(output / "walk_forward_results.csv", index=False)
    summary = result.groupby("model")[["MAE", "RMSE", "sMAPE_pct"]].mean().sort_values("MAE")
    print("Walk-forward development results (not final holdout):")
    print(summary.round(3).to_string())
    print("\nSaved outputs/walk_forward_results.csv")

if __name__ == "__main__":
    main()
