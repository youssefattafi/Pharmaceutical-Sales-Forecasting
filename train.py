from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
CATS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]

def prepare():
    df = pd.read_csv(ROOT / "data" / "salesweekly.csv", parse_dates=["datum"]).sort_values("datum").reset_index(drop=True)
    for c in CATS:
        for lag in (1, 2, 4, 8, 12):
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
        df[f"{c}_rolling4"] = df[c].shift(1).rolling(4).mean()
    week = df.datum.dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.18)
    return df.dropna().reset_index(drop=True)

def main():
    df = prepare()
    features = [c for c in df.columns if c not in CATS + ["datum"]]
    cutoff = int(len(df) * .8)
    train, test = df.iloc[:cutoff], df.iloc[cutoff:]
    xtr, xte = train[features], test[features]
    ytr, yte = train[CATS], test[CATS]
    predictions = {
        "Previous week baseline": test[[f"{c}_lag1" for c in CATS]].to_numpy(),
        "Four-week seasonal baseline": test[[f"{c}_lag4" for c in CATS]].to_numpy(),
    }
    for name, model in [
        ("Ridge", Ridge(alpha=100.0)),
        ("Random Forest", RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1))
    ]:
        model.fit(xtr, ytr)
        predictions[name] = model.predict(xte)
    rows = []
    for name, pred in predictions.items():
        for j, cat in enumerate(CATS):
            actual = yte.iloc[:, j].to_numpy()
            mae = mean_absolute_error(actual, pred[:, j])
            rmse = np.sqrt(mean_squared_error(actual, pred[:, j]))
            denom = np.abs(actual) + np.abs(pred[:, j])
            smape = 100 * np.mean(np.divide(2 * np.abs(actual - pred[:, j]), denom, out=np.zeros_like(actual), where=denom != 0))
            rows.append({"model": name, "category": cat, "MAE": mae, "RMSE": rmse, "sMAPE_pct": smape, "R2": r2_score(actual, pred[:, j])})
    results = pd.DataFrame(rows)
    (ROOT / "outputs").mkdir(exist_ok=True)
    results.to_csv(ROOT / "outputs" / "model_comparison.csv", index=False)
    pd.DataFrame({"date": test.datum, "actual_N02BE": yte["N02BE"], **{f"{name}_N02BE": pred[:, 3] for name, pred in predictions.items()}}).to_csv(ROOT / "outputs" / "n02be_holdout_predictions.csv", index=False)
    print("Training weeks:", len(train), "Test weeks:", len(test))
    print(results.groupby("model")[["MAE", "RMSE", "sMAPE_pct", "R2"]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
