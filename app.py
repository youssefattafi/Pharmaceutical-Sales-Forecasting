from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Pharma Sales Forecasting", layout="wide")
st.title("Pharmaceutical Sales Forecasting")
st.caption("Historical weekly quantities — experimental predictions, not revenue or medical guidance")

results_path = ROOT / "outputs" / "model_comparison.csv"
data_path = ROOT / "data" / "salesweekly.csv"
if not data_path.exists() or not results_path.exists():
    st.warning("Data/results are missing. Download salesweekly.csv from the linked dataset and run python train.py first.")
    st.stop()

results = pd.read_csv(results_path)
category = st.selectbox("Drug category", sorted(results.category.unique()), index=3)
st.subheader("Held-out model comparison")
st.dataframe(results[results.category == category].round(2), hide_index=True, use_container_width=True)
df = pd.read_csv(data_path, parse_dates=["datum"])
st.subheader("Historical weekly quantities")
st.line_chart(df.set_index("datum")[[category]])
pred_path = ROOT / "outputs" / "n02be_holdout_predictions.csv"
if category == "N02BE" and pred_path.exists():
    pred = pd.read_csv(pred_path, parse_dates=["date"]).set_index("date")
    st.subheader("N02BE: held-out actual vs model predictions")
    st.line_chart(pred)
st.info("Chronological holdout evaluation using prior observed weeks; this is one-step-ahead backtesting.")


st.divider()
st.subheader("AI Forecast — one week ahead")
st.caption("Uses the saved Random Forest through a FastAPI service. This is a historical demonstration, not a current-demand forecast.")
import requests
import os

api_url = os.getenv("PHARMA_API_URL", "").strip().rstrip("/")
if not api_url:
    st.info("Forecast API not connected yet. Deploy the FastAPI service and set PHARMA_API_URL in Streamlit app secrets/environment to enable live predictions.")
else:
    if st.button("Generate next-week forecast", type="primary"):
        categories = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
        history_df = df.sort_values("datum").tail(12)
        payload = {"history": [
            {"date": r["datum"].strftime("%Y-%m-%d"),
             "quantities": {c: float(r[c]) for c in categories}}
            for _, r in history_df.iterrows()
        ]}
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=25)
            response.raise_for_status()
            forecast = response.json()
            st.success(f"Forecast for week of {forecast['forecast_week']} · Model v{forecast['model_version']}")
            predictions = forecast["predicted_weekly_quantities"]
            st.metric(f"{category} predicted weekly quantity", f"{predictions[category]:,.2f}")
            st.bar_chart(pd.Series(predictions, name="Predicted quantity"))
            st.caption(forecast.get("note", "Experimental forecast."))
        except (requests.RequestException, ValueError, KeyError) as exc:
            st.error(f"Forecast service unavailable or returned an unexpected response: {exc}")
