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
