# Pharmaceutical Sales Forecasting & AI Demand Prediction

**End-to-end Data Science + AI Engineering portfolio project** — explore weekly pharmaceutical sales, evaluate forecasting models, serve a trained Random Forest through FastAPI, and generate historical one-week-ahead forecasts in a deployed Streamlit dashboard.

[![Live Dashboard](https://img.shields.io/badge/Live-Streamlit%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://pharma-sales-forecasting-ai.streamlit.app/)
[![API Docs](https://img.shields.io/badge/API-FastAPI%20Docs-009688?logo=fastapi&logoColor=white)](https://pharma-forecast-api-vews.onrender.com/docs)
[![GitHub Source](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/youssefattafi/Pharmaceutical-Sales-Forecasting)

### 🚀 Quick links for recruiters

| Explore | Link |
|---|---|
| **Interactive AI forecasting dashboard** | [Launch Streamlit app](https://pharma-sales-forecasting-ai.streamlit.app/) |
| **Live prediction API documentation** | [Open FastAPI Swagger UI](https://pharma-forecast-api-vews.onrender.com/docs) |
| **API health endpoint** | [Check model service](https://pharma-forecast-api-vews.onrender.com/health) |
| **Data science notebook** | [Explore analysis.ipynb](analysis.ipynb) |
| **Training and evaluation** | [train.py](train.py) · [backtest.py](backtest.py) |
| **Model export** | [save_model.py](save_model.py) |
| **API implementation** | [api.py](api.py) |
| **Streamlit implementation** | [app.py](app.py) |
| **Model comparison CSV** | [outputs/model_comparison.csv](outputs/model_comparison.csv) |

> **Demo scope:** Data ends in 2019. The API predicts the week of **October 20, 2019**, using the latest 12 observed historical weeks. These are experimental historical forecasts, **not current pharmaceutical demand**, medical advice, or validated inventory recommendations. Render's free service may take time to wake up.

## What this project demonstrates

**Data Science:** Pandas data preparation, past-only lag/rolling features, time-aware train/test splitting, regression baselines, Random Forest and Ridge, error metrics, chronological holdout testing, and a separate walk-forward validation experiment.

**AI / ML Engineering:** Reproducible model export with metadata, model versioning, FastAPI request validation and inference, deployed REST API on Render, Streamlit frontend on Streamlit Community Cloud, and HTTP-based integration between frontend and backend.

## Architecture

```text
Historical weekly sales (data/salesweekly.csv)
                  |
         Feature engineering
                  |
       Train and evaluate models
         |                  |
    Evaluation CSVs      Saved Random Forest + metadata
         |                  |
         |               FastAPI /predict
         |                  |   hosted on Render
         +-------> Streamlit dashboard
                    hosted on Streamlit Cloud
```

## Working demo result

The deployed dashboard successfully displayed a forecast for **2019-10-20**, model **v1.0**, with predicted weekly quantity **197.81 for N02BE** and a comparison chart for eight drug categories. The prediction was generated through the deployed API.

The API accepts at least 12 weeks of dated historical quantities, with records shaped as `{"date": "YYYY-MM-DD", "quantities": {"M01AB": 1.0, "...": 0.0}}` and returns `forecast_week`, `model_version`, and `predicted_weekly_quantities`.

## Dataset and methodology

**Source:** [Pharma Sales Data — Milan Zdravković (Kaggle)](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data). Original **CC BY-NC 4.0** noncommercial licensing and attribution apply. Historical sales **quantities**, not revenue; no marketing or competitor variables.

- **302 weekly observations** in the supplied dataset; **290 modeled rows** after lag construction.
- Eight categories: M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06.
- Lag features from 1, 2, 4, 8, and 12 previous weeks, trailing averages, and cyclical calendar features.
- Chronological 80% train / 20% test split for the original experiment; no random shuffle.

### Original chronological holdout results

Macro averages across eight categories, from the previously executed local dataset:

| Model | MAE ↓ | RMSE ↓ | sMAPE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| Previous-week baseline | 13.31 | 17.17 | 33.39% | -0.24 |
| Four-week seasonal baseline | 17.14 | 22.67 | 38.66% | -0.62 |
| Ridge Regression | 12.85 | 16.48 | 31.76% | -0.07 |
| **Random Forest** | **12.13** | **16.33** | **29.90%** | **-0.06** |

Random Forest has the lowest average MAE/RMSE in this experiment, **but the negative average R² signals limited predictive strength**. These figures are for the original chronological holdout, not the later walk-forward experiment.

### Separate walk-forward development experiment

A four-fold expanding-window development evaluation was executed separately, with the latest 58 weeks reserved from that experiment. Its reported **development** averages (not final holdout scores):

| Model | MAE ↓ | RMSE ↓ | sMAPE ↓ |
|---|---:|---:|---:|
| Random Forest | **11.575** | **14.623** | **33.330%** |
| Previous-week baseline | 11.908 | 14.811 | 40.430% |
| Histogram Gradient Boosting | 12.687 | 15.569 | 35.254% |
| Ridge | 13.431 | 16.595 | 38.774% |

See [backtest.py](backtest.py) for the implementation. Do not compare these numbers directly with the original holdout scores: the evaluation windows differ.

## Run locally

```bash
git clone https://github.com/youssefattafi/Pharmaceutical-Sales-Forecasting.git
cd Pharmaceutical-Sales-Forecasting
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn joblib
python train.py
python save_model.py
```

Start the API in one terminal:

```bash
uvicorn api:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to inspect/test endpoints. In another terminal, launch the dashboard:

```bash
PHARMA_API_URL=http://127.0.0.1:8000 streamlit run app.py
```

The dashboard also works without an API URL, but its interactive forecast button requires the API service. For the deployed Streamlit app, configure `PHARMA_API_URL` as an environment variable pointing to the Render base URL (no trailing `/predict`).

## Repository map

| Path | Purpose |
|---|---|
| [analysis.ipynb](analysis.ipynb) | Data science exploration |
| [train.py](train.py) | Original chronological model training and holdout evaluation |
| [backtest.py](backtest.py) | Separate walk-forward experiment |
| [save_model.py](save_model.py) | Export trained Random Forest and model metadata |
| [api.py](api.py) | FastAPI health and prediction endpoints |
| [app.py](app.py) | Interactive Streamlit dashboard |
| [data/salesweekly.csv](data/salesweekly.csv) | Historical weekly quantities |
| [outputs/](outputs/) | Saved evaluation results |

## Limitations and responsible use

The historical data comes from a limited source and ends in 2019. This is a one-week-ahead demonstration conditioned on observed history; it does not forecast 2026 demand or demonstrate production-grade monitoring, authentication, automated CI/CD, or clinical suitability. The API is publicly accessible as a portfolio demo. The saved model is generated from the repository's data at deployment/startup rather than retrieved from a production model registry.

## More portfolio projects

- [Amazon Fine Food Reviews — Sentiment Analysis AI](https://github.com/youssefattafi/Amazon-Fine-Food-Reviews-Analysis) · [Live NLP demo](https://amazon-review-sentiment-ai.streamlit.app/)
- [Hotel Booking Data Analysis](https://github.com/youssefattafi/Hotel-Booking-Data-Analysis) · [Live analytics dashboard](https://hotel-booking-analytics-ai.streamlit.app/)

**Author:** [Youssef Douieb El Attafi](https://github.com/youssefattafi)
