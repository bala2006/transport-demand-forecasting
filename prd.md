# Transportation Demand Forecasting Using Machine Learning

> **Project Type:** Open-source research toolkit  
> **Target Audience:** ML practitioners, transportation researchers, open-source contributors  
> **License:** MIT  
> **Status:** Draft v2

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Research Questions](#3-research-questions)
4. [Related Work & Gap](#4-related-work--gap)
5. [Dataset](#5-dataset)
6. [System Architecture](#6-system-architecture)
7. [Technical Stack](#7-technical-stack)
8. [Model Plan](#8-model-plan)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Explainability](#10-explainability)
11. [Project Structure](#11-project-structure)
12. [Installation & Usage](#12-installation--usage)
13. [How to Contribute](#13-how-to-contribute)
14. [Roadmap](#14-roadmap)
15. [Deliverables](#15-deliverables)
16. [License](#16-license)
17. [References](#17-references)

---

## 1. Project Overview

An open-source, reproducible framework for short-term transportation demand forecasting with built-in model interpretability. The toolkit lets you:

- **Download & process** public NYC taxi trip data into hourly demand counts per zone
- **Train & compare** 5 models: Linear Regression, Random Forest, XGBoost, LSTM, TCN
- **Explain predictions** with SHAP and feature importance
- **Extend** to your own city or dataset

All code is MIT-licensed. Contributions, forks, and adaptations are welcome.

---

## 2. Problem Statement

Transportation demand forecasting is a core challenge for urban mobility:

- **Planning:** scheduling vehicles and staff
- **Congestion reduction:** dynamic pricing, traffic management
- **Smart mobility:** fleet placement for ride-hailing and bike-share
- **Public transport:** frequency adjustments based on ridership

Traditional methods (historical averages, ARIMA) miss complex patterns from weather, holidays, and nonlinear seasonality. ML captures these but sacrifices interpretability.

**Can we build accurate demand forecasts while keeping predictions explainable?**

This repo provides a benchmark — implement, compare, and explain multiple models on a single pipeline so the community can build on it.

---

## 3. Research Questions

1. How do traditional ML (Linear Regression, RF, XGBoost) compare with deep sequence models (LSTM, TCN)?
2. What accuracy gap exists between simple and complex models?
3. Which features drive demand most? Do they match domain knowledge?
4. Can SHAP produce meaningful explanations for time-series demand forecasts?
5. What is the minimal viable feature set for reliable prediction?

---

## 4. Related Work & Gap

### 4.1 Existing Work

- **Statistical:** ARIMA, SARIMA — interpretable, weak on nonlinear patterns (Hyndman & Athanasopoulos, 2021)
- **ML:** RF, XGBoost — strong on tabular time-series (Yusuf et al., 2025)
- **Deep learning:** LSTM, TCN — capture long-range dependencies (Bai et al., 2018)
- **Explainability:** SHAP on time-series is growing but still underexplored in transportation (Shukla, 2025)

### 4.2 Gap This Project Fills

| Gap | How We Address It |
|-----|-------------------|
| Most studies focus on accuracy, not interpretability | SHAP + feature importance for every model |
| No single pipeline comparing ML + DL on transport data | 5 models, identical train/val/test splits |
| Few reproducible end-to-end open-source implementations | MIT license, fully scripted pipeline, public data |
| Hard to onboard new researchers | Notebooks as tutorials, modular `src/` code |

---

## 5. Dataset

### Primary: NYC TLC Trip Records

| Attribute | Detail |
|-----------|--------|
| **Source** | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Format** | Parquet, monthly |
| **Time span** | 2022–2024 (~500M trips) |
| **License** | Public domain / NYC Open Data |
| **Key columns** | pickup_datetime, PULocationID, DOLocationID, trip_distance, passenger_count |

### Feature Engineering

| Category | Features |
|----------|----------|
| **Temporal** | hour, day_of_week, month, is_weekend, sin/cos cyclic encoding |
| **Holiday** | US federal holidays, day_after_holiday |
| **Weather** | (NOAA GSOD) temp, precipitation, wind, snow flags |
| **Location** | pickup_zone, borough (from taxi_zone_lookup.csv) |
| **Lag & rolling** | demand_lag_1h, demand_lag_24h, rolling_mean_1h, rolling_mean_3h |

### Aggregation

Raw trip records → **hourly demand per pickup zone** (trips/zone/hour).

### Fallback

Use **Citi Bike** data or synthetic data if TLC is too large for your machine. See `data/README.md`.

---

## 6. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       INPUT                               │
│  TLC Trips + Zone Lookup + NOAA Weather + Holidays       │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    PREPROCESSING                          │
│  Filter → Aggregate → Merge → Feature engineer → Split   │
│  (chronological 70/15/15, fit scaler on train only)      │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│                      MODELS                              │
│  ┌──────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────┐    │
│  │ LinearReg│ │  RF  │ │XGBoost │ │ LSTM │ │ TCN │    │
│  └──────────┘ └──────┘ └────────┘ └──────┘ └──────┘    │
│                    Hyperparameter tuning                 │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│              EVALUATION + EXPLANATION                    │
│  Metrics (MAE/RMSE/MAPE/R²) + SHAP + Feature Importance  │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│                      OUTPUT                              │
│  comparison table · figures · SHAP plots · saved models  │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Technical Stack

| Layer | Choice |
|-------|--------|
| **Language** | Python 3.11+ |
| **Data** | Pandas, NumPy, Polars (Parquet) |
| **ML** | scikit-learn, XGBoost |
| **DL** | PyTorch |
| **XAI** | SHAP |
| **Plots** | Matplotlib, Seaborn |
| **Tests** | pytest |
| **CI** | GitHub Actions (lint + test) |
| **Env** | pip + requirements.txt, conda environment.yml |
| **Docs** | README + inline docstrings |

---

## 8. Model Plan

### 8.1 Models

| Model | Category | Why |
|-------|----------|-----|
| Linear Regression | Baseline | Fast, interpretable, lower bound |
| Random Forest | Tree ensemble | Nonlinear, built-in importance |
| XGBoost | Gradient boosted trees | SOTA for tabular data |
| LSTM | Recurrent | Long-range temporal patterns |
| TCN | Convolutional | Parallelizable, matches LSTM accuracy |

### 8.2 Data Hygiene

- Chronological split only (no shuffle)
- Lag features use past data only
- Scaling parameters fit on training set

---

## 9. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **MAE** | Average absolute error (same units as demand) |
| **RMSE** | Penalizes large outliers |
| **MAPE** | Percentage error, scale-independent |
| **R²** | Variance explained |
| **Actual vs predicted plot** | Visual pattern match |

---

## 10. Explainability

### 10.1 Methods

| Model | Built-in | Post-hoc |
|-------|----------|----------|
| Linear Regression | Coefficients | SHAP |
| Random Forest | `feature_importances_` | SHAP |
| XGBoost | `feature_importances_` | SHAP |
| LSTM | — | SHAP (DeepExplainer) |
| TCN | — | SHAP (DeepExplainer) |

### 10.2 Outputs

- **Global SHAP bar** — feature importance across all predictions
- **Beeswarm summary** — feature effect direction + spread
- **Force plots** — why a single prediction is high/low
- **Dependence plots** — feature value vs SHAP value

---

## 11. Project Structure

```
transport-demand-forecasting/
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # GitHub Actions: test + lint
│   ├── ISSUE_TEMPLATE/
│   └── CONTRIBUTING.md
├── data/
│   ├── raw/                    # Original TLC parquet files
│   ├── processed/              # Aggregated + features
│   └── external/               # Weather, holidays, zone lookup
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_lstm_tcn.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_shap_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── features.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── lstm_model.py
│   │   ├── tcn_model.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── plots.py
│   └── explain/
│       └── shap_analysis.py
├── outputs/
│   ├── figures/
│   ├── models/
│   └── results/
├── tests/
│   ├── test_preprocess.py
│   └── test_models.py
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE                     # MIT
├── README.md
├── prd.md                      # This document
├── requirements.txt
└── environment.yml
```

---

## 12. Installation & Usage

```bash
git clone https://github.com/<your-org>/transport-demand-forecasting
cd transport-demand-forecasting
pip install -r requirements.txt

# Quick start: run the full pipeline
python src/pipeline.py --start 2024-01 --end 2024-03

# Or step through notebooks
jupyter notebook notebooks/
```

### Configuration

Edit `config.yaml` to set:
- Data date range
- Target zones
- Model selection
- Hyperparameter grid

---

## 13. How to Contribute

We welcome contributions of all sizes.

### Ways to contribute

- **Report a bug** — open an issue
- **Suggest a feature** — start a discussion
- **Add a model** — implement in `src/models/`, add it to the comparison
- **Add a dataset adapter** — create a new data source in `src/data/`
- **Improve docs** — README, docstrings, notebook annotations
- **Translate to another city** — swap TLC for your local transport data

### Quick start

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push (`git push origin feat/my-feature`)
5. Open a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

### Code standards

- Follow PEP 8
- Type hints for all public functions
- Tests for new features (pytest, >80% coverage target)
- Notebooks should be clean (Cell → Restart & Run All before PR)

---

## 14. Roadmap

### Phase 1: Core pipeline (current)

- [x] EDA notebook
- [x] Feature engineering module
- [x] 5 model implementations
- [x] Evaluation metrics
- [x] SHAP integration

### Phase 2: Community hardening

- [ ] GitHub Actions CI (lint + pytest)
- [ ] Pre-commit hooks
- [ ] Sphinx / mkdocs documentation site
- [ ] PyPI package (`pip install transport-forecast`)
- [ ] CLI entry point

### Phase 3: Extensions

- [ ] Multi-step forecasting (predict N hours ahead)
- [ ] Spatial component (graph neural network)
- [ ] Adapters for other cities:
  - Chicago taxi data
  - London Oyster card data
  - Tokyo train ridership
- [ ] Real-time inference mode
- [ ] Interactive dashboard (Streamlit / Gradio)

### Phase 4: Advanced research

- [ ] Probabilistic forecasting (quantile regression, Bayesian)
- [ ] Anomaly detection on residuals
- [ ] Causal analysis (beyond correlation)
- [ ] Transformer-based time series (PatchTST, TimesNet)

---

## 15. Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | GitHub repo with README | ✅ |
| 2 | Modular `src/` package | ✅ |
| 3 | 6 Jupyter notebooks | ✅ |
| 4 | Evaluation comparison table | ✅ |
| 5 | SHAP explanation outputs | ✅ |
| 6 | MIT License | ✅ |
| 7 | Contributing guide | 🔲 |
| 8 | CI/CD pipeline | 🔲 |
| 9 | PyPI package | 🔲 |
| 10 | Documentation site | 🔲 |

---

## 16. License

MIT License — see [LICENSE](LICENSE).

You are free to use, modify, distribute, and sublicense. Attribution appreciated but not required.

---

## 17. References

1. Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
2. Yusuf, O., Rasheed, A. & Lindseth, F. (2025). Data-driven predictive modelling of stop-level public transit patterns. *Transportation*.
3. MDPI (2023). Deep Learning and Statistical Models for Forecasting Transportation Demand. *Logistics*, 7(4), 86.
4. Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. Shukla, M.A. (2025). Interpreting Time Series Forecasts with LIME and SHAP. *TechRxiv*.
6. Bai, S., Kolter, J.Z. & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
7. NYC TLC Trip Record Data. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
8. NOAA Global Surface Summary of Day. https://www.ncei.noaa.gov/data/global-summary-of-the-day/
9. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

---

> **Version:** 2.0  
> **Updated:** June 2026  
> **Maintainer:** [Your Name]
