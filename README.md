# Transportation Demand Forecasting

A complete machine learning project that predicts how many taxi rides will happen in New York City, hour by hour, for each taxi zone. It downloads **real NYC taxi data** (publicly available from the NYC Taxi & Limousine Commission), cleans it, creates features, trains 5 different AI models, compares their accuracy, and explains how each model makes its predictions using SHAP.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [How the Pipeline Works (Simple Overview)](#how-the-pipeline-works-simple-overview)
3. [Prerequisites](#prerequisites)
4. [Installation (Step by Step)](#installation-step-by-step)
5. [Running the Pipeline](#running-the-pipeline)
6. [Understanding the Pipeline Steps](#understanding-the-pipeline-steps)
   - [Step 1: Download Data](#step-1-download-data)
   - [Step 2: Preprocess](#step-2-preprocess)
   - [Step 3: Feature Engineering](#step-3-feature-engineering)
   - [Step 4: Train/Test Split](#step-4-traintest-split)
   - [Step 5: Train Models](#step-5-train-models)
   - [Step 6: Compare Results](#step-6-compare-results)
   - [Step 7: Explain Predictions (SHAP)](#step-7-explain-predictions-shap)
7. [Understanding the Outputs](#understanding-the-outputs)
   - [outputs/results/](#outputsresults)
   - [outputs/figures/](#outputsfigures)
   - [outputs/models/](#outputsmodels)
8. [How to Read Each Plot](#how-to-read-each-plot)
   - [Actual vs Predicted Plot](#1-actual-vs-predicted-plot)
   - [SHAP Summary Plot](#2-shap-summary-plot)
   - [SHAP Feature Importance Bar Plot](#3-shap-feature-importance-bar-plot)
   - [SHAP Waterfall Plot](#4-shap-waterfall-plot-force-plot)
9. [Understanding the Features](#understanding-the-features)
10. [Understanding the Models](#understanding-the-models)
11. [Understanding the Metrics](#understanding-the-metrics)
12. [Customizing the Pipeline (config.yaml)](#customizing-the-pipeline-configyaml)
13. [Running the Notebooks (Interactive Tutorial)](#running-the-notebooks-interactive-tutorial)
14. [Running Tests](#running-tests)
15. [Project Structure](#project-structure)
16. [Troubleshooting](#troubleshooting)

---

## What This Project Does

This project answers the question: **"How many taxi pickups will happen in each NYC zone during the next hour?"**

It uses historical NYC taxi trip data (yellow taxis) and trains machine learning models to predict future demand. The predictions help taxi companies, city planners, and drivers understand when and where demand will be highest.

### Why This Matters

- **Taxi companies** can position more cars in high-demand areas
- **City planners** can monitor transportation patterns
- **Drivers** can work when and where demand is highest
- **Students and developers** can learn how ML pipelines work end-to-end

---

## How the Pipeline Works (Simple Overview)

```
Raw NYC taxi data  →  Download  →  Clean & aggregate  →  Create features  →  Train models  →  Evaluate  →  Explain
     (1.5 GB/mo)         Step 1          Step 2                Step 3            Step 5         Step 6        Step 7
```

**What happens at each step:**

1. **Download** — Fetches trip records from the NYC TLC website (about 1.5 GB per month of data)
2. **Clean** — Removes invalid records, keeps only datetime and location
3. **Aggregate** — Counts how many trips started in each zone per hour
4. **Create features** — Adds time-of-day, day-of-week, holiday flags, past demand, weather data
5. **Scale** — Standardizes all numerical features to have zero mean and unit variance
6. **Split** — Divides data chronologically: 70% training, 15% validation, 15% testing
7. **Train** — Fits 5 models on the training data
8. **Evaluate** — Computes accuracy metrics on the test data
9. **Explain** — Uses SHAP to show why each model made its predictions

---

## Prerequisites

### Required Software

| Software | Why You Need It | How to Check |
|----------|----------------|--------------|
| **Python 3.10+** | The language the project is written in | `python --version` |
| **pip** | Python's package installer (comes with Python) | `pip --version` |
| **Terminal / Command Prompt** | To run commands | Already have one open |

### How to Check Your Python Version

Open a terminal and type:

```bash
python --version
```

- If you see `Python 3.10.x`, `3.11.x`, `3.12.x`, or higher: you are ready.
- If you see `Python 2.7.x`: install Python 3 from [python.org](https://www.python.org/downloads/).
- If you see "command not found": install Python from [python.org](https://www.python.org/downloads/).

> **Windows users**: During installation, check **"Add Python to PATH"**. This makes the `python` command work in the terminal.

---

## Installation (Step by Step)

### Step 1: Get the Project Files

**Option A — Clone with Git** (recommended if you have Git installed):

```bash
git clone <repository-url>
cd transport-demand-forecasting
```

**Option B — Download ZIP**:

1. Go to the GitHub repository page
2. Click the green "Code" button → "Download ZIP"
3. Extract the ZIP file
4. Open a terminal in the extracted folder

### Step 2: Install Required Libraries

All Python libraries needed are listed in `requirements.txt`. Install them all with one command:

```bash
pip install -r requirements.txt
```

This installs the following libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.24 | Numerical computations (arrays, math) |
| **pandas** | ≥2.0 | Data manipulation in tables |
| **polars** | ≥0.20 | Fast data processing (like pandas but faster) |
| **scikit-learn** | ≥1.3 | Machine learning models & preprocessing |
| **xgboost** | ≥2.0 | High-performance gradient boosting model |
| **torch** | ≥2.0 | Deep learning framework (LSTM, TCN) |
| **shap** | ≥0.44 | Explains model predictions |
| **matplotlib** | ≥3.7 | Creates charts and saves images |
| **seaborn** | ≥0.12 | Makes matplotlib charts look better |
| **pyyaml** | ≥6.0 | Reads the config.yaml configuration file |
| **requests** | ≥2.31 | Downloads data from the internet |
| **pytest** | ≥7.4 | Runs automated tests |

> If you get permission errors, try: `pip install --user -r requirements.txt`

### Step 3: Verify Installation

```bash
python -c "import torch, shap, xgboost; print('All good!')"
```

If you see "All good!" printed, everything is installed correctly.

---

## Running the Pipeline

The entire pipeline runs from a single command:

```bash
python -m src.pipeline
```

This runs the project as a Python module (the `-m` flag) starting from `src/pipeline.py`.

> **Important**: Always run this command from the **project root folder** (the one containing `src/`, `config.yaml`, and `README.md`).

### What You Will See on Screen

As the pipeline runs, it prints progress updates. Here is the actual output from one run:

```
=== Step 1: Download data ===
Downloading https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet...
Saved data\raw\yellow_tripdata_2024-01.parquet
Downloading zone lookup...
Saved data\external\taxi_zone_lookup.csv

=== Step 2: Preprocess ===
Saved aggregated data to data\processed\hourly_demand.parquet

=== Step 3: Feature engineering ===
Feature matrix: (1858, 28)

=== Step 4: Chronological split ===

=== Step 5: Train models ===

--- Linear Regression ---
--- Random Forest ---
--- XGBoost ---
--- LSTM ---
Epoch 10/50 - train_loss=23.7170, val_loss=68.5767
Epoch 20/50 - train_loss=21.2569, val_loss=66.6155
Early stopping at epoch 24
--- TCN ---
Early stopping at epoch 9

=== Step 6: Results ===
                     MAE    RMSE     MAPE      R²
Model
LinearRegression  2.0984  4.3551  86.3608  0.7456
RandomForest      1.9298  4.7064  73.5452  0.7028
XGBoost           1.9283  4.6853  77.9659  0.7055
LSTM              2.6965  7.9944  96.5476  0.2128
TCN               2.5624  7.2568  93.9446  0.3514

=== Step 7: SHAP explanations ===
100%|##########| 100/100
Pipeline complete!
```

### Estimated Runtime

| Setting | Time |
|---------|------|
| 1 month data (January 2024), all 5 models | 3-5 minutes |
| 3 months data, all 5 models | 8-15 minutes |
| All models except LSTM/TCN | 1-2 minutes |

---

## Understanding the Pipeline Steps

### Step 1: Download Data

The pipeline downloads **NYC yellow taxi trip records** from the official NYC Taxi & Limousine Commission (TLC) website.

**What is in the raw data?** (each file is ~1.5 GB per month)

Each row is a single taxi trip with these columns used:

| Column | Example | Description |
|--------|---------|-------------|
| `tpep_pickup_datetime` | 2024-01-15 08:30:00 | When the trip started |
| `PULocationID` | 74 | The pickup zone ID (number 1-263) |
| `tpep_dropoff_datetime` | 2024-01-15 08:45:00 | When the trip ended |
| `passenger_count` | 2 | How many passengers |
| `trip_distance` | 3.2 | Distance in miles |
| `total_amount` | 15.50 | Fare + tip + tax |

**Data source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**Download code**: `src/data/download.py` — it uses `requests` to fetch `.parquet` files and saves them to `data/raw/`.

The pipeline also downloads a **zone lookup table** that maps Location IDs (numbers like 74) to human-readable zone names (like "Murray Hill") and boroughs ("Manhattan").

### Step 2: Preprocess

The raw trip data is too granular (each trip is one row) — we need to **aggregate** it into hourly demand counts per zone.

**What happens in this step:**

1. **Filter** — Keeps only the pickup datetime and location ID columns
2. **Truncate timestamps** — Rounds each timestamp down to the nearest hour (e.g., 08:30 → 08:00)
3. **Aggregate** — Counts how many trips started in each zone during each hour. This creates the **demand** column (our target variable)
4. **Join zone names** — Adds human-readable zone names and boroughs from the lookup table
5. **Save** — Writes the result to `data/processed/hourly_demand.parquet`

**Code**: `src/data/preprocess.py`

**Example output (hourly demand):**

| pickup_hour | PULocationID | demand | pickup_zone | borough |
|-------------|--------------|--------|-------------|---------|
| 2024-01-01 00:00:00 | 74 | 12 | Murray Hill | Manhattan |
| 2024-01-01 01:00:00 | 74 | 8 | Murray Hill | Manhattan |
| 2024-01-01 02:00:00 | 74 | 3 | Murray Hill | Manhattan |
| ... | ... | ... | ... | ... |

### Step 3: Feature Engineering

Machine learning models cannot just use the raw datetime — they need **features** (numerical values) that capture patterns in time.

**The pipeline creates 17 features** from the raw data using `src/data/features.py`:

#### Temporal Features (from the datetime)

| Feature Name | How It Is Calculated | Why It Matters |
|-------------|----------------------|----------------|
| `hour` | `pickup_hour.hour` (0-23) | Demand changes by time of day |
| `day_of_week` | `pickup_hour.weekday()` (1=Mon, 7=Sun) | Weekdays vs weekends |
| `month` | `pickup_hour.month()` (1-12) | Seasonal patterns |
| `is_weekend` | True if day_of_week is 6 or 7 | Weekends have different patterns |
| `hour_sin` | `sin(hour × 2π / 24)` | Cyclical encoding of hour (midnight = 0, noon = 1) |
| `hour_cos` | `cos(hour × 2π / 24)` | Cyclical encoding (midnight = 1, noon = -1) |
| `dow_sin` | `sin((day-1) × 2π / 7)` | Cyclical encoding of day of week |
| `dow_cos` | `cos((day-1) × 2π / 7)` | Cyclical encoding of day of week |
| `month_sin` | `sin((month-1) × 2π / 12)` | Cyclical encoding of month |
| `month_cos` | `cos((month-1) × 2π / 12)` | Cyclical encoding of month |

> **Why sin/cos?** Time is cyclical — hour 23 and hour 0 are similar (both at night), but a model sees them as 23 vs 0 (very different numbers). Sin/cos encoding preserves this closeness by mapping time onto a circle.

#### Holiday Features

| Feature | Description |
|---------|-------------|
| `is_holiday` | True if the date is a US federal holiday (New Year's, July 4th, Thanksgiving, etc.) |
| `day_after_holiday` | True if the previous day was a holiday |

Holidays affect demand significantly — fewer business trips, more leisure travel.

#### Lag Features (Past Demand)

These are the **most important** features. They capture the recent history of demand for each zone.

| Feature | What It Means |
|---------|---------------|
| `demand_lag_1h` | How many pickups happened **1 hour ago** in this zone |
| `demand_lag_2h` | How many pickups happened **2 hours ago** |
| `demand_lag_3h` | How many pickups happened **3 hours ago** |
| `demand_lag_24h` | How many pickups happened **exactly 24 hours ago** (yesterday at same time) |
| `rolling_mean_1h` | Average demand over the **last 1 hour** |
| `rolling_mean_3h` | Average demand over the **last 3 hours** |
| `rolling_mean_6h` | Average demand over the **last 6 hours** |

> The 24-hour lag captures **daily seasonality** — if there were 50 pickups yesterday at 5 PM, there will likely be ~50 again today at 5 PM.

#### Weather Features

| Feature | Description |
|---------|-------------|
| `temp_c` | Temperature in Celsius (defaults to 20°C if no weather file) |
| `precipitation_mm` | Rainfall in millimeters (defaults to 0) |
| `wind_speed_kph` | Wind speed in km/h (defaults to 10) |
| `snow_flag` | 1 if snowing, 0 otherwise (defaults to 0) |

Bad weather (rain, snow, extreme cold) affects taxi demand significantly.

### Step 4: Train/Test Split

The data is split **chronologically** (not randomly) because we are predicting future demand from past data:

| Split | Percentage | Purpose |
|-------|-----------|---------|
| **Training** | 70% (earliest data) | The model learns patterns from this data |
| **Validation** | 15% (middle) | Used during training to check for overfitting |
| **Test** | 15% (latest data) | Final evaluation — model has never seen this data |

The data is also **scaled** using `StandardScaler`: each feature is transformed so that its mean is 0 and its standard deviation is 1. This is required for neural networks (LSTM, TCN) and helps other models converge faster.

### Step 5: Train Models

Five models are trained on the same training data. Three are "classic" ML models (fast to train) and two are deep learning models (slower but can capture more complex patterns).

**Training code**: `src/models/train.py`
**Model definitions**: `src/models/baseline.py`, `src/models/lstm_model.py`, `src/models/tcn_model.py`

### Step 6: Compare Results

Each model is evaluated on the holdout **test set** (data it has never seen). Four metrics are computed:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **R²** — R-squared (coefficient of determination)

Results are printed as a table and saved to `outputs/results/comparison.csv` and `outputs/results/metrics.json`.

### Step 7: Explain Predictions (SHAP)

**SHAP** (SHapley Additive exPlanations) is a technique that explains why a model made a specific prediction for a specific row. It is based on game theory — each feature gets a "fair share" of credit for pushing the prediction up or down.

For each of the 3 non-deep-learning models (Linear Regression, Random Forest, XGBoost), the pipeline:

1. Creates a **SHAP explainer** for the model
2. Computes SHAP values for the first 100 test samples
3. Generates 3 types of plots:
   - **Summary plot** — shows which features matter most overall
   - **Bar plot** — feature importance ranked by average SHAP value
   - **Waterfall plots** — explains 3 individual predictions feature by feature

**Code**: `src/explain/shap_analysis.py`

---

## Understanding the Outputs

After the pipeline finishes, everything is saved in the `outputs/` folder.

### outputs/results/

Contains performance metrics comparing all models.

#### comparison.csv

A CSV file (readable in Excel or any text editor). Example:

```csv
Model,MAE,RMSE,MAPE,R²
LinearRegression,2.0984,4.3551,86.3608,0.7456
RandomForest,1.9298,4.7064,73.5452,0.7028
XGBoost,1.9283,4.6853,77.9659,0.7055
LSTM,2.6965,7.9944,96.5476,0.2128
TCN,2.5624,7.2568,93.9446,0.3514
```

What this tells us (from the sample run):
- **XGBoost** has the lowest MAE (1.9283), making it the most accurate on average
- **Linear Regression** has the highest R² (0.7456), meaning it explains more variance
- **LSTM** and **TCN** perform worse — the dataset may be too small for deep learning to shine
- **MAPE is high (73-96%)** — this is because demand can be very low (0-2 trips) in some zones/ times, and dividing by a small number makes the percentage large

#### metrics.json

Same data as comparison.csv but in JSON format (machine-readable).

```json
{
  "LinearRegression": {
    "MAE": 2.0984,
    "RMSE": 4.3551,
    "MAPE": 86.3608,
    "R²": 0.7456
  },
  ...
}
```

### outputs/figures/

Contains all generated plots. There are 18 images total (3 models × 6 plot types):

| File Pattern | Example | Content |
|-------------|---------|---------|
| `*_actual_vs_predicted.png` | `RandomForest_actual_vs_predicted.png` | Scatter plot comparing predictions to actual values |
| `*_shap_summary.png` | `RandomForest_shap_summary.png` | SHAP dot plot showing feature importance |
| `*_shap_bar.png` | `RandomForest_shap_bar.png` | Feature importance as average absolute SHAP value |
| `*_force_plot_0.png` | `XGBoost_force_plot_0.png` | Waterfall plot explaining first prediction |
| `*_force_plot_1.png` | `XGBoost_force_plot_1.png` | Waterfall plot explaining second prediction |
| `*_force_plot_2.png` | `XGBoost_force_plot_2.png` | Waterfall plot explaining third prediction |

### outputs/models/

Saved trained models that can be reloaded later for predictions without retraining.

| File | Model | Format |
|------|-------|--------|
| `linear_regression.pkl` | Linear Regression | PyTorch pickle |
| `random_forest.pkl` | Random Forest | PyTorch pickle |
| `lstm.pt` | LSTM weights | PyTorch state dict |
| `tcn.pt` | TCN weights | PyTorch state dict |

> XGBoost and TCN models are not saved by default. You can add `torch.save()` calls in `pipeline.py` if needed.

---

## How to Read Each Plot

### 1. Actual vs Predicted Plot

**File**: `outputs/figures/{ModelName}_actual_vs_predicted.png`

**What it shows**: For each test sample, the x-axis is the actual demand and the y-axis is what the model predicted.

**How to interpret it**:

```
Predicted
    ↑
    |    .   .  .   .
    |   .   .   . .
    |  .   . .   .
    | .   .   .
    |.   . .
    +------------------------→ Actual
```

- **Perfect predictions** — all dots would fall on the red dashed diagonal line (Actual = Predicted)
- **Dots above the line** — model over-predicted (predicted more than actually happened)
- **Dots below the line** — model under-predicted (predicted less than actually happened)
- **Tight cluster around the line** — good model
- **Wide scatter** — poor model (predictions are inaccurate)

### 2. SHAP Summary Plot

**File**: `outputs/figures/{ModelName}_shap_summary.png`

**What it shows**: For every feature, each dot is a single prediction. The x-axis is the SHAP value (impact on prediction). The dot color shows whether the feature value was high (red) or low (blue).

**How to read it**:

```
demand_lag_1h     ●●●●●●●●●●●●●●●●●●●●
                 ●●●●●●●●●●●●●●●●●●●●●
hour_sin          ●●●●●●●●●●●●●●●●●●●●●
                 ●●●●●●●●●●●●●●●●●●●●●
dow_sin         ●●●●●●●●●●●●●●●●●●●●●●
                ●●●●●●●●●●●●●●●●●●●●●▲
rolling_mean_1h   ●●●●●●●●●●●●●●●●●
                   ●●●●●●●●●●●●●●●●●
                  ●●●●●●●●●●●●●●●●●
weather_temp_c   ●●●●●●●●●●●●●●●
                 ●●●●●●●●●●●●●●●
                  ●●●●●●●●●●●●●
        ← lower     SHAP value (impact)    higher →
```

- **Features at the top** are the most important
- **Dots spread wide** = that feature has a big impact on predictions
- **Red dots on the right** = high feature values push predictions up
- **Blue dots on the right** = low feature values push predictions up
- **Red dots on the left** = high feature values push predictions down
- **Blue dots on the left** = low feature values push predictions down

**Example interpretation**: If `demand_lag_1h` has red dots spread widely to the right, it means: when demand was high 1 hour ago, the model predicts higher demand now. This makes intuitive sense — demand is correlated over time.

### 3. SHAP Feature Importance Bar Plot

**File**: `outputs/figures/{ModelName}_shap_bar.png`

**What it shows**: A simple bar chart of average |SHAP value| for each feature. Taller bars = more important features.

```text
demand_lag_1h      |██████████████████████|  (most important)
hour_sin           |███████████████       |
demand_lag_2h      |████████████         |
rolling_mean_1h    |███████████          |
dow_sin            |█████████            |
hour_cos           |███████              |
demand_lag_3h      |██████               |
weather_temp_c     |████                 |
...
```

This is the simplest way to see which features matter most.

### 4. SHAP Waterfall Plot (Force Plot)

**File**: `outputs/figures/{ModelName}_force_plot_0.png`

**What it shows**: A single prediction explained feature by feature. The waterfall shows the base value (average prediction) at the bottom, and each feature pushes the prediction up (red) or down (blue) until we reach the final prediction at the top.

**How to read it**:

```text
f(x) = 28.4                     ← Final prediction (what model predicted for this sample)
                                ╱
   +5.2   ← demand_lag_1h = 12  ← This high lag value pushed prediction up 5.2
  /
 -3.1   ← hour_sin = 0.7       ← This feature value pushed prediction down 3.1
/
  +2.8  ← rolling_mean_3h = 8  ← Pushed prediction up 2.8
/
 -1.4  ← dow_cos = -0.9       ← Pushed prediction down 1.4
/
...
╲
  22.5                           ← Base value (average prediction in the dataset)
```

**What each part means**:
- **Base value** (bottom): The average prediction across all samples. This is where the model starts before considering any features.
- **Red arrows pointing right/up**: Features that pushed the prediction higher than average
- **Blue arrows pointing left/down**: Features that pushed the prediction lower than average
- **Final prediction** (top): The model's actual prediction for this sample

**Example**: If the base value is 22.5 and:
- `demand_lag_1h = 12` (high) → +5.2 → prediction goes up
- `hour_sin = 0.7` (it is around 2 PM) → -3.1 → prediction goes down
- `rolling_mean_3h = 8` → +2.8 → prediction goes up

Final = 22.5 + 5.2 - 3.1 + 2.8 - 1.4 + ... = 28.4

This tells you exactly why the model predicted 28.4 instead of the average 22.5.

---

## Understanding the Features

Here is the complete list of features the models use, in order of importance (based on SHAP analysis from sample data):

| Rank | Feature | Type | What It Captures | Typical SHAP Impact |
|------|---------|------|------------------|-------------------|
| 1 | `demand_lag_1h` | Lag | How much demand there was 1 hour ago | ±5-8 |
| 2 | `hour_sin` | Cyclical | Time of day | ±3-5 |
| 3 | `demand_lag_2h` | Lag | How much demand 2 hours ago | ±2-4 |
| 4 | `rolling_mean_1h` | Rolling | Short-term demand trend | ±2-3 |
| 5 | `dow_sin` | Cyclical | Day of week effect | ±1-3 |
| 6 | `hour_cos` | Cyclical | Time of day (complement) | ±1-2 |
| 7 | `demand_lag_3h` | Lag | How much demand 3 hours ago | ±1-2 |
| 8 | `rolling_mean_3h` | Rolling | Medium-term trend | ±1-2 |
| 9 | `demand_lag_24h` | Lag | Same time yesterday | ±1-2 |
| 10 | `dow_cos` | Cyclical | Day of week (complement) | ±0.5-1 |
| 11 | `rolling_mean_6h` | Rolling | Longer-term trend | ±0.5-1 |
| 12 | `temp_c` | Weather | Temperature | ±0.5-1 |
| 13 | `month_sin` | Cyclical | Month of year | ±0-0.5 |
| 14 | `month_cos` | Cyclical | Month of year (complement) | ±0-0.5 |
| 15 | `is_weekend` | Boolean | Whether it is Saturday/Sunday | ±0-0.5 |
| 16 | `is_holiday` | Boolean | Whether it is a holiday | ±0-0.5 |
| 17 | `day_after_holiday` | Boolean | Day after a holiday | ±0-0.5 |

**Key insight**: The **lag features** (past demand) are by far the most important. The model primarily predicts future demand by looking at recent demand. Cyclical time features (hour of day, day of week) are also important. Weather and holiday features have less impact because they are based on defaults.

---

## Understanding the Models

### 1. Linear Regression

| Property | Detail |
|----------|--------|
| **Type** | Linear regression |
| **Training time** | < 1 second |
| **How it works** | Finds a straight-line relationship: `demand = w1×f1 + w2×f2 + ... + bias` |
| **Strengths** | Fast, interpretable (you can see each feature's weight/coefficient) |
| **Weaknesses** | Cannot capture non-linear patterns or feature interactions |
| **Best for** | Simple baseline, understanding linear relationships |

### 2. Random Forest

| Property | Detail |
|----------|--------|
| **Type** | Ensemble of decision trees |
| **Training time** | 2-10 seconds |
| **How it works** | Builds 50 decision trees (each on a random subset of data and features), averages their predictions |
| **Strengths** | Handles non-linear patterns, works well on tabular data, resistant to overfitting |
| **Weaknesses** | Cannot extrapolate beyond training data range |
| **Best for** | General purpose tabular data |

### 3. XGBoost

| Property | Detail |
|----------|--------|
| **Type** | Gradient boosted trees |
| **Training time** | 2-10 seconds |
| **How it works** | Builds trees sequentially, each new tree tries to correct the errors of all previous trees |
| **Strengths** | Usually the most accurate tree-based model, handles missing values, built-in regularization |
| **Weaknesses** | Can overfit if not tuned properly |
| **Best for** | Competition-grade tabular data predictions |

### 4. LSTM (Long Short-Term Memory)

| Property | Detail |
|----------|--------|
| **Type** | Recurrent neural network |
| **Training time** | 30-120 seconds |
| **How it works** | Uses 2 LSTM layers (64 hidden units each) followed by a linear output layer. Takes a sequence of 24 past time steps as input and predicts the next time step |
| **Strengths** | Designed for time series, can learn long-term dependencies (weeks/months) |
| **Weaknesses** | Needs lots of data, slower to train, hyperparameters matter a lot |
| **Best for** | Long time series with complex temporal patterns |

### 5. TCN (Temporal Convolutional Network)

| Property | Detail |
|----------|--------|
| **Type** | Convolutional neural network for time series |
| **Training time** | 20-60 seconds |
| **How it works** | Uses dilated 1D convolutions with channels [32, 64, 128] and kernel size 3 to process the input sequence |
| **Strengths** | More stable training than LSTM, can be faster |
| **Weaknesses** | May require more data than LSTM to reach peak performance |
| **Best for** | Time series where training stability matters |

---

## Understanding the Metrics

The pipeline computes 4 metrics to evaluate model performance. All metrics compare the model's predictions to the actual values.

### MAE (Mean Absolute Error)

**Formula**: `MAE = average(|actual - predicted|)`

**Example**: If MAE = 2.1, on average the model's prediction is off by 2.1 taxi trips.

| MAE Value | Interpretation |
|-----------|---------------|
| 0 | Perfect predictions |
| 1-2 | Very good (off by 1-2 trips on average) |
| 2-3 | Good |
| 3-5 | Acceptable |
| > 5 | Poor |

### RMSE (Root Mean Squared Error)

**Formula**: `RMSE = sqrt(average((actual - predicted)²))`

RMSE is similar to MAE but **penalizes large errors more heavily** (because it squares the error before averaging).

**Example**: If one prediction is off by 10 trips and all others are off by 1, MAE = ~1.1 but RMSE = ~3.0. The high RMSE alerts you that there are some very bad predictions.

| RMSE vs MAE | What It Tells You |
|-------------|-------------------|
| RMSE ≈ MAE | Errors are evenly distributed (all about the same size) |
| RMSE > MAE | There are some large errors (outliers) |

### MAPE (Mean Absolute Percentage Error)

**Formula**: `MAPE = average(|(actual - predicted) / actual|) × 100`

This tells you the error as a percentage of the actual value.

**Important caveat**: When actual demand is 0 or very small (1-2 trips), even a small absolute error becomes a huge percentage error. This is why MAPE can be high (70-100%) even when MAE is low (2-3 trips).

### R² (R-squared / Coefficient of Determination)

**Formula**: `R² = 1 - (sum of squared errors / total variance)`

**Interpretation**:

| R² Value | Meaning |
|----------|---------|
| 1.0 | Perfect predictions (model explains all variance) |
| 0.7-0.9 | Good model (explains 70-90% of variance) |
| 0.5-0.7 | Moderate |
| 0.2-0.5 | Weak |
| 0.0 | Model is no better than predicting the average every time |
| < 0 | Model is worse than predicting the average |

R² is useful because it is scale-independent (you can compare across different datasets).

---

## Customizing the Pipeline (config.yaml)

The file `config.yaml` controls everything about the pipeline. Here is how to customize it.

### Data Settings

```yaml
data:
  years: [2024]           # Which years to download
  months: [1]             # Which months (1=Jan, 2=Feb, ..., 12=Dec)
  target_zones: null      # null = all zones, or [1,2,3] = specific zone IDs
  weather_path: null      # path to a weather CSV, or null for defaults
```

**Examples:**

```yaml
# Use 3 months of data (January through March 2024)
data:
  years: [2024]
  months: [1, 2, 3]

# Only predict for Manhattan zones (1-10)
data:
  target_zones: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Use custom weather data
data:
  weather_path: "data/external/weather_2024.csv"
```

### Model Settings

```yaml
models:
  active: ["lr", "rf", "xgb", "lstm", "tcn"]   # Which models to train
  seq_length: 24                                 # Sequence length for LSTM/TCN (hours of history)
```

To train only specific models, remove from the list:

```yaml
# Only train Linear Regression and XGBoost (skip RF, LSTM, TCN)
models:
  active: ["lr", "xgb"]
```

### Hyperparameters

You can adjust each model's settings:

```yaml
models:
  hyperparameters:
    rf:
      n_estimators: 50          # Number of trees (more = better but slower)
      max_depth: 10             # Maximum tree depth (deeper = more complex)
      min_samples_leaf: 5       # Minimum samples per leaf (higher = less overfitting)

    xgb:
      n_estimators: 50          # Number of boosting rounds
      max_depth: 6              # Tree depth
      learning_rate: 0.1        # Step size shrinkage (lower = more robust)
      subsample: 0.8            # Fraction of samples per tree
      colsample_bytree: 0.8     # Fraction of features per tree

    lstm:
      hidden_size: 64           # Number of LSTM units
      num_layers: 2             # Number of LSTM layers
      dropout: 0.2              # Dropout rate (higher = less overfitting)
      epochs: 50                # Max training epochs
      batch_size: 256           # Samples per batch
      lr: 0.001                 # Learning rate

    tcn:
      num_channels: [32, 64, 128]  # Channels per layer
      kernel_size: 3               # Convolution kernel size
      dropout: 0.2
      epochs: 50
      batch_size: 256
      lr: 0.001
```

### Output Settings

```yaml
output:
  figures_dir: "outputs/figures"   # Where plots are saved
  models_dir: "outputs/models"     # Where trained models are saved
  results_dir: "outputs/results"   # Where metrics CSVs are saved
```

---

## Running the Notebooks (Interactive Tutorial)

If you prefer to learn step by step with explanations and code cells, use the Jupyter notebooks.

### Start Jupyter

```bash
jupyter notebook notebooks/
```

This opens a browser window showing the notebooks folder.

### Notebook Order

| # | Notebook | What You Will Learn | Code You Will Run |
|---|----------|---------------------|-------------------|
| 01 | `01_eda.ipynb` | Explore raw taxi data | Load parquet files, make histograms, check missing values |
| 02 | `02_feature_engineering.ipynb` | How features are created | Create temporal features, lag features, check correlations |
| 03 | `03_baseline_models.ipynb` | Train LR, RF, XGBoost | Fit models, print metrics, make predictions |
| 04 | `04_lstm_tcn.ipynb` | Train LSTM and TCN | Prepare sequences, train neural networks, plot loss curves |
| 05 | `05_evaluation.ipynb` | Compare all models | Load results, make comparison plots, analyze errors |
| 06 | `06_shap_analysis.ipynb` | Understand model predictions | Compute SHAP values, make summary/waterfall plots |

> **Tip**: Run notebooks in order (01 through 06). Each builds on the previous one.

---

## Running Tests

Automated tests verify that the core functions work correctly:

```bash
pytest tests/
```

These tests check:
- **Preprocessing** — filtering trips, aggregating hourly demand, splitting data
- **Model training** — models can be trained and make predictions

---

## Project Structure

```
transport-demand-forecasting/
|
|-- config.yaml                      # MAIN SETTINGS FILE
|   Change this to control data, models, and outputs.
|
|-- requirements.txt                 # Python libraries to install (pip install)
|
|-- README.md                        # This documentation file
|
|-- src/                             # ALL SOURCE CODE (the brain of the project)
|   |
|   |-- pipeline.py                  # Main entry point — runs all 7 steps
|   |   Run with: python -m src.pipeline
|   |
|   |-- data/                        # DATA LAYER — gets and prepares data
|   |   |-- download.py              # Downloads NYC taxi parquet files & zone lookup
|   |   |-- preprocess.py            # Cleans trips, aggregates hourly demand
|   |   |-- features.py              # Creates temporal/lag/holiday/weather features
|   |
|   |-- models/                      # MODEL LAYER — defines and trains models
|   |   |-- baseline.py              # Linear Regression, Random Forest, XGBoost
|   |   |-- lstm_model.py            # LSTM neural network architecture
|   |   |-- tcn_model.py             # TCN neural network architecture
|   |   |-- train.py                 # Shared training & evaluation functions
|   |
|   |-- evaluation/                  # EVALUATION LAYER — measures performance
|   |   |-- metrics.py               # Computes MAE, RMSE, MAPE, R-squared
|   |   |-- plots.py                 # Actual vs predicted, residuals, time series
|   |
|   |-- explain/                     # EXPLAINABILITY LAYER — interpret predictions
|       |-- shap_analysis.py         # SHAP explainer, summary plots, waterfall plots
|
|-- notebooks/                       # TUTORIAL NOTEBOOKS (interactive learning)
|   |-- 01_eda.ipynb                 # Explore raw data
|   |-- 02_feature_engineering.ipynb # Create features
|   |-- 03_baseline_models.ipynb     # Train basic models
|   |-- 04_lstm_tcn.ipynb            # Train deep learning models
|   |-- 05_evaluation.ipynb          # Evaluate and compare
|   |-- 06_shap_analysis.ipynb       # Explain predictions
|
|-- outputs/                         # GENERATED OUTPUTS (created by the pipeline)
|   |-- figures/                     # Charts and plots (18 PNG files)
|   |   |-- LinearRegression_actual_vs_predicted.png
|   |   |-- RandomForest_shap_summary.png
|   |   |-- XGBoost_force_plot_0.png
|   |   |-- ... (18 files total)
|   |
|   |-- models/                      # Saved trained models
|   |   |-- linear_regression.pkl
|   |   |-- random_forest.pkl
|   |   |-- lstm.pt
|   |   |-- tcn.pt
|   |
|   |-- results/                     # Metrics and comparisons
|       |-- comparison.csv           # Side-by-side model comparison table
|       |-- metrics.json             # Machine-readable metrics
|
|-- data/                            # DATA FILES (downloaded by the pipeline)
|   |-- raw/                         # Raw NYC taxi parquet files (~1.5 GB/month)
|   |-- processed/                   # Cleaned and aggregated hourly demand
|   |-- external/                    # Zone lookup table, weather data
|
|-- tests/                           # AUTOMATED TESTS
    |-- test_preprocess.py           # Tests for data preprocessing
    |-- test_models.py               # Tests for model training
```

---

## Troubleshooting

### "No module named 'src'"

**Problem**: Python cannot find the `src` module.

**Solution**: You are running from the wrong directory. Run from the project root:

```bash
cd transport-demand-forecasting   # Go to the project root
python -m src.pipeline            # Run from there
```

The `-m` flag tells Python to look for `src.pipeline` as a module. It only works if your terminal is in the project root folder.

### "pip install" Fails with Permission Errors

**Problem**: You do not have permission to install packages system-wide.

**Solution**: Install only for your user:

```bash
pip install --user -r requirements.txt
```

Or use a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### CUDA / GPU Errors (LSTM/TCN Training)

**Problem**: Error message about CUDA or GPU when training LSTM/TCN.

**Solution**: The models work fine on CPU. If you get a CUDA-related error, force CPU:

```python
# In pipeline.py, set device='cpu' explicitly
device = torch.device('cpu')
```

If you have a compatible NVIDIA GPU and want to use it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Pipeline Is Slow

**Problem**: Training takes too long.

**Solutions** (from fastest to most effective):

1. **Use less data**: Change `months: [1]` in config.yaml (only January)
2. **Train fewer models**: Set `active: ["lr", "rf", "xgb"]` (skip LSTM and TCN)
3. **Reduce epochs**: Lower `epochs: 10` in the LSTM/TCN hyperparameters
4. **Reduce SHAP samples**: Edit `n_samples` in `src/explain/shap_analysis.py`

### "disk full" or "no space" Errors

**Problem**: Each month of raw NYC taxi data is ~1.5 GB.

**Solution**: Delete the raw data after processing:

```bash
rm -rf data/raw/*.parquet
```

### "ConnectionError" Download Fails

**Problem**: Cannot download the NYC taxi data (no internet, or website is down).

**Solution**: The pipeline automatically falls back to **synthetic data** if download fails. The synthetic data simulates realistic demand patterns using a Poisson distribution with daily seasonality. You can also manually use synthetic mode:

```python
# In preprocess.py, the generate_synthetic_data() function creates fake data
```

### SHAP Is Very Slow

**Problem**: The SHAP step takes a long time.

**Solution**: Reduce the number of samples:

```python
# In src/explain/shap_analysis.py, change:
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_explain[:50])  # Reduced from 100 to 50
```

### Models Have High MAPE ( > 100 )

**Problem**: MAPE is very high even though MAE looks reasonable.

**Explanation**: This is normal. MAPE divides each error by the actual value. When actual demand is 0, 1, or 2 (common for taxi zones at 3 AM), even a small error of 1 trip becomes 50-100% error. **Focus on MAE and R² instead.**

---

## License

MIT — Free to use, modify, and distribute.
