# Agency Revenue Intelligence & Forecasting Platform

A data science portfolio project for analyzing and forecasting a digital marketing agency's recognized revenue and its clients' campaign-attributed revenue. All datasets are synthetic and generated for this project.

## Project overview

The platform analyzes two distinct business targets:

1. **Agency recognized revenue (monthly):** Retainer revenue plus project-fee revenue earned by the agency.
2. **Campaign-attributed revenue (daily):** Client sales attributed to advertising campaigns.

These measures represent different business outcomes. Campaign-attributed sales are not agency revenue and must not be combined or double-counted.

## Technology stack

- **Language:** Python
- **Data processing:** Pandas, NumPy, PyArrow
- **Machine learning:** Scikit-learn, XGBoost
- **Statistical forecasting:** Statsmodels (SARIMA)
- **Deep learning:** PyTorch (LSTM)
- **Visualization:** Plotly
- **Dashboard:** Streamlit
- **Testing:** Pytest
- **Deployment preparation:** Docker

## Forecasting methodology

The project compares baseline, statistical, machine-learning, and deep-learning forecasting approaches.

| Forecast target | Baseline | Statistical / ML | Deep learning |
|---|---|---|---|
| Monthly agency revenue | Seasonal Naive | SARIMA, XGBoost | LSTM |
| Daily campaign revenue | 7-day Naive | XGBoost | LSTM |

### Evaluation methodology

- Chronological train, validation, and test partitions.
- Model selection based on validation WAPE.
- Final evaluation on a separate held-out test period.
- Recursive forecasting for XGBoost and LSTM models.
- Metrics: MAE, RMSE, WAPE, and signed forecast bias.

The monthly dataset contains only 36 observations. Model comparisons are therefore exploratory, and strong annual seasonality cannot be established reliably from this short history.

## Forecast evaluation results

The pipeline was executed successfully using the locally generated candidate datasets.

### Validation results

| Target | Model | WAPE |
|---|---|---:|
| Monthly agency revenue | SARIMA | 5.94% |
| Monthly agency revenue | Seasonal Naive | 39.35% |
| Monthly agency revenue | XGBoost | 33.07% |
| Monthly agency revenue | LSTM | 28.52% |
| Daily campaign revenue | 7-day Naive | 42.61% |
| Daily campaign revenue | XGBoost | 48.85% |
| Daily campaign revenue | LSTM | 57.93% |

### Held-out test results

| Target | Selected model | MAE (INR) | RMSE (INR) | WAPE | Forecast bias |
|---|---|---:|---:|---:|---:|
| Monthly agency revenue | SARIMA | 1,206,698 | 1,360,499 | 16.45% | +16.45% |
| Daily campaign revenue | 7-day Naive | 144,783 | 163,441 | 54.24% | +50.16% |

**Interpretation:**

- SARIMA achieved the best validation performance for monthly agency revenue, but its WAPE increased from 5.94% on validation to 16.45% on the held-out test period.
- The 7-day Naive baseline was the best validation model for daily campaign revenue.
- Daily campaign forecasting performance deteriorated on the held-out test period. This indicates limited generalization and is an area for further improvement.
- XGBoost and LSTM did not outperform the selected baselines in these experiments.

Forecast bias is signed: positive values indicate aggregate overforecasting, while negative values indicate aggregate underforecasting.

These results are specific to the synthetic datasets and evaluation period. They are not evidence of production forecasting accuracy.

## Streamlit dashboard

The dashboard includes three main sections:

1. **Agency revenue:** Monthly revenue composition, retainer revenue, project fees, and client activity.
2. **Campaign performance:** Daily campaign spend and attributed revenue, channel-level ROAS, and conversion funnel metrics.
3. **Forecast evaluation:** Validation model comparison, held-out test metrics, and actual-versus-predicted visualizations.

An illustrative campaign budget scenario calculator uses historical blended ROAS. It is an arithmetic scenario, not a causal estimate of incremental revenue.

## Project setup

Run these commands from the repository root.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r 02-deep-learning/agency-revenue-intelligence/requirements.txt
```

### 3. Generate synthetic datasets

```bash
python 02-deep-learning/agency-revenue-intelligence/src/generate_synthetic_data.py \
  --output-dir 02-deep-learning/agency-revenue-intelligence/data/processed_candidate
```

The generator writes to `data/processed_candidate` and does not overwrite the original processed dataset directory.

### 4. Navigate to the project directory

```bash
cd 02-deep-learning/agency-revenue-intelligence
```

### 5. Run tests

```bash
python -m pytest -q
```

The current local test run completed with **26 passing tests**.

### 6. Run the forecasting pipeline

```bash
python -m src.train_forecasts \
  --data-dir data/processed_candidate \
  --output-dir reports/forecast_results
```

The pipeline generates:

- `reports/forecast_results/validation_metrics.csv`
- `reports/forecast_results/test_metrics.csv`
- `reports/forecast_results/test_predictions.csv`

Additional fitted model artifacts may be generated locally and are not required to be committed.

### 7. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

The dashboard uses the candidate dataset when available and reads its forecast evaluation files from `reports/forecast_results/`.

## Repository structure

```text
agency-revenue-intelligence/
├── configs/
│   └── project.yaml
├── dashboard/
│   └── app.py
├── data/
│   ├── processed/
│   └── processed_candidate/
├── docs/
│   └── data_dictionary.md
├── models/
├── notebooks/
│   └── 01_agency_revenue_eda.ipynb
├── reports/
│   └── forecast_results/
├── src/
│   ├── data_access.py
│   ├── explainability.py
│   ├── features.py
│   ├── forecasting.py
│   ├── generate_synthetic_data.py
│   ├── metrics.py
│   └── train_forecasts.py
├── tests/
├── Dockerfile
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## Data definitions and safeguards

- All client, contract, invoice, payment, and campaign records are synthetic.
- Recognized agency revenue is distinct from invoice value and cash collected.
- ROAS = campaign-attributed revenue / advertising spend.
- ROAS is not profit, gross margin, or contribution margin.
- Lag and rolling features use prior observations to reduce future-data leakage.
- Campaign data is split by unique calendar date to prevent records from the same date appearing across different partitions.

See [docs/data_dictionary.md](docs/data_dictionary.md) for dataset fields and metric definitions.

## Limitations and future improvements

- The monthly forecasting history is short.
- Daily campaign forecasts currently have relatively high test error.
- Synthetic data may not reflect real agency revenue patterns or advertising behavior.
- Historical ROAS does not establish causal advertising effectiveness.
- SHAP explanations and Docker execution require further validation.

See [ROADMAP.md](ROADMAP.md) for the implementation status and remaining work.

## Responsible use

This is an educational portfolio project. Do not present its synthetic records or model results as actual company financial data or production-grade forecasts. Do not commit credentials, private datasets, or generated model binaries.
Git reports \ No newline at end of file