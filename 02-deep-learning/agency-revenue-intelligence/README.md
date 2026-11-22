# Agency Revenue Intelligence & Forecasting Platform

A focused portfolio project for analyzing and forecasting a digital marketing agency's recognized revenue and its clients' campaign-attributed revenue. All records are synthetic.

> **Implementation status:** Reusable data access, leakage-aware time-series features, baseline/SARIMA/XGBoost/LSTM forecasting code, a Streamlit dashboard, Dockerfile and focused tests are now present. Run the commands below locally before treating model comparisons or dashboard output as validated results. No model-performance claim is made in this README.

## Business targets

1. **Agency recognized revenue (monthly):** retainer revenue plus project-fee revenue earned by the agency.
2. **Campaign-attributed revenue (daily):** client sales attributed to advertising campaigns.

These are distinct measures. Campaign-attributed sales are not agency revenue and must never be combined or double-counted.

## Modeling scope

| Target | Baseline | Statistical / ML | Deep learning |
|---|---|---|---|
| Agency monthly recognized revenue | Seasonal Naive | SARIMA, XGBoost | LSTM |
| Campaign daily attributed revenue | 7-day Naive | XGBoost | LSTM |

Evaluation uses chronological train/validation/test partitions and MAE, RMSE, WAPE and signed forecast bias. The monthly series has only 36 observations; treat model comparisons as exploratory and do not infer robust annual seasonality from this short history.

## Local setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r 02-deep-learning/agency-revenue-intelligence/requirements.txt
```

Generate synthetic data into a candidate directory. This does not overwrite the original processed directory:

```bash
python 02-deep-learning/agency-revenue-intelligence/src/generate_synthetic_data.py \
  --output-dir 02-deep-learning/agency-revenue-intelligence/data/processed_candidate
```

Run the project tests from the project directory:

```bash
cd 02-deep-learning/agency-revenue-intelligence
python -m pytest -q
```

Run the forecasting comparison from the project directory:

```bash
python -m src.train_forecasts
```

The pipeline writes validation metrics, held-out test metrics and test predictions to `reports/generated/`. Model artifacts are local and ignored by Git.

Launch the dashboard from the project directory:

```bash
streamlit run dashboard/app.py
```

The dashboard expects the generated Parquet datasets and, for its evaluation tab, the forecast outputs created by the training command.

## Docker

From the repository root, after generating data locally:

```bash
docker build -f 02-deep-learning/agency-revenue-intelligence/Dockerfile -t agency-revenue-intelligence .
docker run --rm -p 8501:8501 \
  -v "$(pwd)/02-deep-learning/agency-revenue-intelligence/data:/app/data" \
  -v "$(pwd)/02-deep-learning/agency-revenue-intelligence/reports:/app/reports" \
  agency-revenue-intelligence
```

Open `http://localhost:8501`.

## Data and responsible interpretation

- All client, contract, invoice, payment and campaign records are synthetic.
- `recognized_agency_revenue` is not the same as invoice value or cash collected.
- ROAS = campaign-attributed revenue / advertising spend. ROAS is not profit.
- Scenario outputs use a fixed historical blended ROAS and are illustrative arithmetic, not causal budget-response forecasts.
- Do not commit credentials, private data, Parquet datasets, or generated model binaries.

See [ROADMAP.md](./ROADMAP.md) and [docs/data_dictionary.md](./docs/data_dictionary.md) for scope, metric definitions and completion criteria.
