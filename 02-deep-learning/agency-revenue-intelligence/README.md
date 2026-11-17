# Agency Revenue Intelligence & Forecasting Platform

A focused, reproducible portfolio project for forecasting a digital marketing agency's financial revenue and its clients' campaign-attributed revenue.

> **Current status:** Phase 1 — data engineering and validation. Synthetic candidate datasets have been generated. Campaign economics evaluation and the complete validation suite are still pending; no forecasting model results are claimed.

## Business objectives

This project has two separate forecasting targets:

1. **Agency recognized revenue (monthly):** retainer revenue plus project-fee revenue earned by the agency.
2. **Campaign-attributed revenue (daily):** sales attributed to clients' marketing campaigns.

Campaign-attributed sales are not agency revenue. The two measures must not be combined or double-counted.

## Focused project scope

The project is intentionally limited to five phases:

1. **Data engineering and validation** — generate and validate synthetic agency and campaign data; document assumptions and approve candidate datasets.
2. **EDA and feature engineering** — analyze agency financial performance and campaign funnels; create leakage-safe time-series features.
3. **Forecasting models** — compare simple baselines, statistical models, machine learning and a focused deep-learning model.
4. **Dashboard and explainability** — build a Streamlit dashboard with forecasts, channel performance, SHAP explanations and a simple scenario simulator.
5. **Finalization** — add Docker, focused automated tests, architecture documentation and reproducible setup instructions.

See [ROADMAP.md](./ROADMAP.md) for phase-level tasks and completion criteria.

## Planned forecasting models

| Forecasting target | Baseline | Statistical / ML | Deep learning |
|---|---|---|---|
| Agency monthly recognized revenue | Seasonal Naive | SARIMA, XGBoost | LSTM |
| Campaign daily attributed revenue | Naive | XGBoost | LSTM |

Models will be evaluated using chronological train/validation/test splits and MAE, RMSE, WAPE and forecast bias. A complex model will not be considered better unless it demonstrates better out-of-sample performance than an appropriate baseline.

## Planned application

A single Streamlit application will present:

- Agency revenue actuals and forecasts.
- Campaign revenue, funnel metrics and channel-level ROAS.
- Actual-versus-predicted comparisons.
- Forecast intervals where supported.
- SHAP explanations for the XGBoost model.
- A simple assumption-based scenario simulator.

## Data strategy

All agency-specific client, contract, invoice and payment records are synthetic. Campaign data is also generated synthetically for this project. Synthetic records are labelled and must never be represented as real company financial data.

Original processed datasets are preserved while candidate datasets are being validated. Candidate data is not promoted until the complete validation suite passes.

### Latest candidate dataset snapshot

| Dataset | Rows |
|---|---:|
| Clients | 50 |
| Contracts | 99 |
| Invoices | 945 |
| Payments | 891 |
| Campaigns | 100 |
| Agency monthly | 36 |
| Campaign daily | 20,051 |

These counts confirm successful generation only; they do not indicate that the data has passed final approval.

## Repository structure

```text
agency-revenue-intelligence/
├── configs/                  # Project and experiment configuration
├── data/
│   ├── processed/            # Existing approved datasets; preserve until validation
│   └── processed_candidate/  # Candidate datasets awaiting validation
├── notebooks/                # EDA and model experiments
├── src/                      # Data generation, features, models and evaluation
├── dashboard/                # Streamlit application (planned)
├── tests/                    # Data, feature and inference tests
├── reports/                  # Evaluation reports and figures
├── models/                   # Local model artifacts; do not commit binaries
├── ROADMAP.md
└── README.md
```

## Deliberately out of scope

To keep the project achievable and portfolio-focused, Kafka/Redpanda, real-time streaming, PostgreSQL, FastAPI, complex production monitoring, separate scheduling infrastructure, extensive CI/CD and multiple additional deep-learning architectures are postponed.

## Evaluation and responsible interpretation

Forecasts are estimates, not guarantees. Scenario outputs depend on explicit assumptions; observational campaign data does not establish causal effects of changing advertising budgets. Financial reporting must distinguish recognized revenue, invoiced amounts, cash collected and client-attributed campaign sales.

## Local setup

Python 3.11 is the initial target. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r 02-deep-learning/agency-revenue-intelligence/requirements.txt
```

See the project configuration and [ROADMAP.md](./ROADMAP.md) for the current implementation sequence.
