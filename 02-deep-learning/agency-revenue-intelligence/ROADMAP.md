# Project 3 Roadmap — Focused Portfolio Scope

**Project:** Agency Revenue Intelligence & Forecasting Platform  
**Branch:** `feature/agency-revenue-intelligence`  
**Status:** Phase 1 — data engineering and validation  
**Estimated effort:** 17–24 days at 2–3 hours per day

## Project objective

Build a reproducible portfolio project that forecasts two distinct business outcomes:

1. **Agency recognized revenue** — monthly retainer revenue plus project-fee revenue earned by the agency.
2. **Campaign-attributed revenue** — daily sales attributed to clients' marketing campaigns.

These targets must remain separate. Campaign-attributed sales are not agency revenue.

## Scope and deliverables

### Phase 1 — Data engineering and validation (in progress)

- [x] Synthetic data generator for clients, contracts, invoices, payments, campaigns, agency-monthly metrics and campaign-daily metrics.
- [x] Candidate dataset generation in `data/processed_candidate/`.
- [x] Acquisition timing, churn eligibility, campaign timing and complete 36-month agency series refinements.
- [x] Separate channel-level click-to-lead, lead-to-customer and average-order-value assumptions in the campaign generator.
- [ ] Evaluate campaign ROAS and funnel conversion rates.
- [ ] Run the complete data-quality validation suite against the latest candidate datasets.
- [ ] Approve candidate data and promote it to `data/processed/`.
- [ ] Document synthetic-data assumptions and limitations.

**Data safety:** Keep original processed datasets unchanged until candidate validation passes. Synthetic records must remain clearly labelled. Do not commit credentials, private client data, or generated datasets unless explicitly intended.

### Phase 2 — EDA and feature engineering

Deliver two focused notebooks:

- Agency financial performance: revenue trends, retainers, project fees, client activity, acquisition, churn and seasonality.
- Campaign performance: spend, impressions, clicks, leads, conversions, attributed revenue, channel performance and ROAS.

Build reusable, leakage-safe features including calendar features, lags, rolling statistics and relevant business drivers. Use chronological splits; never use future information to construct training features.

### Phase 3 — Forecasting models

**Agency monthly recognized revenue**
- Seasonal Naive baseline
- SARIMA
- XGBoost
- LSTM

**Campaign daily attributed revenue**
- Naive baseline
- XGBoost
- LSTM

Compare models using chronological train/validation/test splits. Tune only where justified by validation results. Report MAE, RMSE, WAPE and forecast bias. Do not assume a deep-learning model is superior to a simpler baseline.

### Phase 4 — Dashboard and explainability

Build one Streamlit application that presents:

- Agency revenue actuals and forecasts.
- Campaign revenue and channel-level ROAS.
- Actual-versus-predicted comparisons.
- Forecast intervals where supported by the selected model.
- XGBoost feature explanations using SHAP.
- One simple assumption-based scenario simulator.

Clearly label forecasts and scenarios as estimates, not guarantees. Do not imply that observational campaign data proves causal effects of budget changes.

### Phase 5 — Finalization

- Dockerize the application.
- Add focused automated tests for data validation, feature generation and inference.
- Add a project README, data dictionary and architecture diagram.
- Include reproducible setup/run instructions and representative screenshots.
- Record model comparison results and project limitations.

## Deliberately out of scope

To keep the project achievable and focused, the following are postponed or excluded:

- Kafka / Redpanda and real-time event streaming.
- PostgreSQL and database migrations.
- FastAPI model-serving layer.
- Complex production monitoring and alerting.
- Separate production scheduling infrastructure.
- Extensive CI/CD and cloud deployment.
- Multiple competing deep-learning architectures such as GRU, TCN and Transformer.
- Extensive multi-variable scenario simulation.

These can be added later only if they solve a demonstrated need.

## Current dataset snapshot

Latest candidate generation reported:

| Dataset | Rows |
|---|---:|
| Clients | 50 |
| Contracts | 99 |
| Invoices | 945 |
| Payments | 891 |
| Campaigns | 100 |
| Agency monthly | 36 |
| Campaign daily | 20,051 |

These row counts confirm generation only; they do **not** constitute final data approval. Campaign economics and the full validation suite are still pending.

## Completion criteria

The project is complete when:

1. Candidate data passes documented integrity and business-rule checks.
2. Both forecasting targets have a baseline and model comparison using chronological evaluation.
3. The Streamlit dashboard loads validated model artifacts and displays forecasts accurately.
4. Tests and Docker setup work from documented instructions.
5. README, assumptions, evaluation results and limitations are documented.
