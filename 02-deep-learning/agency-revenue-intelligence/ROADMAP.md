# Project 3 Roadmap — Focused Portfolio Scope

**Project:** Agency Revenue Intelligence & Forecasting Platform  
**Branch:** `feature/agency-revenue-intelligence`

## Objective

Forecast two distinct business outcomes: monthly agency recognized revenue and daily client campaign-attributed revenue. Campaign sales are not agency revenue.

## Phase 1 — Data engineering and validation

- [x] Synthetic data generator and seven Parquet dataset outputs.
- [x] Candidate dataset generation performed locally.
- [x] Local structural/business-rule validation suite reported 16 passing tests.
- [x] Channel-level campaign funnel assumptions and ROAS diagnostics completed locally.
- [ ] Synchronize the latest locally refined generator and configuration to GitHub.
- [ ] Re-run the complete validation suite against the exact synchronized generator output.
- [ ] Approve and promote candidate data only after that run; preserve original processed data.
- [ ] Document final generator assumptions and limitations.

**Important:** The GitHub branch's generator is not yet the same as the locally refined generator. Do not treat remote generation as equivalent until the local refinements are synchronized and validated.

## Phase 2 — EDA and feature engineering

- [x] Agency monthly EDA notebook started locally; initial data-quality, descriptive-statistics and revenue-trend sections completed locally.
- [x] Reusable lag, rolling and calendar feature functions added.
- [ ] Finish and commit the local agency EDA notebook after synchronization review.
- [ ] Complete campaign EDA: funnel, channel spend, revenue, conversion rates and ROAS.
- [ ] Review leakage and feature availability for both targets.

Features use prior observations for lag and rolling calculations. Campaign partitions are split by unique date so records from the same date cannot leak across partitions.

## Phase 3 — Forecasting models

- [x] Seasonal Naive and 7-day Naive baselines.
- [x] SARIMA monthly forecasting implementation.
- [x] XGBoost recursive forecasting implementation for both targets.
- [x] PyTorch LSTM recursive forecasting implementation for both targets.
- [x] Chronological validation selection and held-out test evaluation pipeline.
- [ ] Run the pipeline locally and inspect failures/metrics.
- [ ] Add SHAP explanations for the selected XGBoost model where available.
- [ ] Record validated model comparison results; do not assume LSTM/XGBoost outperforms baselines.

Metrics: MAE, RMSE, WAPE and signed forecast bias.

## Phase 4 — Dashboard and explainability

- [x] Streamlit dashboard for agency revenue, client activity, campaign performance and channel ROAS.
- [x] Clearly labelled illustrative fixed-ROAS scenario calculator.
- [x] Test metrics and actual-vs-predicted results view.
- [ ] Run dashboard locally against generated data and review charts.
- [ ] Add and validate SHAP feature explanations for XGBoost.
- [ ] Capture representative screenshots after local verification.

Scenario output is arithmetic under a fixed historical ROAS assumption; it is not a causal estimate of incremental budget return.

## Phase 5 — Finalization

- [x] Dockerfile and reproducible setup/run instructions.
- [x] Data dictionary and focused feature tests.
- [ ] Run the complete test suite after pulling all changes.
- [ ] Build and launch the Docker image locally.
- [ ] Review README, assumptions, results and limitations.
- [ ] Commit the completed EDA notebooks and verified evaluation report.

## Deliberately out of scope

Kafka/Redpanda, real-time streaming, PostgreSQL, FastAPI, complex monitoring, separate scheduling infrastructure, extensive CI/CD/cloud deployment, additional deep-learning architectures and a large scenario simulator remain postponed.
