# Project 3 Roadmap — Focused Portfolio Scope

**Project:** Agency Revenue Intelligence & Forecasting Platform  
**Branch:** `feature/agency-revenue-intelligence`

## Objective

Forecast two distinct business outcomes:

1. Monthly agency recognized revenue.
2. Daily client campaign-attributed revenue.

Campaign-attributed sales are not agency revenue and must not be combined with it.

## Phase 1 — Data engineering and validation

- [x] Synthetic data generator and seven Parquet dataset outputs.
- [x] Candidate dataset generated locally.
- [x] Structural and business-rule validation suite: 26 passing tests.
- [x] Channel-level campaign funnel assumptions and ROAS diagnostics.
- [ ] Verify that the latest locally refined generator and configuration are synchronized with GitHub.
- [ ] Revalidate the exact synchronized generator output.
- [ ] Approve and promote candidate data only after validation; preserve original processed data.
- [ ] Document final generator assumptions and limitations.

**Data safeguard:** Candidate data is kept separate from the original processed dataset. Do not overwrite the original data during validation or promotion.

## Phase 2 — EDA and feature engineering

- [x] Agency monthly EDA notebook completed, executed, reviewed and committed.
- [x] Reusable lag, rolling and calendar feature functions implemented.
- [x] Complete campaign EDA: funnel, channel spend, attributed revenue, conversion rates and ROAS.
- [x] Review feature availability and leakage safeguards for both forecasting targets.

**Feature engineering:** Lag and rolling features use prior observations. Campaign forecasting uses chronological partitions based on calendar dates.

## Phase 3 — Forecasting models

- [x] Seasonal Naive and 7-day Naive baselines.
- [x] SARIMA monthly forecasting implementation.
- [x] XGBoost recursive forecasting for both targets.
- [x] PyTorch LSTM recursive forecasting for both targets.
- [x] Chronological train, validation and held-out test pipeline.
- [x] Full forecasting pipeline executed successfully on candidate datasets.
- [x] Validation and held-out test metrics generated and reviewed.
- [ ] Add SHAP explanations for XGBoost where appropriate.
- [ ] Investigate daily campaign forecasting error and improve generalization.

**Selected models based on validation WAPE:**

| Target | Selected model | Validation WAPE | Test WAPE |
|---|---|---:|---:|
| Monthly agency revenue | SARIMA | 5.94% | 16.45% |
| Daily campaign revenue | 7-day Naive | 42.61% | 54.24% |

**Evaluation metrics:** MAE, RMSE, WAPE and signed forecast bias.

**Interpretation:** Daily campaign forecasting has relatively high held-out test error. XGBoost and LSTM did not outperform the selected baselines in these experiments. Treat all results as exploratory and specific to synthetic data.

## Phase 4 — Dashboard and explainability

- [x] Streamlit dashboard for agency revenue, client activity, campaign performance and channel ROAS.
- [x] Illustrative fixed-ROAS scenario calculator.
- [x] Validation model comparison and held-out test metrics view.
- [x] Actual-versus-predicted visualizations.
- [x] Dashboard launched locally and its main sections reviewed.
- [ ] Add and validate SHAP feature explanations for XGBoost.
- [ ] Capture representative dashboard screenshots.

**Scenario limitation:** Scenario outputs use historical blended ROAS as a fixed arithmetic assumption. They are not causal estimates of incremental advertising revenue.

## Phase 5 — Finalization

- [x] Dockerfile and setup/run instructions added.
- [x] Data dictionary created.
- [x] Focused data validation, feature and forecasting tests implemented.
- [x] Complete local test suite executed successfully: 26 passed.
- [x] Forecast evaluation CSV artifacts generated.
- [x] README updated with setup instructions, methodology, evaluation results and limitations.
- [ ] Build and launch the Docker image locally.
- [ ] Complete final documentation and reproducibility review.
- [x] Commit the completed EDA notebooks.

## Deliberately out of scope

The following remain postponed to keep the portfolio project focused:

- Kafka/Redpanda and real-time streaming.
- PostgreSQL and separate database infrastructure.
- FastAPI.
- Complex monitoring and alerting.
- Separate scheduling infrastructure.
- Extensive CI/CD and cloud deployment.
- Additional deep-learning architectures.
- A large scenario simulation engine.