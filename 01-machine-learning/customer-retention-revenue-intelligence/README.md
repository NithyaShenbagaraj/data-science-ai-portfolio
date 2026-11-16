# Customer Retention & Revenue Intelligence

An end-to-end customer analytics and machine-learning project using UCI Online Retail II. The goal is to understand revenue and customer behaviour, segment customers, estimate repeat-purchase propensity, and explain model predictions.

**Prediction definition:** Using a customer's behaviour in the 180 days before a monthly snapshot date, estimate whether they will place at least one valid purchase in the following 90 days. This is a repeat-purchase propensity target, not a directly observed contractual churn event.

## Business questions
1. How are sales, customers, products and cancellations changing over time?
2. Which customers contribute the most revenue, and how do RFM profiles differ?
3. Which currently active customers are less likely to purchase again in the next 90 days?
4. Which behavioural signals explain predictions, and how could a retention team prioritize outreach?

The UCI source contains 1,067,371 transaction rows from a UK-based online retailer between 1 December 2009 and 9 December 2011. UCI notes missing values and a wholesaler-heavy customer mix. Official source: https://archive.ics.uci.edu/dataset/502/online+retail%2Bii

## Technology
- Python: Pandas, NumPy
- SQL warehouse: DuckDB
- Machine learning: Scikit-learn, Logistic Regression, Random Forest
- Explainability: SHAP
- Application: Streamlit
- Quality: Pytest, deterministic pipeline and explicit data contracts

## Architecture
UCI workbook -> Python ingestion and validation -> cleaned Parquet + DuckDB -> SQL revenue / returns / RFM / cohort analytics -> leakage-aware customer snapshots -> temporal model selection -> SHAP -> Streamlit scoring.

## Repository structure
- app.py: Streamlit retention scoring app
- data/raw: source workbook, excluded from Git
- data/processed: Parquet and DuckDB, excluded from Git
- sql/analytics.sql: business analysis queries
- src/data.py: ingestion, cleaning and DuckDB load
- src/features.py: 180-day features and 90-day label
- src/train.py: temporal model selection and evaluation
- src/explain.py: SHAP global explanations
- src/run_sql.py: execute business SQL
- tests/test_features.py: feature and label boundary tests
- models and reports: generated artifacts, excluded from Git

## Guided Jupyter notebooks

The notebooks provide the step-by-step analytical narrative. The `src/` modules remain the reusable implementation used by tests and the Streamlit app; `app.py`, SQL scripts and tests remain Python/SQL files.

| Notebook | Purpose |
|---|---|
| `notebooks/01_data_understanding.ipynb` | Workbook structure, schema and data-quality profiling |
| `notebooks/02_cleaning_and_sql_analytics.ipynb` | Cleaning pipeline and DuckDB business queries |
| `notebooks/03_customer_rfm_and_cohorts.ipynb` | RFM, cohort retention and revenue concentration |
| `notebooks/04_feature_engineering.ipynb` | Past-only features and forward repeat-purchase label |
| `notebooks/05_model_training_evaluation.ipynb` | Temporal model selection and held-out evaluation |
| `notebooks/06_model_explainability.ipynb` | SHAP global explanations |

Install the optional notebook environment after the core project dependencies:

~~~bash
pip install -r requirements-notebooks.txt
jupyter lab
~~~

Open the notebooks from the project folder and run them in numerical order. Notebook 01 requires the UCI workbook; later notebooks depend on the outputs from earlier stages. The notebooks are committed without execution outputs and have not been run against the local workbook yet.

## 1. Obtain the data
Download online_retail_II.xlsx (approximately 43.5 MB) from the official UCI page and place it at data/raw/online_retail_II.xlsx. The workbook contains two annual sheets; ingestion combines both. Do not commit the raw workbook, processed database or customer-level exports.

Attribution: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Licensed CC BY 4.0.

## 2. Setup
Python 3.10+ is recommended.

~~~bash
cd 01-machine-learning/customer-retention-revenue-intelligence
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

## 3. Ingest and clean
~~~bash
python -m src.data
~~~
Both sheets are combined, column names standardized, dates and numeric fields parsed, duplicate rows removed, cancellation rows retained for returns analysis, and valid positive-quantity / positive-price purchases flagged. Missing CustomerID rows remain available for transaction-level analysis but are excluded from customer-level modelling.

## 4. SQL analytics
~~~bash
python -m src.run_sql
~~~
Exports monthly revenue, country performance, cancellations, RFM segments, cohort retention and customer revenue concentration to reports/sql/.

## 5. Build modelling snapshots
~~~bash
python -m src.features
~~~
Features use only valid purchases in the 180 days before each monthly cutoff. The label is 1 when at least one valid invoice occurs in the following 90 days. Snapshots without a complete 90-day outcome window are excluded.

## 6. Train and evaluate
~~~bash
python -m src.train
~~~
Logistic Regression is the baseline and Random Forest the non-linear candidate. Chronological train, validation and test periods are separated by a 90-day embargo to reduce overlap between forward labels and later feature windows. Validation selects the model; the final test period remains held out. Metrics include ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix and precision/lift in the highest-scored 10% and 20%.

## 7. Explain and run
~~~bash
python -m src.explain
streamlit run app.py
pytest -q
~~~
SHAP describes model contributions, not causality. The app accepts engineered customer feature CSVs and returns ranked repeat-purchase probabilities. Never upload confidential production customer data to a public demo.

## Important limitations
- Invoice numbers beginning with C represent cancellations. They are retained for returns analysis but excluded from purchase features and labels.
- Non-positive quantity or unit price rows are not valid purchases.
- Customers without an ID cannot be used for customer-level modelling.
- Historical results from a wholesaler-heavy retailer should not be generalized to a modern retailer without external validation.
- This is not a direct churn label and does not estimate the causal effect of a retention campaign.
- No performance claims are made until the pipeline has been run and outputs verified locally.

## Milestones
- [ ] Download and profile the official workbook.
- [ ] Reconcile row counts and data-quality exclusions.
- [ ] Review SQL outputs and write a business insight memo.
- [ ] Inspect class balance by snapshot period.
- [ ] Train and document held-out test results.
- [ ] Review SHAP and top-decile lift.
- [ ] Capture app screenshots and deploy with non-sensitive artifacts only.
