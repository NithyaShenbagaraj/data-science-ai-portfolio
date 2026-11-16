# Customer Churn Prediction

An end-to-end machine-learning project that estimates customer churn for a subscription-based telecommunications business and translates model outputs into retention insights.

**Status:** Initial implementation scaffold. Performance results will be published only after the dataset is downloaded and the evaluation pipeline has been run and verified.

## Business problem
Customer churn reduces recurring revenue and increases acquisition costs. The goal is to identify customers with a higher likelihood of leaving so a retention team can prioritize investigation and outreach. This is a decision-support demonstration, not a guarantee of customer behaviour.

## Dataset
Use the IBM Telco Customer Churn dataset distributed through Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Expected file: WA_Fn-UseC_-Telco-Customer-Churn.csv

Place it at data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv. The raw dataset is not committed to this repository. Review the dataset page's terms and attribution requirements before reuse.

## Workflow
1. Validate source data and convert TotalCharges to numeric.
2. Explore churn distribution and customer/service characteristics.
3. Engineer tenure bands and a service-count feature.
4. Use stratified train/test splitting and leakage-safe preprocessing.
5. Compare Logistic Regression and Random Forest using stratified cross-validation.
6. Evaluate the selected pipeline on a held-out test set.
7. Explain model behaviour using permutation importance and SHAP.
8. Serve batch churn probabilities through Streamlit.

## Structure
- app.py — Streamlit batch-scoring application
- src/data.py — dataset loading and validation
- src/features.py — feature engineering and target separation
- src/train.py — model comparison, training and evaluation
- src/explain.py — permutation importance and SHAP
- tests/test_features.py — feature engineering tests
- data/raw/ — local source dataset (not committed)
- models/ — trained pipeline (not committed)
- reports/ — generated metrics and plots

## Setup
Python 3.10 or newer is recommended.

    cd 01-machine-learning/customer-churn-prediction
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Download the CSV from the dataset page and place it in data/raw/.

## Train and evaluate

    python -m src.train

The script saves the fitted pipeline under models/ and evaluation outputs under reports/.

## Explain the model

    python -m src.explain

This generates permutation-importance results and a SHAP summary plot. Explanations describe patterns learned by the model; they do not establish causation.

## Run the application

    streamlit run app.py

Upload a CSV containing the same customer input columns as the source dataset. The app returns churn probabilities and a configurable prioritization view. Do not upload real customer personal data to a public demo.

## Evaluation
Accuracy alone is insufficient when churn classes are imbalanced. The evaluation reports precision, recall, F1-score, ROC-AUC and a confusion matrix. The default classification threshold is 0.50; a production threshold should be selected using intervention capacity and the business costs of false positives and false negatives.

## Reproducibility and limitations
- The split is stratified and uses a fixed random seed.
- Imputation, scaling and one-hot encoding are fitted inside an sklearn Pipeline.
- The held-out test set is not used for model selection.
- No performance claims are published until the training run has completed.
- The public sample dataset may not represent a current telecom customer base; performance and fairness should be reassessed before operational use.

## Milestones
- [ ] Complete EDA and document data-quality findings.
- [ ] Run baseline model comparison and record test metrics.
- [ ] Review threshold trade-offs and calibration.
- [ ] Generate SHAP and permutation-importance reports.
- [ ] Add validated screenshots and deployment instructions.
