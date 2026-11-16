# Healthcare Risk Prediction

An end-to-end machine learning project that studies binary heart-disease classification using the UCI Heart Disease dataset. The project demonstrates reproducible data acquisition, exploratory analysis, preprocessing, model comparison, evaluation, explainability, and an interactive Streamlit interface.

> **Research and education only.** This model is not a medical device, is not clinically validated, and must not be used to diagnose, screen, or make treatment decisions for any person.

## Problem statement

Can routinely recorded clinical attributes help a machine-learning model distinguish records with an observed presence of heart disease from records without it in the UCI Cleveland data?

The target is derived from the source field `num`: values greater than zero are mapped to 1 (presence); zero is mapped to 0 (absence). This is a dataset label, not a patient's future risk or a clinical diagnosis.

## Dataset

- Source: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Dataset ID: 45
- The UCI collection contains records from multiple institutions. This implementation uses the dataset returned by the UCI repository client; inspect the dataset metadata and sample count after download.
- The original data contains missing values, represented in some fields by `?`. These are converted to missing values and imputed inside the training pipeline.
- The raw dataset is downloaded locally and is not committed to this repository.

## Technical workflow

1. Download and cache the source data.
2. Validate columns, target values, and missingness.
3. Split data using a stratified holdout set.
4. Impute numeric and categorical features separately.
5. Scale numeric features and one-hot encode categorical features.
6. Compare Logistic Regression, Random Forest, and XGBoost using stratified cross-validation on the training partition.
7. Select the model with the highest mean cross-validation ROC-AUC and evaluate it once on the holdout set.
8. Save the fitted preprocessing/model pipeline and evaluation artifacts.
9. Generate model explanations with SHAP.
10. Run a Streamlit interface for a single-record demonstration.

## Repository layout

```text
healthcare-risk-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/                 # downloaded data; ignored by Git
├── models/                  # trained pipeline; ignored by Git
├── reports/                 # generated metrics and plots
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── eda.py
│   ├── explain.py
│   ├── train.py
│   └── tune.py
└── tests/
    └── test_data.py
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
cd 01-machine-learning/healthcare-risk-prediction
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

From this project directory:

```bash
python -m src.train
```

The first run downloads the dataset. The training script writes:

- `models/healthcare_risk_pipeline.joblib`
- `reports/metrics.json`
- `reports/model_comparison.csv`
- `reports/test_predictions.csv`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`

The split uses a fixed random seed for reproducibility. Cross-validation is performed only on the training partition; the holdout partition is reserved for final evaluation.

## Explore the dataset

Run the descriptive analysis before training:

```bash
python -m src.eda
```

It writes a label and missingness overview, a cohort summary, numeric feature statistics, and categorical value counts to `reports/`. These summaries describe the downloaded sample; they are not clinical conclusions.

## Explain the fitted model

After training:

```bash
python -m src.explain
```

This generates a SHAP summary plot in `reports/shap_summary.png`. SHAP values explain model output associations; they do not establish causality.

## Launch the demo

```bash
streamlit run app.py
```

The app requires a trained model. Run training first.

## Tests

```bash
pytest -q
```

## Evaluation strategy

The project reports:

- ROC-AUC
- Average precision / PR-AUC
- Accuracy
- Precision
- Recall (sensitivity)
- F1-score
- Specificity
- Confusion matrix

Accuracy alone is not sufficient for imbalanced or high-impact classification. Review sensitivity, specificity, precision-recall trade-offs, calibration, and uncertainty before drawing conclusions. The small, historical dataset and single holdout split limit generalisability.

## Hyperparameter tuning

A separate experiment optimized Logistic Regression and Random Forest using hyperparameter search.

### Search methodology

**Logistic Regression — GridSearchCV**
- C values: [0.01, 0.1, 1, 10, 100]
- Class weights: None, balanced
- Scoring: ROC-AUC
- Cross-validation: Stratified 5-fold

**Random Forest — RandomizedSearchCV**
- Estimators: [200, 400, 600]
- Maximum depth: [None, 3, 5, 8]
- Minimum samples split: [2, 5, 10]
- Minimum samples leaf: [1, 2, 4]
- Maximum features: [sqrt, 0.8]
- Class weights: None, balanced, balanced_subsample
- Search iterations: 30
- Scoring: ROC-AUC
- Cross-validation: Stratified 5-fold

### Best configurations

| Model | Best CV ROC-AUC | Best parameters |
|---|---:|---|
| Tuned Logistic Regression | 0.90314 | C = 0.1, class_weight = balanced |
| Tuned Random Forest | 0.90162 | 400 estimators, depth = 5, min_samples_split = 10, min_samples_leaf = 2, max_features = sqrt, class_weight = balanced |

Tuned Logistic Regression was selected based on its higher mean cross-validation ROC-AUC.

### Baseline vs tuned performance

| Metric | Baseline | Tuned |
|---|---:|---:|
| ROC-AUC | 96.65% | 96.65% |
| Average Precision | 96.34% | 96.39% |
| Accuracy | 86.89% | 88.52% |
| Precision | 81.25% | 83.87% |
| Recall / Sensitivity | 92.86% | 92.86% |
| Specificity | 81.82% | 84.85% |
| F1-score | 86.67% | 88.14% |

### Findings and limitations

- Tuning modestly improved accuracy, precision, specificity, and F1-score.
- Recall remained unchanged.
- ROC-AUC remained approximately the same.
- The tuned model correctly classified one additional negative-class observation on the 61-row holdout set.

These results are exploratory. The dataset is small, and holdout results have been inspected during model development. Independent external validation is necessary before drawing conclusions about generalisation.

The tuning implementation is available in `src/tune.py`. Generated reports and model artifacts are excluded from version control.

## Responsible-use limitations

- The dataset is small, historical, and not representative of all current populations or care settings.
- The project does not establish prospective performance, clinical utility, fairness, or external validity.
- A model score is not an individual diagnosis or a substitute for a qualified clinician.
- Do not enter real patient information into the demo.
- Any real-world clinical application would require appropriate governance, privacy controls, external validation, prospective evaluation, and regulatory review.

## Future improvements

- Add nested cross-validation and confidence intervals.
- Evaluate calibration and decision thresholds without tuning on the final test set.
- Add subgroup performance analysis where sample sizes support responsible interpretation.
- Package the service with FastAPI and Docker after the modelling workflow is stable.

## Author

Nithya Shenbagaraj — Data Analyst transitioning into Applied Machine Learning and AI Engineering.

[Portfolio repository](https://github.com/NithyaShenbagaraj/data-science-ai-portfolio)
