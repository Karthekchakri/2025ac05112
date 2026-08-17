# Telco Customer Churn — Classification Assignment

BITS Pilani WILP | M.Tech (AIML/DSE) | Machine Learning | Assignment 2

## a. Problem Statement

Customer churn (a customer discontinuing a subscription service) is one of
the most expensive problems for subscription-based businesses — acquiring a
new customer typically costs far more than retaining an existing one. This
project builds and compares five classification models that predict whether
a telecom customer will **churn (leave the service)** based on their
demographic profile, account information, and the services they have
subscribed to. The trained models are served through an interactive
Streamlit web application so that predictions and evaluation metrics can be
explored on demand.

## b. Dataset Description

- **Name:** Telco Customer Churn (IBM sample dataset)
- **Source:** downloaded programmatically at runtime (see `model/train_models.ipynb`, Section 1) from a direct open URL serving the same file published on [Kaggle as `blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The notebook also includes commented-out Kaggle API (`kagglehub` / Kaggle CLI) download cells as drop-in alternatives if you'd rather authenticate with your own Kaggle account. A local copy is saved to `telco.csv` when the notebook runs.
- **Instances:** 7,043 customer records (assignment minimum: 500 ✅)
- **Features used:** 19 (assignment minimum: 12 ✅) — 4 numeric
  (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) and 15
  categorical (gender, contract type, internet service, add-on services,
  payment method, etc.)
- **Target variable:** `Churn` — binary (`Yes` / `No`), encoded as 1 / 0
- **Class balance:** ~26.5% churn, ~73.5% no-churn (moderately imbalanced,
  which is why AUC and MCC are reported alongside accuracy)
- **Preprocessing:** dropped the `customerID` identifier column; imputed the
  handful of blank `TotalCharges` values with the column median; numeric
  features were standard-scaled and categorical features one-hot encoded
  inside a scikit-learn `ColumnTransformer` so the exact same pipeline is
  reused for training, evaluation, and the Streamlit app.
- **Train/test split:** 80% train / 20% held-out test, stratified on
  `Churn`. Only the **test split** (`test_data.csv`, 1,409 rows) is shipped
  in this repository, per the assignment's Streamlit free-tier guidance.

## c. GitHub Repository Link

> **https://github.com/Karthekchakri/2025ac05112**

## d. Models Used

All 5 models were trained on the identical preprocessed dataset and
evaluated on the same held-out 20% test split (1,409 records).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.806 | 0.842 | 0.657 | 0.559 | 0.604 | 0.479 |
| Decision Tree | 0.775 | 0.798 | 0.588 | 0.508 | 0.545 | 0.399 |
| kNN | 0.778 | 0.821 | 0.584 | 0.564 | 0.574 | 0.424 |
| Naive Bayes | 0.695 | 0.807 | 0.459 | 0.837 | 0.593 | 0.424 |
| Random Forest (Ensemble) | 0.804 | 0.841 | 0.667 | 0.524 | 0.587 | 0.467 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall balance of accuracy (0.806) and AUC (0.842). As a linear model it handles the mostly-categorical, one-hot-encoded feature space well and stays stable without overfitting. Its main weakness is recall (0.559) — it misses a meaningful share of customers who actually churn. |
| Decision Tree | Weakest model on every metric (AUC 0.798, MCC 0.399). A single tree, even depth-limited, overfits to specific split rules in this noisy, mostly-categorical dataset and generalises worse than the ensemble or linear alternatives. |
| kNN | Middle-of-the-pack performance (AUC 0.821). Distance-based similarity works reasonably once numeric features are scaled, but performance is sensitive to the mix of one-hot categorical dimensions, which dilutes the "closeness" signal that kNN relies on. |
| Naive Bayes | Lowest precision (0.459) and lowest accuracy (0.695), but by far the **highest recall (0.837)** — it over-predicts churn. This is a direct consequence of the (violated) feature-independence assumption plus its bias toward the minority class; it would be the model of choice only if catching almost every potential churner matters more than avoiding false alarms. |
| Random Forest (Ensemble) | Highest precision (0.667) and close to Logistic Regression on accuracy/AUC (0.804 / 0.841), while being more robust to noise than the single Decision Tree thanks to bagging across 300 trees. Recall (0.524) is still moderate — like Logistic Regression, it is conservative about flagging churn. |
| **Overall Winner for this dataset** | **Logistic Regression** — it delivers the best AUC (0.842) and F1 (0.604) with a simple, fast, well-calibrated model, and its performance is matched but not beaten on any metric by the heavier Random Forest ensemble. For a business that specifically wants to **catch more churners** even at the cost of false positives, Naive Bayes (recall 0.837) would be the practical alternative to pair with a retention campaign. |

## Repository Structure

```
project-folder/
│-- app.py                    # Streamlit web application
│-- requirements.txt          # Python dependencies
│-- README.md                 # This file
│-- test_data.csv             # Held-out test split (1,409 rows) used by the app
│-- telco.csv                 # Full source dataset (downloaded by the notebook/script)
│-- model/
│   │-- train_models.ipynb    # End-to-end training + evaluation (Jupyter/Colab notebook)
│   │-- train_models.py       # Same pipeline as a plain Python script
│   │-- metrics.json          # Saved evaluation metrics for all 5 models
│   │-- metadata.json         # Feature/column metadata used by the app
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- random_forest_ensemble.joblib
```

## Streamlit App Features

- **Dataset upload (CSV):** upload your own test CSV (same schema as
  `test_data.csv`), or use the bundled sample with one click.
- **Model selection dropdown:** switch between all 5 trained models live.
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1, MCC
  computed on whatever data is loaded (when ground-truth `Churn` labels are
  present).
- **Confusion matrix & classification report:** rendered per selected model
  and dataset.

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # optional — pre-trained .joblib files are already included
streamlit run app.py
```

## Live Streamlit App Link

> **`<PASTE-YOUR-STREAMLIT-COMMUNITY-CLOUD-URL-HERE>`**
>
> Deploy at https://streamlit.io/cloud → "New app" → select
> `Karthekchakri/2025ac05112` → branch `main` → file `app.py` → Deploy.
