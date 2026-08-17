"""
Streamlit app - Telco Customer Churn Classification Demo
BITS Pilani WILP | M.Tech (AIML/DSE) | Machine Learning | Assignment 2

Features:
  a. CSV upload of test data
  b. Model selection dropdown (5 trained models)
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + full classification report
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Telco Churn Classifier", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.joblib",
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_metadata():
    with open("model/metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_training_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)


st.title("Telco Customer Churn - Classification Demo")
st.caption(
    "BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning · Assignment 2 "
    "— 5 classifiers trained on the IBM Telco Customer Churn dataset "
    "(19 features, 7,043 rows, binary target: Churn Yes/No)."
)

metadata = load_metadata()
target_col = metadata["target_col"]
feature_cols = metadata["feature_cols"]

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Upload the provided test_data.csv, or any CSV with the same "
         "columns (features + Churn label).",
)

use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead", value=uploaded_file is None)

st.sidebar.markdown("---")
st.sidebar.subheader("All-model comparison (training run)")
train_metrics = load_training_metrics()
compare_df = pd.DataFrame(train_metrics).T.round(3)
st.sidebar.dataframe(compare_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if uploaded_file is not None and not use_sample:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    data = pd.read_csv("test_data.csv")
else:
    st.info("Upload a CSV file from the sidebar, or check 'Use bundled test_data.csv'.")
    st.stop()

st.subheader("Preview of loaded data")
st.dataframe(data.head(10), use_container_width=True)
st.caption(f"{data.shape[0]} rows x {data.shape[1]} columns")

has_labels = target_col in data.columns

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
pipe = load_model(MODEL_FILES[model_name])

X = data[feature_cols] if set(feature_cols).issubset(data.columns) else data.drop(
    columns=[c for c in [target_col] if c in data.columns]
)

try:
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else preds
except Exception as e:
    st.error(f"Prediction failed — check that the uploaded CSV has the expected columns. Details: {e}")
    st.stop()

result_df = data.copy()
result_df["Predicted_Churn"] = ["Yes" if p == 1 else "No" for p in preds]
result_df["Churn_Probability"] = proba.round(3)

st.subheader(f"Predictions — {model_name}")
st.dataframe(result_df.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# Metrics (only computable if ground-truth labels are present)
# ---------------------------------------------------------------------------
st.subheader("Evaluation Metrics")

if has_labels:
    y_true = data[target_col]
    if y_true.dtype == object:
        y_true = y_true.map({"Yes": 1, "No": 0}).fillna(y_true)

    metrics = {
        "Accuracy": accuracy_score(y_true, preds),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1 Score": f1_score(y_true, preds),
        "MCC": matthews_corrcoef(y_true, preds),
    }

    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_b:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, preds, target_names=["No Churn", "Churn"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
else:
    st.warning(
        "Uploaded CSV has no 'Churn' ground-truth column, so evaluation metrics "
        "and the confusion matrix can't be computed — showing predictions only."
    )

st.markdown("---")
st.caption(
    "Dataset source: IBM Telco Customer Churn sample dataset. "
    "Models: Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, "
    "Random Forest — trained on an 80/20 stratified split, evaluated on the held-out test set."
)
