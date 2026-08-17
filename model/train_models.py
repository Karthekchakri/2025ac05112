"""
train_models.py
----------------
Trains 5 classification models (Logistic Regression, Decision Tree,
K-Nearest Neighbors, Gaussian Naive Bayes, Random Forest) on the
Telco Customer Churn dataset, evaluates each with 6 metrics, and
saves the fitted pipelines + a held-out test CSV for the Streamlit app.

Dataset: Telco Customer Churn (IBM sample dataset)
Source : https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Rows   : 7043   |  Features used: 19 (12+ requirement satisfied)
Task   : Binary classification -> predict `Churn` (Yes/No)
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. Load & clean data
# --------------------------------------------------------------------------
df = pd.read_csv("../telco.csv")

# TotalCharges has some blank strings for customers with tenure == 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

target_col = "Churn"
feature_cols = [c for c in df.columns if c != target_col]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print(f"Total features: {len(feature_cols)} "
      f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
print(f"Total instances: {len(df)}")

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the held-out TEST set only (per assignment instructions: Streamlit
# free tier has limited capacity, so only test data is shipped with the repo)
test_export = X_test.copy()
test_export[target_col] = y_test.values
test_export.to_csv("../test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_export)} rows")

# --------------------------------------------------------------------------
# 2. Preprocessing pipeline (shared by all models)
# --------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ]
)

# --------------------------------------------------------------------------
# 3. Define the 5 required models
# --------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, max_depth=10
    ),
}

results = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results[name] = metrics
    fitted_pipelines[name] = pipe

    # Save the fitted pipeline (preprocessing + model together)
    filename = "../model/" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    joblib.dump(pipe, filename)
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  -> saved to {filename}")

# --------------------------------------------------------------------------
# 4. Persist metrics + feature metadata for README / Streamlit app
# --------------------------------------------------------------------------
with open("../model/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

metadata = {
    "target_col": target_col,
    "feature_cols": feature_cols,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
}
with open("../model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nAll models trained. Metrics saved to model/metrics.json")

# Print markdown comparison table for README
print("\n\n| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
print("|---|---|---|---|---|---|---|")
for name, m in results.items():
    print(f"| {name} | {m['Accuracy']:.3f} | {m['AUC']:.3f} | {m['Precision']:.3f} | "
          f"{m['Recall']:.3f} | {m['F1']:.3f} | {m['MCC']:.3f} |")
