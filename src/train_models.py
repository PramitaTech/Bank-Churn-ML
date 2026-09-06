import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 📁 PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DATA_PATH = DATA_DIR / "bank_churn_500.csv"

MODEL_PATH = BASE_DIR / "decision_tree_model.pkl"


# ============================================================
# 🌍 20 COUNTRIES
# ============================================================

countries = [
    "India",
    "USA",
    "UK",
    "Canada",
    "Australia",
    "Germany",
    "France",
    "Spain",
    "Italy",
    "Japan",
    "China",
    "Singapore",
    "UAE",
    "Switzerland",
    "Netherlands",
    "Brazil",
    "Mexico",
    "South Africa",
    "New Zealand",
    "Sweden"
]


# ============================================================
# 📂 LOAD EXISTING DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset loaded successfully!")
print("Records:", len(df))


# ============================================================
# 🌍 UPDATE GEOGRAPHY TO 20 COUNTRIES
# ============================================================

np.random.seed(42)

df["Geography"] = np.random.choice(
    countries,
    size=len(df)
)


# ============================================================
# 💾 SAVE UPDATED DATASET
# ============================================================

df.to_csv(DATA_PATH, index=False)

print("\n🌍 Dataset updated with 20 countries!")
print("Countries available:")

for country in sorted(df["Geography"].unique()):
    print("-", country)


# ============================================================
# 🎯 FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "CustomerID",
        "Exited"
    ]
)

y = df["Exited"]


# ============================================================
# 🔤 CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Geography",
    "Gender"
]


# ============================================================
# 🔢 NUMERICAL FEATURES
# ============================================================

numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


# ============================================================
# ⚙️ PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ============================================================
# ✂️ TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 🤖 MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
}


# ============================================================
# 📊 MODEL TRAINING
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )


    results.append(
        [
            name,
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    )


    trained_models[name] = pipeline


# ============================================================
# 📊 MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)


print("\n")
print("=" * 70)
print("🤖 MODEL COMPARISON")
print("=" * 70)

print(
    results_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 🌳 SELECT DECISION TREE
# ============================================================

dt_pipeline = trained_models[
    "Decision Tree"
]


dt_pred = dt_pipeline.predict(
    X_test
)


# ============================================================
# 🔍 CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    dt_pred
)


print("\n")
print("=" * 70)
print("🔍 DECISION TREE CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# 📋 CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("📋 DECISION TREE CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        dt_pred,
        target_names=[
            "Stayed",
            "Churned"
        ],
        zero_division=0
    )
)


# ============================================================
# 🌟 FEATURE IMPORTANCE
# ============================================================

feature_names = (
    dt_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = (
    dt_pipeline
    .named_steps["model"]
    .feature_importances_
)


feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n")
print("=" * 70)
print("🌟 TOP FEATURES")
print("=" * 70)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 💾 SAVE DECISION TREE MODEL
# ============================================================

joblib.dump(
    dt_pipeline,
    MODEL_PATH
)


print("\n")
print("✅ Decision Tree model saved successfully!")

print(
    "📁 Model location:",
    MODEL_PATH
)

print("\n🎉 MODEL TRAINING COMPLETED!")