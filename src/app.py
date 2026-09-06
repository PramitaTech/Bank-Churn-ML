import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# 🏦 BANK CUSTOMER CHURN PREDICTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "decision_tree_model.pkl"

DATA_PATH = BASE_DIR / "data" / "bank_churn_500.csv"


# ============================================================
# 📂 LOAD MODEL AND DATA
# ============================================================

model = joblib.load(
    MODEL_PATH
)

data = pd.read_csv(
    DATA_PATH
)


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
# ⚙️ PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# 🎨 DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 🏦 HEADER
# ============================================================

st.markdown(
    '<div class="title">🏦 Bank Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '🤖 Machine Learning Based Customer Churn Analysis & Prediction 📊'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "👋 Welcome! Enter customer details below to predict "
    "the likelihood of customer churn."
)


# ============================================================
# 📊 OVERVIEW
# ============================================================

st.subheader(
    "📊 Customer Churn Overview"
)


total_customers = len(data)

churned_customers = int(
    data["Exited"].sum()
)

stayed_customers = (
    total_customers -
    churned_customers
)

churn_percentage = (
    churned_customers /
    total_customers
) * 100


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "👥 Total Customers",
        total_customers
    )


with col2:

    st.metric(
        "🚨 Churned Customers",
        churned_customers
    )


with col3:

    st.metric(
        "😊 Stayed Customers",
        stayed_customers
    )


with col4:

    st.metric(
        "📉 Churn Rate",
        f"{churn_percentage:.2f}%"
    )


st.divider()


# ============================================================
# 👤 CUSTOMER INFORMATION
# ============================================================

st.subheader(
    "👤 Customer Information"
)

st.write(
    "📝 Enter the customer's details to generate a prediction."
)


col1, col2 = st.columns(2)


# ============================================================
# LEFT SIDE
# ============================================================

with col1:

    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )


    geography = st.selectbox(
        "🌍 Country",
        countries
    )


    gender = st.selectbox(
        "👩‍💼 Gender",
        [
            "Male",
            "Female"
        ]
    )


    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=35
    )


    tenure = st.number_input(
        "📅 Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )


# ============================================================
# RIGHT SIDE
# ============================================================

with col2:

    balance = st.number_input(
        "💰 Account Balance",
        min_value=0.0,
        value=50000.0
    )


    num_products = st.number_input(
        "📦 Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )


    has_card = st.selectbox(
        "💳 Has Credit Card?",
        [
            "Yes",
            "No"
        ]
    )


    active_member = st.selectbox(
        "⚡ Is Active Member?",
        [
            "Yes",
            "No"
        ]
    )


    salary = st.number_input(
        "💵 Estimated Salary",
        min_value=0.0,
        value=50000.0
    )


st.divider()


# ============================================================
# 🔮 PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮✨ Predict Customer Churn",
    use_container_width=True
)


if predict_button:

    # ========================================================
    # CUSTOMER DATA
    # ========================================================

    customer_data = pd.DataFrame(
        {
            "CreditScore": [
                credit_score
            ],

            "Geography": [
                geography
            ],

            "Gender": [
                gender
            ],

            "Age": [
                age
            ],

            "Tenure": [
                tenure
            ],

            "Balance": [
                balance
            ],

            "NumOfProducts": [
                num_products
            ],

            "HasCrCard": [
                1 if has_card == "Yes"
                else 0
            ],

            "IsActiveMember": [
                1 if active_member == "Yes"
                else 0
            ],

            "EstimatedSalary": [
                salary
            ]
        }
    )


    # ========================================================
    # 🤖 PREDICTION
    # ========================================================

    prediction = model.predict(
        customer_data
    )[0]


    probability = model.predict_proba(
        customer_data
    )[0][1]


    probability_percentage = (
        probability * 100
    )


    # ========================================================
    # 🎯 RESULT
    # ========================================================

    st.subheader(
        "🎯 Prediction Result"
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️🚨 Customer is likely to CHURN."
            )

        else:

            st.success(
                "😊✅ Customer is likely to STAY."
            )


    with result_col2:

        st.metric(
            "📊 Churn Probability",
            f"{probability_percentage:.2f}%"
        )


    # ========================================================
    # 🚦 RISK LEVEL
    # ========================================================

    st.subheader(
        "🚦 Customer Risk Level"
    )


    if probability >= 0.60:

        risk_level = "🔴 High Risk"

        st.error(
            "🔴🚨 HIGH RISK — Immediate customer "
            "retention action is recommended."
        )


    elif probability >= 0.30:

        risk_level = "🟡 Medium Risk"

        st.warning(
            "🟡👀 MEDIUM RISK — Customer should be "
            "monitored and engagement improved."
        )


    else:

        risk_level = "🟢 Low Risk"

        st.success(
            "🟢😊 LOW RISK — Customer currently shows "
            "a lower likelihood of churn."
        )


    # ========================================================
    # 📈 PROGRESS BAR
    # ========================================================

    st.subheader(
        "📈 Churn Probability"
    )


    st.progress(
        min(
            int(probability_percentage),
            100
        )
    )


    st.write(
        f"**🚦 Risk Classification:** {risk_level}"
    )


    # ========================================================
    # 📋 CUSTOMER SUMMARY
    # ========================================================

    st.subheader(
        "📋 Customer Summary"
    )


    summary_col1, summary_col2, summary_col3 = st.columns(3)


    with summary_col1:

        st.write(
            f"🌍 **Country:** {geography}"
        )

        st.write(
            f"👩‍💼 **Gender:** {gender}"
        )

        st.write(
            f"🎂 **Age:** {age}"
        )


    with summary_col2:

        st.write(
            f"💳 **Credit Score:** {credit_score}"
        )

        st.write(
            f"💰 **Balance:** ₹{balance:,.2f}"
        )

        st.write(
            f"📅 **Tenure:** {tenure} years"
        )


    with summary_col3:

        st.write(
            f"📦 **Products:** {num_products}"
        )

        st.write(
            f"⚡ **Active Member:** {active_member}"
        )

        st.write(
            f"💳 **Credit Card:** {has_card}"
        )


    # ========================================================
    # 💡 RECOMMENDATION
    # ========================================================

    st.subheader(
        "💡 Recommended Action"
    )


    if probability >= 0.60:

        st.info(
            "💬 Consider proactive retention strategies such as "
            "personalized offers, loyalty benefits, customer "
            "support follow-up, and suitable financial products."
        )


    elif probability >= 0.30:

        st.info(
            "💬 Monitor this customer and improve engagement "
            "through personalized communication and product "
            "recommendations."
        )


    else:

        st.info(
            "💬 Continue normal customer engagement and "
            "maintain good service quality to retain the customer."
        )


# ============================================================
# 🤖 MODEL COMPARISON
# ============================================================

st.divider()

st.subheader(
    "🤖 Machine Learning Model Comparison"
)


st.write(
    "📊 These models were evaluated on the current "
    "20-country dataset."
)


# NOTE:
# The values below will be replaced automatically
# after retraining if you later save the results.
# For now, we calculate the comparison again here.


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
    roc_auc_score
)

from xgboost import XGBClassifier


X = data.drop(
    columns=[
        "CustomerID",
        "Exited"
    ]
)

y = data["Exited"]


categorical_features = [
    "Geography",
    "Gender"
]


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


comparison_preprocessor = ColumnTransformer(
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


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


comparison_models = {

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


comparison_results = []


for name, algorithm in comparison_models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                comparison_preprocessor
            ),

            (
                "model",
                algorithm
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


    comparison_results.append(
        [
            name,

            accuracy_score(
                y_test,
                y_pred
            ),

            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            roc_auc_score(
                y_test,
                y_prob
            )
        ]
    )


comparison_data = pd.DataFrame(
    comparison_results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)


display_data = comparison_data.copy()


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]:

    display_data[column] = (
        display_data[column] * 100
    ).round(2).astype(str) + "%"


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 📈 MODEL GRAPH
# ============================================================

st.subheader(
    "📈 Model Performance Comparison"
)


chart_data = comparison_data.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
]


st.bar_chart(
    chart_data,
    use_container_width=True
)


# ============================================================
# 🏆 MODEL SUMMARY
# ============================================================

best_recall_model = comparison_data.loc[
    comparison_data["Recall"].idxmax()
]


best_f1_model = comparison_data.loc[
    comparison_data["F1 Score"].idxmax()
]


best_accuracy_model = comparison_data.loc[
    comparison_data["Accuracy"].idxmax()
]


best_auc_model = comparison_data.loc[
    comparison_data["ROC-AUC"].idxmax()
]


st.subheader(
    "🏆 Model Evaluation Summary"
)


st.info(
    f"""
    🎯 **Best Accuracy:** {best_accuracy_model['Model']}
    ({best_accuracy_model['Accuracy'] * 100:.2f}%)

    🔎 **Best Recall:** {best_recall_model['Model']}
    ({best_recall_model['Recall'] * 100:.2f}%)

    ⭐ **Best F1 Score:** {best_f1_model['Model']}
    ({best_f1_model['F1 Score'] * 100:.2f}%)

    🚀 **Best ROC-AUC:** {best_auc_model['Model']}
    ({best_auc_model['ROC-AUC'] * 100:.2f}%)

    🌳 The Decision Tree model is used for the customer
    churn prediction feature of this application.
    """
)


# ============================================================
# 📊 DATASET INFORMATION
# ============================================================

st.divider()


with st.expander(
    "📊 Dataset Information"
):

    st.write(
        f"👥 **Total Records:** {len(data)}"
    )

    st.write(
        f"📋 **Total Columns:** {len(data.columns)}"
    )

    st.write(
        "🌍 **Number of Countries:** "
        f"{data['Geography'].nunique()}"
    )

    st.write(
        "🌍 **Countries:**"
    )

    st.write(
        ", ".join(
            sorted(
                data["Geography"].unique()
            )
        )
    )

    st.write(
        "🔢 **Dataset Features:**"
    )

    st.write(
        ", ".join(
            data.columns
        )
    )


# ============================================================
# 🔍 CONFUSION MATRIX
# ============================================================

with st.expander(
    "🔍 Decision Tree Confusion Matrix"
):

    dt_model = Pipeline(
        steps=[
            (
                "preprocessor",
                comparison_preprocessor
            ),

            (
                "model",
                DecisionTreeClassifier(
                    random_state=42
                )
            )
        ]
    )


    dt_model.fit(
        X_train,
        y_train
    )


    dt_pred = dt_model.predict(
        X_test
    )


    cm = pd.DataFrame(
        confusion_matrix(
            y_test,
            dt_pred
        ),

        index=[
            "Actual Stayed",
            "Actual Churned"
        ],

        columns=[
            "Predicted Stayed",
            "Predicted Churned"
        ]
    )


    st.dataframe(
        cm,
        use_container_width=True
    )


# ============================================================
# 🌟 FEATURE IMPORTANCE
# ============================================================

with st.expander(
    "🌟 Important Churn Factors"
):

    feature_names = (
        dt_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )


    feature_values = (
        dt_model
        .named_steps["model"]
        .feature_importances_
    )


    feature_data = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": feature_values
        }
    )


    feature_data = (
        feature_data
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )


    feature_data["Importance"] = (
        feature_data["Importance"] * 100
    ).round(2)


    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )


    st.write(
        "💡 These values show which features were most "
        "important for the Decision Tree model."
    )


# ============================================================
# ℹ️ ABOUT PROJECT
# ============================================================

with st.expander(
    "ℹ️ About This Project"
):

    st.write(
        """
        🎓 **Project Title:**

        A Comparative Analysis of Machine Learning Algorithms
        for Bank Customer Churn Prediction and Customer Behavior Analysis.


        🎯 **Objective:**

        To predict whether a bank customer is likely to leave
        the bank and identify customers who may require
        retention strategies.


        🤖 **Algorithms Used:**

        🔹 Logistic Regression

        🔹 Decision Tree

        🔹 Random Forest

        🔹 Support Vector Machine

        🔹 XGBoost


        🌍 **Geographical Categories:**

        The dataset contains 20 countries.


        📊 **Evaluation Metrics:**

        Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
        """
    )


# ============================================================
# ⚠️ DATASET NOTE
# ============================================================

with st.expander(
    "⚠️ Dataset Note"
):

    st.write(
        """
        This project currently uses a synthetic dataset
        created for academic and demonstration purposes.

        The dataset contains 500 customer records.

        The geographical feature contains 20 countries.

        The Decision Tree model was retrained after updating
        the geographical categories.
        """
    )


# ============================================================
# ❤️ FOOTER
# ============================================================

st.divider()


st.markdown(
    """
    <div style="text-align:center;">

    💙 Thank you for using the Bank Churn Prediction System! 😊

    <br>

    🎓 Machine Learning Academic Project |
    🤖 AI-Based Prediction |
    🌍 20 Countries

    </div>
    """,

    unsafe_allow_html=True
)