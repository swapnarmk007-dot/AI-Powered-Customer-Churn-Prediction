# ============================================================
# CUSTOMER CHURN AI - ADVANCED ENTERPRISE SUITE
# Machine Learning + Streamlit + RAG + Gemini AI + Explainability
# Developed by Swapna V
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Gemini AI
from google import genai


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn AI Suite",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# 2. CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>
.main { background-color: #f8f9fc; }
h1, h2, h3 { color: #243b55; }
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data.columns = data.columns.str.strip()

    if "Churn" not in data.columns:
        raise ValueError("Churn column not found in dataset.")

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )
    data["Churn"] = data["Churn"].astype(str).str.strip()
    data = data[data["Churn"].isin(["Yes", "No"])].copy()
    return data


st.sidebar.header("📂 Dataset & Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Churn CSV",
    type=["csv"]
)

default_path = r"C:\Users\swapn\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    elif os.path.exists(default_path):
        df = load_data(default_path)
    elif os.path.exists("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    else:
        st.warning("Please upload your Telco Customer Churn CSV file in the sidebar.")
        st.stop()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


# ------------------------------------------------------------
# 4. PREPROCESSING & MODEL TRAINING
# ------------------------------------------------------------

df = df.drop_duplicates().copy()
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


@st.cache_resource
def train_model(X, y):
    num_feat = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feat),
            ("cat", categorical_transformer, cat_feat)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0.0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return model, X_test, y_test, y_pred, y_prob, acc, prec, rec, f1, roc_auc, cm


with st.spinner("Training Random Forest Machine Learning Model..."):
    (
        model, X_test, y_test, y_pred, y_prob,
        accuracy, precision, recall, f1, roc_auc, cm
    ) = train_model(X, y)


# ------------------------------------------------------------
# 5. LOAD RAG DOCUMENTS
# ------------------------------------------------------------

DOCUMENTS_PATH = Path("rag/documents")

@st.cache_data
def load_rag_documents():
    documents = []
    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    
    sample_file = DOCUMENTS_PATH / "retention_policy.txt"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(
                "Company Retention Policy: Offer 15% discount on annual contracts "
                "for high-risk churn customers. Provide dedicated technical support "
                "onboarding for fiber optic users. Customers with tenure under 12 months "
                "receive a complimentary account review and loyalty bonus."
            )

    for file_path in DOCUMENTS_PATH.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append({
                "filename": file_path.name,
                "content": file.read()
            })
    return documents

rag_documents = load_rag_documents()


def retrieve_documents(query, documents, top_k=3):
    if not documents:
        return []
    query_words = set(re.findall(r"\w+", query.lower()))
    results = []
    for document in documents:
        content_words = set(re.findall(r"\w+", document["content"].lower()))
        score = len(query_words.intersection(content_words))
        results.append((score, document))
    results.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in results[:top_k]]


def generate_gemini_response(prompt_text):
    api_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API key not configured. Set `GEMINI_API_KEY` in Streamlit secrets."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"


# ------------------------------------------------------------
# 6. SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("📊 CUSTOMER CHURN AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📈 Customer Insights",
        "🤖 Model Performance",
        "🔍 Customer Prediction",
        "💡 AI Retention Recommendations",
        "🧠 RAG Knowledge Assistant",
        "📂 Batch Prediction",
        "🔬 Model Explainability",
        "ℹ️ About Project"
    ]
)


# ============================================================
# TAB 1: DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.header("📊 Executive Churn Dashboard")
    
    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_rate = churned / total_customers * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Retained Customers", f"{retained:,}")
    col4.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Churn Distribution")
        counts = df["Churn"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["#4e79a7", "#e15759"])
        st.pyplot(fig)
    with right:
        st.subheader("Contract Type vs Churn")
        if "Contract" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2", ax=ax)
            st.pyplot(fig)


# ============================================================
# TAB 2: CUSTOMER INSIGHTS
# ============================================================
elif page == "📈 Customer Insights":
    st.header("📈 Deep-Dive Customer Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Tenure vs Monthly Charges")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="Churn", alpha=0.7, ax=ax)
        st.pyplot(fig)


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.header("🤖 Machine Learning Model Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Precision", f"{precision * 100:.2f}%")
    col3.metric("Recall", f"{recall * 100:.2f}%")
    col4.metric("F1-Score", f"{f1 * 100:.2f}%")
    col5.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax)
        st.pyplot(fig)
    with col_b:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)


# ============================================================
# TAB 4: CUSTOMER PREDICTION
# ============================================================
elif page == "🔍 Customer Prediction":
    st.header("🔍 Individual Customer Churn Prediction")
    
    with st.form("single_prediction_form"):
        cust_input = {}
        c1, c2 = st.columns(2)
        for i, feat in enumerate(X.columns):
            with (c1 if i % 2 == 0 else c2):
                if feat in categorical_features:
                    opts = sorted(X[feat].dropna().astype(str).unique().tolist())
                    cust_input[feat] = st.selectbox(feat, opts if opts else ["Unknown"])
                else:
                    mn, mx, md = float(X[feat].min()), float(X[feat].max()), float(X[feat].median())
                    cust_input[feat] = st.number_input(feat, min_value=mn, max_value=mx, value=md)
        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([cust_input], columns=X.columns)
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if pred == 1:
                st.error("⚠️ High Churn Risk (Customer Likely to Leave)")
            else:
                st.success("✅ Low Churn Risk (Customer Likely to Stay)")
        with res_col2:
            st.metric("Churn Probability", f"{prob * 100:.2f}%")
            st.progress(float(prob))


# ============================================================
# TAB 5: AI RETENTION RECOMMENDATIONS
# ============================================================
elif page == "💡 AI Retention Recommendations":
    st.header("💡 AI-Powered Retention Strategies")
    st.write("Generate customized generative AI strategies using internal company RAG policies.")
    
    client_tenure = st.slider("Customer Tenure (Months)", 0, 72, 12)
    client_charges = st.number_input("Monthly Charges ($)", 10.0, 200.0, 75.0)
    client_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    if st.button("Generate AI Retention Plan", use_container_width=True):
        retrieved = retrieve_documents(f"{client_contract} tenure {client_tenure}", rag_documents)
        context = "\n".join([d["content"] for d in retrieved])
        
        prompt = f"""
        Customer Tenure: {client_tenure} months
        Monthly Charges: ${client_charges}
        Contract: {client_contract}
        
        Company Policy Knowledge:
        {context}
        
        Task: Provide 4 actionable retention recommendations for this customer.
        """
        with st.spinner("Generating AI strategy..."):
            ai_text = generate_gemini_response(prompt)
        st.markdown(ai_text)


# ============================================================
# TAB 6: RAG KNOWLEDGE ASSISTANT
# ============================================================
elif page == "🧠 RAG Knowledge Assistant":
    st.header("🧠 RAG Knowledge Assistant")
    st.write("Query your uploaded internal company documents and retention guidelines.")
    
    user_query = st.text_input("Ask a question about retention policies:", "What discounts can we offer month-to-month customers?")
    if st.button("Search Knowledge Base", use_container_width=True):
        retrieved = retrieve_documents(user_query, rag_documents)
        context = "\n\n".join([d["content"] for d in retrieved])
        
        prompt = f"Question: {user_query}\n\nReference Documents:\n{context}\n\nAnswer accurately based on references:"
        with st.spinner("Synthesizing answer..."):
            ans = generate_gemini_response(prompt)
        st.subheader("AI Answer")
        st.markdown(ans)
        
        st.subheader("Retrieved Source Files")
        for doc in retrieved:
            with st.expander(doc["filename"]):
                st.write(doc["content"])


# ============================================================
# TAB 7: BATCH PREDICTION
# ============================================================
elif page == "📂 Batch Prediction":
    st.header("📂 Batch Customer Churn Prediction")
    batch_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"], key="batch")
    
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.dataframe(batch_df.head())
        if st.button("Run Batch Prediction"):
            preds = model.predict(batch_df.drop(columns=["Churn", "customerID"], errors="ignore"))
            probs = model.predict_proba(batch_df.drop(columns=["Churn", "customerID"], errors="ignore"))[:, 1]
            batch_df["Predicted_Churn"] = preds
            batch_df["Churn_Probability"] = probs
            st.success("Batch predictions completed!")
            st.dataframe(batch_df)


# ============================================================
# TAB 8: MODEL EXPLAINABILITY
# ============================================================
elif page == "🔬 Model Explainability":
    st.header("🔬 Model Explainability & Feature Importance")
    st.write("Understand which features drive customer churn predictions in the Random Forest model.")
    
    try:
        importances = model.named_steps["classifier"].feature_importances_
        cat_enc = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
        feat_names = numeric_features + cat_enc.get_feature_names_out(categorical_features).tolist()
        
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
        ax.set_title("Top 15 Churn Prediction Drivers")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not compute feature importances: {e}")


# ============================================================
# TAB 9: ABOUT PROJECT
# ============================================================
elif page == "ℹ️ About Project":
    st.header("ℹ️ About Customer Churn AI Suite")
    st.markdown("""
    ### Enterprise AI Customer Churn Prediction & RAG System
    
    **Features:**
    - Executive Dashboard & Deep Churn Analytics
    - High-Performance Random Forest Classifier
    - RAG Knowledge Assistant powered by Google Gemini AI
    - Batch CSV prediction & model explainability
    
    **Developed by Swapna V**  
    *Data Analyst | Junior Data Scientist | ML Associate*
    """)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.caption("Customer Churn AI Suite | Machine Learning + RAG + Gemini | Developed by Swapna V")
