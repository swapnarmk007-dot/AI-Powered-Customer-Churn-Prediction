# ==========================================================
# AI CUSTOMER CHURN PREDICTION WEB APPLICATION
# Python + Streamlit + Random Forest + RAG + Gemini AI
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Gemini AI
from google import genai


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Customer Churn Prediction")
st.subheader("Machine Learning + RAG + Generative AI")

st.write(
    "Predict customer churn and generate intelligent "
    "customer retention recommendations."
)

st.divider()


# --------------------------------------------------
# 2. LOAD OR AUTO-CREATE MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def load_or_create_model():
    model_path = "churn_model.pkl"
    features_path = "feature_columns.pkl"

    # Auto-create model files if they do not exist
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        np.random.seed(42)
        n_samples = 200

        train_df = pd.DataFrame({
            "gender": np.random.choice(["Female", "Male"], n_samples),
            "SeniorCitizen": np.random.choice([0, 1], n_samples),
            "Partner": np.random.choice(["Yes", "No"], n_samples),
            "Dependents": np.random.choice(["Yes", "No"], n_samples),
            "tenure": np.random.randint(0, 72, n_samples),
            "PhoneService": np.random.choice(["Yes", "No"], n_samples),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_samples),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n_samples),
            "MonthlyCharges": np.random.uniform(20.0, 120.0, n_samples),
            "TotalCharges": np.random.uniform(100.0, 8000.0, n_samples),
            "Churn": np.random.choice([0, 1], n_samples)
        })

        X = train_df.drop(columns=["Churn"])
        y = train_df["Churn"]

        X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
        feature_columns = list(X_encoded.columns)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_encoded, y)

        joblib.dump(model, model_path)
        joblib.dump(feature_columns, features_path)

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    return model, feature_columns


try:
    model, feature_columns = load_or_create_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# --------------------------------------------------
# 3. LOAD OR AUTO-CREATE RAG DOCUMENTS
# --------------------------------------------------

DOCUMENTS_PATH = Path("rag/documents")


@st.cache_data
def load_rag_documents():
    documents = []
    
    # Ensure directory exists and has a sample doc if empty
    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    sample_file = DOCUMENTS_PATH / "retention_policy.txt"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("Company Retention Policy: Offer 15% discount on annual contracts for high risk customers. Provide dedicated technical support onboarding for fiber optic users.")

    for file_path in DOCUMENTS_PATH.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append({
                "filename": file_path.name,
                "content": file.read()
            })

    return documents


try:
    documents = load_rag_documents()
except Exception as e:
    st.error(f"RAG document loading error: {e}")
    st.stop()


# --------------------------------------------------
# 4. RAG RETRIEVAL
# --------------------------------------------------

def retrieve_documents(query, documents, top_k=3):
    query_words = set(re.findall(r"\w+", query.lower()))
    results = []

    for document in documents:
        content = document["content"]
        content_words = set(re.findall(r"\w+", content.lower()))
        score = len(query_words.intersection(content_words))
        results.append((score, document))

    results.sort(key=lambda item: item[0], reverse=True)

    return [
        document
        for score, document in results[:top_k]
        if score >= 0
    ]


# --------------------------------------------------
# 5. GEMINI AI RESPONSE
# --------------------------------------------------

def generate_recommendations(customer_details, prediction, probability):
    api_key = st.secrets.get("GEMINI_API_KEY", None)

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Gemini API key is not configured. "
            "Please configure `GEMINI_API_KEY` in Streamlit secrets or environment variables "
            "to generate AI-powered recommendations."
        ), []

    query = f"""
    Customer churn prediction: {prediction}.
    Churn probability: {probability:.2f}%.
    Customer details: {customer_details}.
    Suggest customer retention strategies.
    """

    retrieved_documents = retrieve_documents(query, documents)

    context = "\n\n".join(
        document["content"]
        for document in retrieved_documents
    ) if retrieved_documents else "No internal policy documents found."

    prompt = f"""
    You are an experienced customer retention analyst.

    Use the following retrieved knowledge as your reference.

    RETRIEVED KNOWLEDGE:
    {context}

    CUSTOMER INFORMATION:
    {customer_details}

    MACHINE LEARNING PREDICTION:
    {prediction}

    CHURN PROBABILITY:
    {probability:.2f}%

    TASK:
    1. Explain the prediction in simple language.
    2. Identify possible customer churn risk factors.
    3. Recommend five practical retention strategies.
    4. Explain how customer support can help.
    5. Suggest suitable pricing or loyalty offers.

    Only use the retrieved knowledge for company-specific recommendations. 
    Clearly identify any additional suggestions as general recommendations.
    Do not claim that the prediction proves why a customer will leave.
    """

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text, retrieved_documents

    except Exception as e:
        return f"Gemini API error: {e}", retrieved_documents


# --------------------------------------------------
# 6. CUSTOMER INFORMATION FORM
# --------------------------------------------------

st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

with col2:
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)

st.divider()


# --------------------------------------------------
# 7. CUSTOMER CHURN PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Customer Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    input_encoded = pd.get_dummies(input_data, drop_first=True, dtype=int)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1] * 100

    prediction_label = "Churn" if prediction == 1 else "No Churn"

    # --------------------------------------------------
    # 8. DISPLAY ML RESULTS
    # --------------------------------------------------

    st.header("📈 Machine Learning Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{probability:.2f}%")

    with col2:
        st.metric("Prediction", prediction_label)

    if prediction == 1:
        st.error("⚠️ Customer is predicted to churn.")
    else:
        st.success("✅ Customer is predicted to stay.")

    chart_data = pd.DataFrame({
        "Customer Status": ["No Churn", "Churn"],
        "Probability": [100 - probability, probability]
    })

    st.subheader("Customer Churn Probability")
    st.bar_chart(chart_data.set_index("Customer Status"))

    # --------------------------------------------------
    # 9. RAG + GEMINI RECOMMENDATIONS
    # --------------------------------------------------

    st.divider()
    st.header("🤖 RAG + Gemini AI Recommendations")

    customer_details = input_data.iloc[0].to_dict()

    with st.spinner("Retrieving relevant documents and generating recommendations..."):
        ai_response, retrieved_documents = generate_recommendations(
            customer_details,
            prediction_label,
            probability
        )

    st.subheader("AI Retention Recommendations")
    st.markdown(ai_response)

    st.subheader("📚 Retrieved RAG Documents")

    if retrieved_documents:
        for document in retrieved_documents:
            with st.expander(document["filename"]):
                st.write(document["content"])
    else:
        st.info("No relevant documents retrieved.")


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    **AI-Powered Customer Churn Prediction System**

    Technologies:
    - Python
    - Random Forest Machine Learning
    - Streamlit
    - RAG
    - Google Gemini AI

    Developed by Swapna V
    """
)