# AI-Powered Customer Churn Prediction & Retention Recommendation System

## Project Overview

The **AI-Powered Customer Churn Prediction & Retention Recommendation System** is a machine learning application that predicts whether a customer is likely to leave a company and provides AI-generated retention recommendations.

The project combines a Random Forest classification model, a document-based retrieval system, Google Gemini AI, and Streamlit to support customer retention decisions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Google Gemini AI
- Retrieval-Augmented Generation (RAG)
- Streamlit
- Joblib

## Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer information, including:

- Customer demographics
- Services subscribed to
- Contract details
- Monthly charges
- Total charges
- Customer tenure
- Churn status

**Target variable:** `Churn`

The model predicts whether a customer is likely to churn (`Yes`) or remain with the company (`No`).

## Machine Learning Model

A **Random Forest Classifier** is used to predict customer churn.

### Model workflow

1. Load the customer dataset.
2. Clean and preprocess the data.
3. Handle missing and incorrect values.
4. Encode categorical features.
5. Split the dataset into training and testing sets.
6. Train the Random Forest model.
7. Save the trained model using Joblib.
8. Use the model to predict customer churn.

## AI-Powered Retention Recommendation

The project includes a document-based retrieval system that searches local knowledge-base documents for relevant retention strategies.

The retrieved information is passed to Google Gemini AI to generate customer retention recommendations.

**Current retrieval approach:** Keyword-based document retrieval.

**Future enhancement:** Semantic search using embeddings and ChromaDB.

## Application Features

- Customer churn prediction
- Random Forest machine learning model
- Customer information input
- Churn prediction results
- Document-based retention strategy retrieval
- Google Gemini AI-generated recommendations
- Interactive Streamlit interface

## Project Architecture

```text
Customer Information
        |
        v
Data Preprocessing
        |
        v
Random Forest Model
        |
        v
Churn Prediction
        |
        v
Knowledge Base Retrieval
        |
        v
Google Gemini AI
        |
        v
Retention Recommendations
        |
        v
Streamlit Dashboard
```

## Project Structure

```text
AI-Powered Customer Churn Prediction/
│
├── app.py
├── churn_model.pkl
├── feature_columns.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── rag/
│   ├── rag_pipeline.py
│   ├── customer_support.txt
│   ├── pricing_strategies.txt
│   │
│   └── documents/
│       ├── retention_policy.txt
│       └── retention_strategies.txt
│
├── .streamlit/
│   └── secrets.toml
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** The `.streamlit/secrets.toml` file is for local configuration only. Do not upload it to GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/swapnarmk007-dot/AI-Powered-Customer-Churn-Prediction.git
```

### 2. Open the project folder

```bash
cd AI-Powered-Customer-Churn-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Gemini API Configuration

The application uses Google Gemini AI to generate retention recommendations.

1. Visit Google AI Studio.
2. Generate a Gemini API key.
3. Create a `.streamlit` folder if it does not exist.
4. Inside it, create a file named `secrets.toml`.
5. Add the following configuration:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace `YOUR_GEMINI_API_KEY` with your actual API key.

**Security:** Never upload your API key or `secrets.toml` file to GitHub.

## Run the Application

Run the following command in the terminal:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Future Enhancements

- Implement semantic search using embeddings.
- Integrate ChromaDB as a vector database.
- Improve model performance through hyperparameter tuning.
- Add customer churn analytics dashboards.
- Deploy the application with Streamlit Community Cloud.

## Author

**Swapna V**

- GitHub: https://github.com/swapnarmk007-dot
- LinkedIn: https://linkedin.com/in/swapna-v-5323663b4

## Project Purpose

This project demonstrates the application of machine learning, document retrieval, and generative AI to customer churn prediction and retention decision support.

## 🚀 Live Demo

**Live Application:** [https://ai-powered-customer-churn-prediction-hue7ctp3dgwaealbz49678.streamlit.app/](https://ai-powered-customer-churn-prediction-hue7ctp3dgwaealbz49678.streamlit.app/)
   **[live demo](https://ai-powered-customer-churn-prediction-hue7ctp3dgwaealbz49678.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-powered-customer-churn-pue7ctp3dgwaealbz49678.streamlit.app/)
