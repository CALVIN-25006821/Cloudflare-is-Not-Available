# Cloudflare-is-Not-Available

🚀 Xenber AI — Hybrid LLM & ML Credit Scoring System

A full-stack intelligent credit-risk assessment system combining:

• Machine Learning (Random Forest) for structured financial data

• LLM (Gemini / OpenAI) for text-based behavioural insights

• Fusion Model that merges both signals for fairer and more explainable scoring

• FastAPI Backend for model serving

• Streamlit Dashboard for real-time predictions

---

🧩 Roles & Responsibilities

1. Calvin Kong Hao Xuan — Data Scientist

Deliverables:

• Cleaned structured dataset

• Baseline ML models (Logistic Regression & Random Forest)

• Google Colab notebook


Work Summary:

• Preprocess raw credit dataset

• Feature selection (income, DTI, loan ratio, credit history, etc.)

• Train baseline models

• Evaluate using AUC & F1

• Export cleaned_data.csv + models

---

2. Choo Kah Lok — LLM Engineer

Deliverables:

• LLM prompts

• Text feature extraction notebook

• text_features.csv


Work Summary:

• Generate synthetic loan-purpose descriptions

• Extract sentiment, risk patterns, urgency, financial behaviour

• Convert text into numeric features

---

3. Soo Kang Shi — Fusion Model Engineer

Deliverables:

• Fusion dataset

• SHAP explainability

• final_model.pkl


Work Summary:

• Merge structured + text features

• Train fusion model

• Hyperparameter tuning

• Generate SHAP visuals + explanations

---

4. Leow Shen En — Streamlit UI Developer

Deliverables:

• app.py (Streamlit app)


Work Summary:

• Build clean input form

• Connect to prediction backend

• Show final score & explanation

• Plot feature importance

---

5. Prescilia Cheong Ee Cheng — Documentation Lead

Deliverables:

• Project report

• Architecture diagrams

• Presentation script


Work Summary:

• Create system diagrams

• Write methodology & justification

• Prepare presentation

---


🏛️ System Architecture

User (Streamlit UI)
   │
   ▼
FastAPI Backend
   │
   ├──▶ Random Forest Model (Structured Risk)
   │
   └──▶ LLM (Gemini/OpenAI) Text Analysis (Behavioural Risk)
            │
            ▼
       Fusion Layer (70% ML + 30% LLM)
              │
              ▼
   Final Risk Score + Explanation


---

🔄 Project Workflow

Step 1 — Data Generation & Cleaning

• Synthetic Malaysian credit dataset

• Feature engineering

• Missing value handling


Step 2 — Baseline ML (Calvin)

• Train LR + RF

• Evaluate

• Export .pkl models


Step 3 — LLM Text Extraction (Kah Lok)

• Gemini prompts

• Extract text-based behavioural features

• Generate text_features.csv


Step 4 — Fusion Model (Kang Shi)

• Merge datasets

• Train final model

• SHAP interpretability


Step 5 — FastAPI Backend

• Endpoint: /predict

• Loads RF model

• Calls LLM

• Returns final score + explanation


Step 6 — Streamlit Dashboard

• User inputs

• Calls backend

• Displays prediction + charts



---

🧠 Fusion Logic

Structured Risk (RF Model Input)

• Age

• Income

• Loan amount

• Loan term

• DTI

• Credit history

• Dependents


Output: Probability of default (0–1)


---

Text Risk (LLM Input)

• Analyzes loan-purpose description for:

• Sentiment

• Responsibility

• Urgency / stress

• Red flags

• Clarity


Output: Text risk score (0–1)


---

Final Fusion Formula

final_score = (0.7 × structured_risk) + (0.3 × text_risk)

Decision rule:

Score > 0.5 → REJECT

Score ≤ 0.5 → APPROVE



---

📁 Folder Structure

/project
│
├── data/
│   ├── cleaned_data.csv
│   └── text_features.csv
│
├── models/
│   ├── baseline_model_lr.pkl
│   ├── baseline_model_rf.pkl
│   └── final_model.pkl
│
├── backend/
│   ├── main.py
│   └── risk_engine.py
│
├── ui/
│   └── app.py
│
├── notebooks/
│   ├── structured_data.ipynb
│   ├── text_analysis.ipynb
│   └── fusion_model.ipynb
│
└── README.md


---

⚙️ Installation

pip install -r requirements.txt

requirements.txt

fastapi
uvicorn
python-dotenv
google-generativeai
pandas
numpy
scikit-learn
joblib
pydantic
streamlit


---

▶️ How to Run

• Run backend

• uvicorn main:app --reload

• Run Streamlit

• streamlit run app.py


---

⭐ Project Highlights

• Hybrid ML + LLM risk assessment

• Fairer and more explainable than traditional scoring

• Malaysian-style synthetic dataset

• SHAP interpretability

• Real-time dashboard

• Modular design for easy extension
