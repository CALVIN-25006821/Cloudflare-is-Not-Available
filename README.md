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

<br>

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

<br>

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

<br>

Work Summary:

• Merge structured + text features

• Train fusion model

• Hyperparameter tuning

• Generate SHAP visuals + explanations

---

4. Leow Shen En — Streamlit UI Developer

Deliverables:

• app.py (Streamlit app)

<br>

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


<br>
Work Summary:

• Create system diagrams

• Write methodology & justification

• Prepare presentation

---

🏛️ System Architecture

• User (Streamlit UI)
<br>

• FastAPI Backend

-> Random Forest Model – Structured Risk

-> LLM (Gemini/OpenAI) – Text Analysis (Behavioural Risk)

• Fusion Layer – 70% ML + 30% LLM

• Final Risk Score + Explanation


---

🔄 Project Workflow

Step 1 — Data Generation & Cleaning

• Synthetic Malaysian credit dataset

• Feature engineering

• Missing value handling
<br>


Step 2 — Baseline ML (Calvin)

• Train LR + RF

• Evaluate

• Export .pkl models
<br>


Step 3 — LLM Text Extraction (Kah Lok)

• Gemini prompts

• Extract text-based behavioural features

• Generate text_features.csv
<br>


Step 4 — Fusion Model (Kang Shi)

• Merge datasets

• Train final model

• SHAP interpretability
<br>


Step 5 — FastAPI Backend

• Endpoint: /predict

• Loads RF model

• Calls LLM

• Returns final score + explanation
<br>


Step 6 — Streamlit Dashboard

• User inputs

• Calls backend

• Displays prediction + charts
<br>


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

📂 Folder Structure


📁 data/ – Contains datasets

📄 cleaned_data.csv – Preprocessed dataset

📄 text_features.csv – Extracted text features
<br>

📁 models/ – Saved machine learning models

📄 baseline_model_lr.pkl – Logistic Regression baseline

📄 baseline_model_rf.pkl – Random Forest baseline

📄 final_model.pkl – Final tuned model
<br>

📁 backend/ – FastAPI backend code

📄 main.py – API entry point

📄 risk_engine.py – Core risk assessment logic
<br>

📁 ui/ – User interface

📄 app.py – Frontend (Streamlit/FastAPI)
<br>

📁 notebooks/ – Jupyter notebooks for experimentation

📄 structured_data.ipynb – Structured data analysis

📄 text_analysis.ipynb – NLP/text feature analysis

📄 fusion_model.ipynb – ML + LLM fusion experiments

📄 README.md – Project documentation
<br>

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
