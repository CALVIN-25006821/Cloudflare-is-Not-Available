# Hybrid AI Credit Scoring System
Produce by the team of Cloudflare-Is-Not-Available, this project uses a hybrid approach to assess credit risk, combining a *Random Forest model* (Math Brain) for structured financial data and *Google Gemini* (Text Brain) for analyzing unstructured user stories.

<br>

---

## 🚀 Step-by-Step Setup Guide
Follow these instructions to get the project running on your local machine.

### 1. Prerequisites

- Python 3.9 or higher
- A Google Gemini API Key (Get one [here](https://aistudio.google.com/app/apikey))
  
<br>

### 2. Set up the Environment

1.  *Clone/Download* this repository.
2.  *Create a Virtual Environment* (recommended to keep dependencies isolated):
    bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    
3.  *Install Dependencies*:
    bash
    pip install -r requirements.txt
    
<br>

### 3. Configure API Key

1.  Create a file named .env in the root directory.
2.  Add your Gemini API key to it:
    env
    GEMINI_API_KEY=your_actual_api_key_here
    
<br>

### 4. Train the Model (The "Math Brain")

Before running the app, you need to generate the machine learning model.
1.  Run the training script:
    bash
    python train_model.py
    
2.  *What happens:*
    - Generates synthetic Malaysian loan data.
    - Trains a Random Forest model.
    - Saves baseline_model_rf.pkl (The Model) and cleaned_data.csv (The Data).
    - You should see "✅ All done!" when finished.
      
<br>

### 5. Run the Application

You have two options to run the system:

#### Option A: Interactive Dashboard (Streamlit) - *RECOMMENDED*

This is the user-friendly web interface.
1.  Run the app:
    bash
    ./.venv/bin/streamlit run app.py
2.  Open your browser at the URL shown (usually http://localhost:8501).
3.  Enter applicant details and a story to see the AI assessment.

#### Option B: Backend API (FastAPI)

If you want to use the system as a REST API for other apps.
1.  Start the server:
    bash
    uvicorn main:app --reload
2.  Access the API documentation at http://127.0.0.1:8000/docs.

<br>

---
        
## 📂 Project Structure

- *app.py*                : The frontend dashboard (Streamlit).
- *risk_engine.py*        : The core logic combining Math + Text models.
- *train_model.py*        : Generates data and trains the Random Forest model.
- *main.py*               : The backend API (FastAPI).
- *requirements.txt*      : List of Python libraries needed.
- *baseline_model_rf.pkl* : The saved "Math Brain" model.
  
<br>

▶️ How to Run

- Run backend
- uvicorn main:app --reload
- Run Streamlit
- streamlit run app.py

Streamlit APP: https://hackathon112025-sxahghfka9xuedaeytk5hv.streamlit.app/ 

<br>

---

## 🧩 Roles & Responsibilities

### 1. Calvin Kong Hao Xuan — Data Scientist

Deliverables:
- Cleaned structured dataset
- Baseline ML models (Logistic Regression & Random Forest)
- Google Colab notebook

Work Summary:
- Preprocessed raw credit dataset.
- Featured selection (income, DTI, loan ratio, credit history, etc.).
- Trained baseline models.
- Evaluated using AUC & F1.
- Exported cleaned_data.csv + models.

<br>

### 2. Choo Kah Lok — LLM & Model Fusion Engineer 

Deliverables:
- NLP Pipeline & Text Features
- Final Fusion Model (final_model.pkl)
- SHAP Explainability Visuals

Work Summary:
- Generated synthetic loan narratives for testing.
- Extracted risk patterns using LLMs.
- Merged text data with financial metrics.
- Trained and tuned the final fusion model.
- Visualized model decisions using SHAP.

<br>

### 3. Leow Shen En — Prototype Developer
Deliverables:
- Initial V1 App Structure
- Input Logic Configuration

Work Summary:
- Initialized the project environment.
- Built the functional V1 application skeleton.
- Defined the core data input requirements.

<br>

### 4. Soo Kang Shi — Streamlit UI & Integration Developer

Deliverables:
- Production dashboard (app.py)
- Automated report generator

Work Summary:
- Integrated Hybrid AI models.
- Implemented session state for history tracking.
- Engineered responsive, no-scroll dashboard layout.
- Developed automated PDF and PNG export features.

<br>

### 5. Priscilia Cheong Ee Cheng — Documentation Lead

Deliverables:
- Project report
- Architecture diagrams
- Presentation script

Work Summary:
- Created system diagrams.
- Wrote methodology & justification.
- Prepared presentation.

<br>

---

## 🏛️ System Architecture

- User (Streamlit UI)

- FastAPI Backend
-> Random Forest Model – Structured Risk
-> LLM (Gemini/OpenAI) – Text Analysis (Behavioural Risk)

- Fusion Layer – 70% ML + 30% LLM
  
- Final Risk Score + Explanation

<br>

### 🧠 Fusion Logic

#### Structured Risk (RF Model Input)

- Age

- Income

- Loan amount

- Loan term

- DTI

- Credit history

- Dependents


Output: Probability of default (0–1)

<br>

#### Text Risk (LLM Input)

- Analyzes loan-purpose description for:

- Sentiment

- Responsibility

- Urgency / stress

- Red flags

- Clarity


Output: Text risk score (0–1)

<br>

#### Final Fusion Formula

- final_score = (0.7 × structured_risk) + (0.3 × text_risk)

- Decision rule:

- Score > 0.5 → REJECT

- Score ≤ 0.5 → APPROVE

<br>

---

## 🔄 Project Workflow

### Step 1 — Data Generation & Cleaning

- Synthetic Malaysian credit dataset

- Feature engineering

- Missing value handling
  
<br>


### Step 2 — Baseline ML

- Train LR + RF

- Evaluate

- Export .pkl models
  
<br>


### Step 3 — LLM Text Extraction 

- Gemini prompts

- Extract text-based behavioural features

- Generate text_features.csv
  
<br>


### Step 4 — Fusion Model 

- Merge datasets

- Train final model

- SHAP interpretability
  
<br>


### Step 5 — FastAPI Backend

- Endpoint: /predict

- Loads RF model

- Calls LLM

- Returns final score + explanation
  
<br>


### Step 6 — Streamlit Dashboard

- User inputs

- Calls backend

- Displays prediction + charts

<br>

---

## 📂 Folder Structure


### 📁 data/ – Contains datasets

    📄 cleaned_data.csv – Preprocessed dataset

    📄 text_features.csv – Extracted text features

<br>

### 📁 models/ – Saved machine learning models

    📄 baseline_model_lr.pkl – Logistic Regression baseline

    📄 baseline_model_rf.pkl – Random Forest baseline

    📄 final_model.pkl – Final tuned model

<br>

### 📁 backend/ – FastAPI backend code

    📄 main.py – API entry point

    📄 risk_engine.py – Core risk assessment logic

<br>

### 📁 ui/ – User interface

    📄 app.py – Frontend (Streamlit/FastAPI)

<br>

### 📁 notebooks/ – Jupyter notebooks for experimentation

    📄 structured_data.ipynb – Structured data analysis

    📄 text_analysis.ipynb – NLP/text feature analysis

    📄 fusion_model.ipynb – ML + LLM fusion experiments

    📄 README.md – Project documentation

<br>

---

## ⭐ Project Highlights

- Hybrid ML + LLM risk assessment

- Fairer and more explainable than traditional scoring

- Malaysian-style synthetic dataset

- SHAP interpretability

- Real-time dashboard

- Modular design for easy extension
