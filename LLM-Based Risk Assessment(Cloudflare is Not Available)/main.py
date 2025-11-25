import os
import json
import joblib
import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# --- 1. Setup Environment ---
load_dotenv() 

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ WARNING: API Key not found")
else:
    genai.configure(api_key=api_key)

llm_model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

try:
    rf_model = joblib.load("baseline_model_rf.pkl")
    print("✅ Numeric model loaded successfully!")
except:
    print("❌ ERROR: baseline_model_rf.pkl not found")
    rf_model = None

# ==========================================
# 👇👇👇 Key Point! Must be named app 👇👇👇
app = FastAPI(title="Xenber AI Credit Scoring")
# ==========================================

class CreditRequest(BaseModel):
    age: int
    monthly_income: float
    loan_amount: float
    loan_term: int
    dti: float
    credit_history: int
    num_dependents: int
    applicant_name: str
    application_text: str

@app.get("/")
def home():
    return {"message": "Xenber Backend is Running!"}

@app.post("/predict")
async def predict_credit_risk(data: CreditRequest):
    base_risk = 0.5 
    if rf_model:
        input_data = pd.DataFrame([{
            "age": data.age,
            "monthly_income": data.monthly_income,
            "loan_amount": data.loan_amount,
            "loan_term": data.loan_term,
            "dti": data.dti,
            "credit_history": data.credit_history,
            "num_dependents": data.num_dependents
        }])
        base_risk = float(rf_model.predict_proba(input_data)[0][1])

    prompt = f"""
    You are a Credit Risk Analyst.
    Applicant Text: "{data.application_text}"
    Base Numeric Risk Score: {base_risk}
    
    Output JSON format:
    {{
        "analysis": "Short explanation",
        "score_adjustment": float (between -0.2 and +0.2)
    }}
    """
    
    try:
        response = llm_model.generate_content(prompt)
        ai_result = json.loads(response.text)
    except Exception:
        ai_result = {"analysis": "AI Error", "score_adjustment": 0.0}

    final_score = base_risk + ai_result.get("score_adjustment", 0)
    final_score = max(0.0, min(1.0, final_score))
    decision = "REJECT" if final_score > 0.5 else "APPROVE"

    return {
        "base_model_risk": round(base_risk, 2),
        "final_risk_score": round(final_score, 2),
        "decision": decision,
        "ai_explanation": ai_result
    }
