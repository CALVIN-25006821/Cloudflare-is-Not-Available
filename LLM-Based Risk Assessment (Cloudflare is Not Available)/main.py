"""
FILENAME: main.py
PURPOSE: This is the main entry point for the FastAPI backend.
DESCRIPTION:
    - Sets up the FastAPI application.
    - Loads environment variables and the Gemini API key.
    - Loads the pre-trained Random Forest model (baseline_model_rf.pkl).
    - Defines the data model for credit requests.
    - Exposes the `/predict` endpoint to handle credit risk assessments.
    - Combines the numeric model score with the LLM-based text analysis.
"""
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from risk_engine import get_total_risk

# --- 1. Setup Environment ---
load_dotenv() 

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
    try:
        # Call the centralized Risk Engine
        result = get_total_risk(
            age=data.age,
            income=data.monthly_income,
            loan_amount=data.loan_amount,
            loan_term=data.loan_term,
            dti=data.dti,
            credit_history=data.credit_history,
            dependents=data.num_dependents,
            user_story=data.application_text
        )
        
        # Determine decision based on Final_Risk (0-100)
        # Using the same threshold as the UI (60 for High Risk, 40 for Medium)
        # But for simple Approve/Reject, let's say > 50 is Reject
        final_score = result['Final_Risk']
        decision = "REJECT" if final_score > 50 else "APPROVE"

        return {
            "base_model_risk": result['Math_Score'],
            "text_risk_score": result['Text_Score'],
            "final_risk_score": final_score,
            "decision": decision,
            "ai_explanation": result['Text_Analysis']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
