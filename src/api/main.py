"""
FastAPI Backend — Anomaly Detection Platform
Endpoints:
  POST /predict/fraud
  POST /predict/intrusion
  GET  /health
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

app = FastAPI(
    title="Anomaly Detection Platform API",
    description="Fraud and Network Intrusion Detection using ML",
    version="1.0.0"
)

# ── Model Registry ──
MODELS = {}

def load_models():
    """Load all trained models at startup."""
    try:
        MODELS['fraud_xgb'] = joblib.load('models/fraud/xgboost_model.pkl')
        MODELS['fraud_iso'] = joblib.load('models/fraud/isolation_forest.pkl')
        MODELS['fraud_scaler'] = joblib.load('models/fraud/fraud_scaler.pkl')
        print("✅ Fraud models loaded.")
    except Exception as e:
        print(f"⚠️ Fraud models not found: {e}")

    try:
        MODELS['intrusion_xgb'] = joblib.load('models/intrusion/xgboost_model.pkl')
        MODELS['intrusion_scaler'] = joblib.load('models/intrusion/intrusion_scaler.pkl')
        MODELS['intrusion_encoders'] = joblib.load('models/intrusion/label_encoders.pkl')
        print("✅ Intrusion models loaded.")
    except Exception as e:
        print(f"⚠️ Intrusion models not found: {e}")

load_models()


# ── Input Schemas ──

class FraudInput(BaseModel):
    """
    Input schema for fraud detection.
    Expects V1-V28 PCA features + Amount.
    Time is dropped during preprocessing.
    """
    features: list[float] = Field(
        ...,
        min_length=29,
        max_length=29,
        description="29 features: V1-V28 + Amount"
    )

class IntrusionInput(BaseModel):
    """
    Input schema for intrusion detection.
    Expects 41 NSL-KDD features as a flat list.
    """
    features: list[float] = Field(
        ...,
        min_length=41,
        max_length=41,
        description="41 NSL-KDD features"
    )


# ── Response Helpers ──

def get_risk_level(score: float) -> str:
    """Convert probability score to human-readable risk level."""
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    elif score >= 0.3:
        return "LOW"
    else:
        return "MINIMAL"


# ── Endpoints ──

@app.get("/health")
def health_check():
    """Check API and model status."""
    return {
        "status": "running",
        "models_loaded": list(MODELS.keys()),
        "version": "1.0.0"
    }


@app.post("/predict/fraud")
def predict_fraud(input_data: FraudInput):
    """
    Predict if a transaction is fraudulent.
    Returns probability, binary prediction, and risk level.
    """
    if 'fraud_xgb' not in MODELS:
        raise HTTPException(status_code=503,
                           detail="Fraud model not loaded")

    try:
        # Feature names V1-V28 + Amount
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        features_array = np.array(input_data.features).reshape(1, -1)
        df = pd.DataFrame(features_array, columns=feature_names)

        # Scale Amount using saved scaler
        scaler = MODELS['fraud_scaler']
        df['Amount'] = scaler.transform(df[['Amount']])

        # XGBoost prediction
        xgb_prob = float(MODELS['fraud_xgb'].predict_proba(df)[0][1])

        # Isolation Forest anomaly score
        iso_score = float(-MODELS['fraud_iso'].decision_function(df)[0])
        iso_normalized = max(0, min(1, (iso_score + 0.5)))

        # Combined risk score
        combined_score = round(0.7 * xgb_prob + 0.3 * iso_normalized, 4)
        prediction = int(combined_score >= 0.5)

        return {
            "prediction": prediction,
            "label": "FRAUD" if prediction == 1 else "NORMAL",
            "fraud_probability": round(xgb_prob, 4),
            "anomaly_score": round(iso_normalized, 4),
            "combined_risk_score": combined_score,
            "risk_level": get_risk_level(combined_score)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/intrusion")
def predict_intrusion(input_data: IntrusionInput):
    """
    Predict if network traffic is an intrusion.
    Returns probability, binary prediction, and risk level.
    """
    if 'intrusion_xgb' not in MODELS:
        raise HTTPException(status_code=503,
                           detail="Intrusion model not loaded")

    try:
        # NSL-KDD feature names (41 features, no label/difficulty)
        from src.intrusion.preprocessor import COLUMNS, CATEGORICAL_COLS
        feature_names = [c for c in COLUMNS
                        if c not in ['label', 'difficulty']]

        features_array = np.array(input_data.features).reshape(1, -1)
        df = pd.DataFrame(features_array, columns=feature_names)

        # Scale using saved scaler
        scaler = MODELS['intrusion_scaler']
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=feature_names
        )

        # XGBoost prediction
        prob = float(
            MODELS['intrusion_xgb'].predict_proba(df_scaled)[0][1]
        )
        prediction = int(prob >= 0.5)

        return {
            "prediction": prediction,
            "label": "ATTACK" if prediction == 1 else "NORMAL",
            "attack_probability": round(prob, 4),
            "risk_level": get_risk_level(prob)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))