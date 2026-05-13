"""
FastAPI Backend - Anomaly Detection Platform
Endpoints:
  POST /predict/fraud
  POST /predict/intrusion
  GET  /health
"""

from __future__ import annotations

import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.fraud.preprocessor import transform_fraud_records  # noqa: E402

app = FastAPI(
    title="Anomaly Detection Platform API",
    description="Fraud and Network Intrusion Detection using ML",
    version="1.0.0",
)

MODELS: dict[str, Any] = {}


def load_models() -> None:
    """Load all trained models at startup."""
    try:
        fraud_xgb = joblib.load("models/fraud/xgboost_model.pkl")
        fraud_preprocessor = joblib.load("models/fraud/fraud_preprocessor.pkl")
        MODELS["fraud_xgb"] = fraud_xgb
        MODELS["fraud_preprocessor"] = fraud_preprocessor
        if os.path.exists("models/fraud/isolation_forest.pkl"):
            MODELS["fraud_iso"] = joblib.load("models/fraud/isolation_forest.pkl")
        print("Fraud models loaded.")
    except Exception as exc:
        print(f"Fraud models not loaded: {exc}")

    try:
        MODELS["intrusion_xgb"] = joblib.load("models/intrusion/xgboost_model.pkl")
        MODELS["intrusion_scaler"] = joblib.load("models/intrusion/intrusion_scaler.pkl")
        MODELS["intrusion_encoders"] = joblib.load("models/intrusion/label_encoders.pkl")
        print("Intrusion models loaded.")
    except Exception as exc:
        print(f"Intrusion models not loaded: {exc}")


load_models()


class FraudInput(BaseModel):
    """
    Input schema for fraud detection.

    For IEEE-CIS, send a raw transaction record with CSV columns such as
    TransactionAmt, ProductCD, card*, C*, D*, M*, V*, and id_*.
    """

    record: dict[str, Any] | None = Field(
        default=None,
        description="Single raw IEEE-CIS fraud transaction record",
    )
    records: list[dict[str, Any]] | None = Field(
        default=None,
        description="Batch of raw IEEE-CIS fraud transaction records",
    )
    features: list[float] | None = Field(
        default=None,
        description="Legacy support only: 29 values for V1-V28 plus Amount",
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
        description="41 NSL-KDD features",
    )


def get_risk_level(score: float) -> str:
    """Convert probability score to human-readable risk level."""
    if score >= 0.8:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    if score >= 0.3:
        return "LOW"
    return "MINIMAL"


@app.get("/health")
def health_check():
    """Check API and model status."""
    return {
        "status": "running",
        "models_loaded": list(MODELS.keys()),
        "version": "1.0.0",
    }


def _coerce_fraud_records(input_data: FraudInput) -> tuple[list[dict[str, Any]], bool]:
    if input_data.records is not None:
        return input_data.records, True
    if input_data.record is not None:
        return [input_data.record], False
    if input_data.features is not None:
        if len(input_data.features) != 29:
            raise ValueError("Legacy fraud features must contain V1-V28 plus Amount (29 values).")
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        return [dict(zip(feature_names, input_data.features))], False
    raise ValueError("Provide either 'record', 'records', or legacy 'features'.")


@app.post("/predict/fraud")
def predict_fraud(input_data: FraudInput):
    """
    Predict fraudulent IEEE-CIS transactions.
    Returns probability, binary prediction, and risk level.
    """
    if "fraud_xgb" not in MODELS or "fraud_preprocessor" not in MODELS:
        raise HTTPException(
            status_code=503,
            detail="Fraud model/preprocessor not loaded. Retrain fraud models for the IEEE dataset.",
        )

    try:
        raw_records, is_batch = _coerce_fraud_records(input_data)
        transformed = transform_fraud_records(raw_records)

        xgb_probs = MODELS["fraud_xgb"].predict_proba(transformed)[:, 1]

        if "fraud_iso" in MODELS:
            iso_scores = -MODELS["fraud_iso"].decision_function(transformed)
            min_score = float(np.min(iso_scores))
            max_score = float(np.max(iso_scores))
            if max_score > min_score:
                anomaly_scores = (iso_scores - min_score) / (max_score - min_score)
            else:
                anomaly_scores = np.clip(iso_scores + 0.5, 0, 1)
        else:
            anomaly_scores = xgb_probs

        combined_scores = 0.7 * xgb_probs + 0.3 * anomaly_scores
        predictions = (combined_scores >= 0.5).astype(int)

        results = []
        for i, prediction in enumerate(predictions):
            combined_score = round(float(combined_scores[i]), 4)
            results.append(
                {
                    "prediction": int(prediction),
                    "label": "FRAUD" if prediction == 1 else "NORMAL",
                    "fraud_probability": round(float(xgb_probs[i]), 4),
                    "anomaly_score": round(float(anomaly_scores[i]), 4),
                    "combined_risk_score": combined_score,
                    "risk_level": get_risk_level(combined_score),
                }
            )

        return {"results": results} if is_batch else results[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/intrusion")
def predict_intrusion(input_data: IntrusionInput):
    """
    Predict if network traffic is an intrusion.
    Returns probability, binary prediction, and risk level.
    """
    if "intrusion_xgb" not in MODELS:
        raise HTTPException(status_code=503, detail="Intrusion model not loaded")

    try:
        from src.intrusion.preprocessor import COLUMNS

        feature_names = [c for c in COLUMNS if c not in ["label", "difficulty"]]
        features_array = np.array(input_data.features).reshape(1, -1)
        df = pd.DataFrame(features_array, columns=feature_names)

        scaler = MODELS["intrusion_scaler"]
        df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)

        prob = float(MODELS["intrusion_xgb"].predict_proba(df_scaled)[0][1])
        prediction = int(prob >= 0.5)

        return {
            "prediction": prediction,
            "label": "ATTACK" if prediction == 1 else "NORMAL",
            "attack_probability": round(prob, 4),
            "risk_level": get_risk_level(prob),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
