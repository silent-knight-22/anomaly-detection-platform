"""
Quick API test script.
Run this while the FastAPI server is running.
"""

import requests
import json
import numpy as np
import pandas as pd
import os

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("=== HEALTH CHECK ===")
    print(json.dumps(response.json(), indent=2))


def test_fraud():
    """Test fraud endpoint with a sample transaction."""
    path = "data/raw/fraud/train_transaction.csv"
    if not os.path.exists(path):
        print("Skipping fraud test: IEEE train_transaction.csv not found")
        return
    record = pd.read_csv(path, nrows=1).iloc[0]
    record = record.drop(labels=["isFraud"], errors="ignore")
    record = record.where(pd.notna(record), None)

    payload = {"record": record.to_dict()}
    response = requests.post(f"{BASE_URL}/predict/fraud", json=payload)

    print("\n=== FRAUD PREDICTION (IEEE sample transaction) ===")
    print(json.dumps(response.json(), indent=2))


def test_fraud_suspicious():
    """Test with values that mimic fraud patterns."""
    path = "data/raw/fraud/train_transaction.csv"
    if not os.path.exists(path):
        print("Skipping fraud suspicious test: IEEE train_transaction.csv not found")
        return
    rows = pd.read_csv(path, nrows=5000)
    fraud_rows = rows[rows.get("isFraud", 0) == 1]
    row = fraud_rows.iloc[0] if not fraud_rows.empty else rows.iloc[-1]
    record = row.drop(labels=["isFraud"], errors="ignore")
    record = record.where(pd.notna(record), None)

    payload = {"record": record.to_dict()}
    response = requests.post(f"{BASE_URL}/predict/fraud", json=payload)

    print("\n=== FRAUD PREDICTION (IEEE suspicious sample) ===")
    print(json.dumps(response.json(), indent=2))


def test_intrusion():
    """Test intrusion endpoint with sample network traffic."""
    # 41 features for NSL-KDD (all zeros = normal-like traffic)
    features = [0.0] * 41
    features[0] = 1.0    # duration
    features[4] = 1000.0 # src_bytes
    features[5] = 500.0  # dst_bytes

    payload = {"features": features}
    response = requests.post(f"{BASE_URL}/predict/intrusion", json=payload)

    print("\n=== INTRUSION PREDICTION ===")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("Testing Anomaly Detection API...\n")
    test_health()
    test_fraud()
    test_fraud_suspicious()
    test_intrusion()
    print("\n✅ All tests complete.")
