"""
Quick API test script.
Run this while the FastAPI server is running.
"""

import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("=== HEALTH CHECK ===")
    print(json.dumps(response.json(), indent=2))


def test_fraud():
    """Test fraud endpoint with a sample transaction."""
    # Simulate a transaction: V1-V28 random PCA values + Amount
    np.random.seed(42)
    features = list(np.random.randn(28)) + [150.0]  # Amount = $150

    payload = {"features": features}
    response = requests.post(f"{BASE_URL}/predict/fraud", json=payload)

    print("\n=== FRAUD PREDICTION (Normal transaction) ===")
    print(json.dumps(response.json(), indent=2))


def test_fraud_suspicious():
    """Test with values that mimic fraud patterns."""
    # V14 strongly negative = fraud signal (from SHAP analysis)
    features = list(np.random.randn(28)) + [1.0]
    features[13] = -8.0   # V14 strongly negative
    features[3] = -6.0    # V4 strongly negative

    payload = {"features": features}
    response = requests.post(f"{BASE_URL}/predict/fraud", json=payload)

    print("\n=== FRAUD PREDICTION (Suspicious transaction) ===")
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