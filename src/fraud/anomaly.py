"""
Isolation Forest — Anomaly Detection Layer for Fraud.
Adds an unsupervised anomaly score on top of XGBoost predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.fraud.preprocessor import load_fraud_data, preprocess_fraud


def train_isolation_forest(X_train, contamination=0.01):
    """
    Train Isolation Forest on training data.
    contamination = expected proportion of anomalies (fraud rate ~0.17%)
    We use 0.01 (1%) as a slightly relaxed threshold.
    """
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    print("Isolation Forest trained.")
    return iso_forest


def evaluate_isolation_forest(iso_forest, X_test, y_test):
    """
    Evaluate Isolation Forest predictions.
    IF returns: -1 = anomaly, 1 = normal
    We convert to: 1 = fraud, 0 = normal
    """
    # Raw predictions (-1 or 1)
    raw_preds = iso_forest.predict(X_test)

    # Convert: -1 → 1 (fraud), 1 → 0 (normal)
    if_preds = (raw_preds == -1).astype(int)

    # Anomaly scores (lower = more anomalous)
    anomaly_scores = iso_forest.decision_function(X_test)
    # Invert so higher = more anomalous
    anomaly_scores = -anomaly_scores

    print("\n=== ISOLATION FOREST RESULTS ===")
    print(classification_report(y_test, if_preds,
                                target_names=['Normal', 'Fraud']))

    return if_preds, anomaly_scores


def combine_predictions(xgb_probs, if_scores, threshold=0.5):
    """
    Combine XGBoost probability with Isolation Forest anomaly score.
    Final risk score = weighted combination.
    This is the 'dual-layer' detection that makes the system powerful.
    """
    # Normalize IF scores to [0, 1]
    if_normalized = (if_scores - if_scores.min()) / \
                    (if_scores.max() - if_scores.min())

    # Weighted combination: 70% XGBoost + 30% Isolation Forest
    combined_score = 0.7 * xgb_probs + 0.3 * if_normalized

    combined_preds = (combined_score >= threshold).astype(int)
    return combined_score, combined_preds


def plot_anomaly_scores(anomaly_scores, y_test):
    """Visualize anomaly score distribution by class."""
    plt.figure(figsize=(12, 5))

    fraud_scores = anomaly_scores[y_test.values == 1]
    normal_scores = anomaly_scores[y_test.values == 0]

    plt.hist(normal_scores, bins=100, alpha=0.6,
             color='steelblue', label='Normal', density=True)
    plt.hist(fraud_scores, bins=30, alpha=0.7,
             color='crimson', label='Fraud', density=True)

    plt.xlabel('Anomaly Score (higher = more anomalous)')
    plt.ylabel('Density')
    plt.title('Isolation Forest Anomaly Score Distribution')
    plt.legend()
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/isolation_forest_scores.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: isolation_forest_scores.png")


def run_anomaly_detection():
    """Full anomaly detection pipeline."""

    print("=== ISOLATION FOREST ANOMALY DETECTION ===\n")

    # Load and preprocess
    df = load_fraud_data('data/raw/fraud/creditcard.csv')
    X_train, X_test, y_train, y_test, _ = preprocess_fraud(df)

    # Load XGBoost for combination
    xgb_model = joblib.load('models/fraud/xgboost_model.pkl')
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Train Isolation Forest
    iso_forest = train_isolation_forest(X_train)

    # Evaluate standalone
    if_preds, anomaly_scores = evaluate_isolation_forest(
        iso_forest, X_test, y_test
    )

    # Plot score distribution
    plot_anomaly_scores(anomaly_scores, y_test)

    # Combine with XGBoost
    print("\n=== COMBINED DETECTION (XGBoost + Isolation Forest) ===")
    combined_scores, combined_preds = combine_predictions(
        xgb_probs, anomaly_scores
    )
    combined_auc = roc_auc_score(y_test, combined_scores)
    print(classification_report(y_test, combined_preds,
                                target_names=['Normal', 'Fraud']))
    print(f"Combined ROC-AUC: {combined_auc:.4f}")

    # Save Isolation Forest model
    os.makedirs('models/fraud', exist_ok=True)
    joblib.dump(iso_forest, 'models/fraud/isolation_forest.pkl')
    print("\nIsolation Forest saved to models/fraud/")

    print("\n✅ Anomaly detection complete.")
    return iso_forest, combined_scores, y_test


if __name__ == "__main__":
    run_anomaly_detection()