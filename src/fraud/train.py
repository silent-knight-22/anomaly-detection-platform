"""
Fraud Detection Model Training
Models: Logistic Regression (baseline) + XGBoost (main)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, average_precision_score
)
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.fraud.preprocessor import load_fraud_data, preprocess_fraud


def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model and print full metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"ROC-AUC Score:        {roc_auc:.4f}")
    print(f"Avg Precision Score:  {avg_precision:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    return y_prob, roc_auc


def plot_roc_curves(results: dict, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 6))

    for model_name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Fraud Detection Models')
    plt.legend()
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/fraud_roc_curves.png', dpi=150)
    plt.show()
    print("ROC curve saved.")


def plot_precision_recall(results: dict, y_test):
    """Plot Precision-Recall curves — more informative for imbalanced data."""
    plt.figure(figsize=(10, 6))

    for model_name, y_prob in results.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, linewidth=2,
                 label=f'{model_name} (AP={ap:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves — Fraud Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/fraud_pr_curves.png', dpi=150)
    plt.show()
    print("PR curve saved.")


def train_all():
    """Main training function — runs full pipeline."""

    # Load and preprocess
    df = load_fraud_data('data/raw/fraud/creditcard.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_fraud(df)

    results = {}
    os.makedirs('models/fraud', exist_ok=True)

    # ── MODEL 1: Logistic Regression (Baseline) ──
    print("\n\nTraining Logistic Regression (baseline)...")
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    y_prob_lr, _ = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    results['Logistic Regression'] = y_prob_lr
    joblib.dump(lr, 'models/fraud/logistic_regression.pkl')
    print("Logistic Regression model saved.")

    # ── MODEL 2: XGBoost (Main Model) ──
    print("\n\nTraining XGBoost...")

    # scale_pos_weight handles remaining imbalance
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    y_prob_xgb, _ = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results['XGBoost'] = y_prob_xgb
    joblib.dump(xgb_model, 'models/fraud/xgboost_model.pkl')
    print("XGBoost model saved.")

    # ── PLOTS ──
    plot_roc_curves(results, y_test)
    plot_precision_recall(results, y_test)

    print("\n\n✅ All models trained and saved to models/fraud/")
    return results, y_test


if __name__ == "__main__":
    train_all()
