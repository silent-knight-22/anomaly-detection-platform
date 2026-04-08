"""
Intrusion Detection Model Training.
Models: Random Forest + XGBoost (binary: normal vs attack)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, average_precision_score,
    confusion_matrix
)
import xgboost as xgb
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.intrusion.preprocessor import preprocess_intrusion


def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate and print full metrics."""
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
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Attack']
    ))

    return y_prob, roc_auc


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    """Plot and save confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    fname = model_name.lower().replace(' ', '_')
    plt.savefig(f'reports/figures/intrusion_cm_{fname}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved for {model_name}")


def plot_roc_curves(results: dict, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 6))

    for model_name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, linewidth=2,
                 label=f'{model_name} (AUC={auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Intrusion Detection Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/intrusion_roc_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("ROC curves saved.")


def train_all():
    """Main training function."""

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_intrusion(
        'data/raw/intrusion/KDDTrain+.txt',
        'data/raw/intrusion/KDDTest+.txt'
    )[:4]

    results = {}
    os.makedirs('models/intrusion', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    # ── MODEL 1: Random Forest ──
    print("\n\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob_rf, _ = evaluate_model(rf, X_test, y_test, "Random Forest")
    results['Random Forest'] = y_prob_rf
    plot_confusion_matrix(rf, X_test, y_test, "Random Forest")
    joblib.dump(rf, 'models/intrusion/random_forest.pkl')
    print("Random Forest saved.")

    # ── MODEL 2: XGBoost ──
    print("\n\nTraining XGBoost...")
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
    y_prob_xgb, _ = evaluate_model(
        xgb_model, X_test, y_test, "XGBoost"
    )
    results['XGBoost'] = y_prob_xgb
    plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost")
    joblib.dump(xgb_model, 'models/intrusion/xgboost_model.pkl')
    print("XGBoost saved.")

    # ROC curves
    plot_roc_curves(results, y_test)

    print("\n\n✅ All intrusion models trained and saved.")
    return results, y_test


if __name__ == "__main__":
    train_all()