"""
SHAP Explainability for Intrusion Detection XGBoost Model.
"""

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.intrusion.preprocessor import preprocess_intrusion


def run_shap_analysis():
    """Generate SHAP explanations for intrusion detection model."""

    print("=== SHAP EXPLAINABILITY — INTRUSION DETECTION ===\n")

    # Load and preprocess
    X_train, X_test, y_train, y_test = preprocess_intrusion(
        'data/raw/intrusion/KDDTrain+.txt',
        'data/raw/intrusion/KDDTest+.txt'
    )[:4]

    # Load trained XGBoost
    model = joblib.load('models/intrusion/xgboost_model.pkl')
    print("XGBoost intrusion model loaded.")

    # Sample for speed
    X_sample = X_test.iloc[:500]

    print("Computing SHAP values (1-2 minutes)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print("SHAP values computed.")

    os.makedirs('reports/figures', exist_ok=True)

    # Plot 1: Bar importance
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title("SHAP Feature Importance — Intrusion Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig('reports/figures/intrusion_shap_bar.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: intrusion_shap_bar.png")

    # Plot 2: Dot summary
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        show=False,
        max_display=15
    )
    plt.title("SHAP Summary — Intrusion Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig('reports/figures/intrusion_shap_dot.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: intrusion_shap_dot.png")

    # Top features
    mean_shap = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(mean_shap.head(10).to_string(index=False))

    print("\n✅ Intrusion SHAP analysis complete.")
    return shap_values, X_sample


if __name__ == "__main__":
    run_shap_analysis()