"""
SHAP Explainability for Fraud Detection XGBoost Model.
Generates feature importance visualizations.
"""

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.fraud.preprocessor import load_fraud_data, preprocess_fraud


def run_shap_analysis():
    """Load trained XGBoost model and generate SHAP explanations."""

    print("=== SHAP EXPLAINABILITY ANALYSIS ===\n")

    # Step 1: Load and preprocess data
    df = load_fraud_data('data/raw/fraud/creditcard.csv')
    X_train, X_test, y_train, y_test, _ = preprocess_fraud(df)

    # Step 2: Load trained XGBoost model
    model = joblib.load('models/fraud/xgboost_model.pkl')
    print("\nXGBoost model loaded successfully.")

    # Step 3: Create SHAP explainer
    print("Creating SHAP explainer (this may take 1-2 minutes)...")
    explainer = shap.TreeExplainer(model)

    # Use a sample of test data for speed (500 rows is enough)
    X_sample = X_test.iloc[:500]
    shap_values = explainer.shap_values(X_sample)
    print("SHAP values computed.")

    os.makedirs('reports/figures', exist_ok=True)

    # Plot 1: Summary Bar Plot (global feature importance)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title("SHAP Feature Importance (Mean |SHAP value|)", fontsize=13)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_importance_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_importance_bar.png")

    # Plot 2: Summary Dot Plot (direction + magnitude)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        show=False,
        max_display=15
    )
    plt.title("SHAP Summary Plot", fontsize=13)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary_dot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary_dot.png")

    # Plot 3: Waterfall plot for a single fraud case
    fraud_indices = np.where(y_test.values == 1)[0]
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        if fraud_idx < 500:
            print(f"\nGenerating waterfall plot for fraud case at index {fraud_idx}...")
            shap_exp = shap.Explanation(
                values=shap_values[fraud_idx],
                base_values=explainer.expected_value,
                data=X_sample.iloc[fraud_idx],
                feature_names=X_sample.columns.tolist()
            )
            plt.figure()
            shap.waterfall_plot(shap_exp, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig('reports/figures/shap_waterfall_fraud.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved: shap_waterfall_fraud.png")

    # Print top features
    mean_shap = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(mean_shap.head(10).to_string(index=False))

    print("\n✅ SHAP analysis complete. Plots saved to reports/figures/")
    return shap_values, X_sample


if __name__ == "__main__":
    run_shap_analysis()