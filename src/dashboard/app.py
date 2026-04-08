"""
Streamlit Dashboard — Anomaly Detection Platform
Unified UI for Fraud and Intrusion Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
import shap
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ── Page Config ──
st.set_page_config(
    page_title="Anomaly Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
    }
    .risk-high {
        color: #dc2626;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .risk-medium {
        color: #d97706;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .risk-low {
        color: #2563eb;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .risk-minimal {
        color: #16a34a;
        font-weight: 700;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──

def check_api():
    """Check if FastAPI backend is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False


def get_risk_color(risk_level: str) -> str:
    colors = {
        "HIGH": "#dc2626",
        "MEDIUM": "#d97706",
        "LOW": "#2563eb",
        "MINIMAL": "#16a34a"
    }
    return colors.get(risk_level, "#6b7280")


def render_risk_badge(risk_level: str):
    color = get_risk_color(risk_level)
    st.markdown(
        f'<div style="display:inline-block; background:{color}; '
        f'color:white; padding:6px 18px; border-radius:20px; '
        f'font-weight:700; font-size:1rem;">{risk_level} RISK</div>',
        unsafe_allow_html=True
    )


def plot_score_gauge(score: float, title: str):
    """Simple horizontal bar as score gauge."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    color = '#dc2626' if score > 0.7 else '#d97706' if score > 0.4 else '#16a34a'
    ax.barh([0], [score], color=color, height=0.5)
    ax.barh([0], [1], color='#e5e7eb', height=0.5, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"{title}: {score:.3f}", fontsize=11)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.tight_layout()
    return fig


def generate_shap_plot(domain: str, features_df: pd.DataFrame):
    """Generate SHAP waterfall for a single prediction."""
    try:
        if domain == "fraud":
            model = joblib.load('models/fraud/xgboost_model.pkl')
        else:
            model = joblib.load('models/intrusion/xgboost_model.pkl')

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=features_df.iloc[0],
            feature_names=features_df.columns.tolist()
        )
        shap.waterfall_plot(shap_exp, show=False, max_display=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None


# ── Sidebar ──

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.markdown("## 🛡️ Anomaly Detection")
    st.markdown("---")

    domain = st.selectbox(
        "Select Detection Domain",
        ["Fraud Detection", "Intrusion Detection"],
        help="Choose which ML model to use"
    )

    st.markdown("---")
    st.markdown("### API Status")
    if check_api():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline — Run FastAPI first")
        st.code("uvicorn src.api.main:app --reload --port 8000")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Models Used:**
    - XGBoost (primary)
    - Isolation Forest (anomaly layer)
    - SHAP (explainability)

    **Datasets:**
    - Credit Card Fraud (Kaggle)
    - NSL-KDD (Intrusion)
    """)


# ── Main Header ──
st.markdown('<div class="main-header">🛡️ Anomaly Detection Platform</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Unified ML system for Financial Fraud '
    'and Network Intrusion Detection</div>',
    unsafe_allow_html=True
)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Prediction",
    "📁 Batch CSV Upload",
    "📊 Model Performance"
])


# ══════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════
with tab1:
    st.markdown(f"### {domain} — Single Record Prediction")
    st.markdown("Enter feature values manually or use the sample buttons below.")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📥 Load Normal Sample"):
            st.session_state['load_sample'] = 'normal'
        if st.button("🚨 Load Suspicious Sample"):
            st.session_state['load_sample'] = 'suspicious'

    if domain == "Fraud Detection":
        st.markdown("#### Transaction Features")
        st.caption("V1–V28 are PCA-transformed features. Amount is in USD.")

        load = st.session_state.get('load_sample', None)
        np.random.seed(42)

        if load == 'suspicious':
            default_v = list(np.random.randn(28))
            default_v[13] = -8.0
            default_v[3] = -6.0
            default_amount = 1.0
        else:
            default_v = list(np.random.randn(28))
            default_amount = 150.0

        cols = st.columns(4)
        v_values = []
        for i in range(28):
            with cols[i % 4]:
                v = st.number_input(
                    f"V{i+1}",
                    value=round(default_v[i], 4),
                    format="%.4f",
                    key=f"v{i+1}"
                )
                v_values.append(v)

        amount = st.number_input("Amount ($)",
                                  value=default_amount,
                                  min_value=0.0,
                                  format="%.2f")

        if st.button("🔍 Predict Fraud", type="primary"):
            if not check_api():
                st.error("API is not running. Start FastAPI first.")
            else:
                features = v_values + [amount]
                with st.spinner("Analyzing transaction..."):
                    r = requests.post(
                        f"{API_URL}/predict/fraud",
                        json={"features": features}
                    )
                    result = r.json()

                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Prediction", result['label'])
                with c2:
                    st.metric("Fraud Probability",
                              f"{result['fraud_probability']:.3f}")
                with c3:
                    st.metric("Combined Risk Score",
                              f"{result['combined_risk_score']:.3f}")
                with c4:
                    render_risk_badge(result['risk_level'])

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig = plot_score_gauge(
                        result['fraud_probability'], "Fraud Probability"
                    )
                    st.pyplot(fig)
                    plt.close()
                with col_g2:
                    fig = plot_score_gauge(
                        result['combined_risk_score'], "Combined Risk Score"
                    )
                    st.pyplot(fig)
                    plt.close()

                # SHAP explanation
                with st.expander("🔬 View SHAP Explanation"):
                    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
                    features_df = pd.DataFrame(
                        [features], columns=feature_names
                    )
                    scaler = joblib.load('models/fraud/fraud_scaler.pkl')
                    features_df['Amount'] = scaler.transform(
                        features_df[['Amount']]
                    )
                    with st.spinner("Generating SHAP explanation..."):
                        fig = generate_shap_plot("fraud", features_df)
                    if fig:
                        st.pyplot(fig)
                        plt.close()

    else:  # Intrusion Detection
        st.markdown("#### Network Traffic Features")
        st.caption("41 NSL-KDD features describing network connection.")

        load = st.session_state.get('load_sample', None)
        if load == 'suspicious':
            defaults = [0.0] * 41
            defaults[4] = 50000.0   # high src_bytes
            defaults[22] = 500.0    # high count
        else:
            defaults = [0.0] * 41
            defaults[0] = 1.0
            defaults[4] = 1000.0
            defaults[5] = 500.0

        from src.intrusion.preprocessor import COLUMNS
        feat_names = [c for c in COLUMNS
                     if c not in ['label', 'difficulty']]

        cols = st.columns(3)
        feat_values = []
        for i, fname in enumerate(feat_names):
            with cols[i % 3]:
                v = st.number_input(
                    fname,
                    value=float(defaults[i]),
                    format="%.2f",
                    key=f"intr_{fname}"
                )
                feat_values.append(v)

        if st.button("🔍 Predict Intrusion", type="primary"):
            if not check_api():
                st.error("API is not running.")
            else:
                with st.spinner("Analyzing network traffic..."):
                    r = requests.post(
                        f"{API_URL}/predict/intrusion",
                        json={"features": feat_values}
                    )
                    result = r.json()

                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Prediction", result['label'])
                with c2:
                    st.metric("Attack Probability",
                              f"{result['attack_probability']:.3f}")
                with c3:
                    render_risk_badge(result['risk_level'])

                fig = plot_score_gauge(
                    result['attack_probability'], "Attack Probability"
                )
                st.pyplot(fig)
                plt.close()


# ══════════════════════════════════════
# TAB 2 — Batch CSV Upload
# ══════════════════════════════════════
with tab2:
    st.markdown("### 📁 Batch Prediction from CSV")
    st.markdown("Upload a CSV file to get predictions for multiple records at once.")

    if domain == "Fraud Detection":
        st.info("CSV must have columns: V1–V28, Amount (no Class column needed)")
    else:
        st.info("CSV must have 41 NSL-KDD feature columns")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {len(df):,} rows × {len(df.columns)} columns")
        st.dataframe(df.head(5))

        if st.button("🚀 Run Batch Prediction", type="primary"):
            if not check_api():
                st.error("API is not running.")
            else:
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, row in df.iterrows():
                    try:
                        if domain == "Fraud Detection":
                            feat_cols = [f'V{j}' for j in range(1, 29)] + ['Amount']
                            features = row[feat_cols].tolist()
                            r = requests.post(
                                f"{API_URL}/predict/fraud",
                                json={"features": features}
                            )
                            res = r.json()
                            results.append({
                                'row': i,
                                'prediction': res['label'],
                                'fraud_probability': res['fraud_probability'],
                                'risk_level': res['risk_level'],
                                'combined_risk_score': res['combined_risk_score']
                            })
                        else:
                            from src.intrusion.preprocessor import COLUMNS
                            feat_names = [c for c in COLUMNS
                                         if c not in ['label', 'difficulty']]
                            features = row[feat_names].tolist()
                            r = requests.post(
                                f"{API_URL}/predict/intrusion",
                                json={"features": features}
                            )
                            res = r.json()
                            results.append({
                                'row': i,
                                'prediction': res['label'],
                                'attack_probability': res['attack_probability'],
                                'risk_level': res['risk_level']
                            })
                    except Exception as e:
                        results.append({'row': i, 'error': str(e)})

                    progress.progress((i + 1) / len(df))
                    status.text(f"Processing row {i+1} of {len(df)}...")

                results_df = pd.DataFrame(results)
                st.success(f"✅ Batch prediction complete — {len(results_df)} records processed")
                st.dataframe(results_df)

                # Summary
                if 'prediction' in results_df.columns:
                    st.markdown("### Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        pred_counts = results_df['prediction'].value_counts()
                        fig, ax = plt.subplots(figsize=(5, 4))
                        pred_counts.plot(kind='bar', ax=ax,
                                        color=['#16a34a', '#dc2626'])
                        ax.set_title("Prediction Distribution")
                        ax.set_xlabel("")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        risk_counts = results_df['risk_level'].value_counts()
                        fig, ax = plt.subplots(figsize=(5, 4))
                        colors = ['#dc2626', '#d97706', '#2563eb', '#16a34a']
                        risk_counts.plot(kind='pie', ax=ax,
                                        autopct='%1.1f%%',
                                        colors=colors[:len(risk_counts)])
                        ax.set_title("Risk Level Distribution")
                        ax.set_ylabel("")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )


# ══════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Performance Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔵 Fraud Detection")
        st.markdown("""
        | Model | ROC-AUC | Avg Precision |
        |-------|---------|---------------|
        | Logistic Regression | 0.9604 | — |
        | XGBoost | 0.9694 | 0.7899 |
        | XGB + Isolation Forest | 0.9582 | — |
        """)

        if os.path.exists('reports/figures/fraud_roc_curves.png'):
            st.image('reports/figures/fraud_roc_curves.png',
                     caption="Fraud ROC Curves")
        if os.path.exists('reports/figures/shap_importance_bar.png'):
            st.image('reports/figures/shap_importance_bar.png',
                     caption="Fraud SHAP Feature Importance")

    with col2:
        st.markdown("#### 🔴 Intrusion Detection")
        st.markdown("""
        | Model | ROC-AUC | Avg Precision |
        |-------|---------|---------------|
        | Random Forest | 0.9691 | 0.9699 |
        | XGBoost | 0.9696 | 0.9725 |
        """)

        if os.path.exists('reports/figures/intrusion_roc_curves.png'):
            st.image('reports/figures/intrusion_roc_curves.png',
                     caption="Intrusion ROC Curves")
        if os.path.exists('reports/figures/intrusion_shap_bar.png'):
            st.image('reports/figures/intrusion_shap_bar.png',
                     caption="Intrusion SHAP Feature Importance")

    st.markdown("---")
    st.markdown("#### 📁 All Generated Figures")
    figures_dir = 'reports/figures'
    if os.path.exists(figures_dir):
        figs = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        cols = st.columns(3)
        for i, fig_name in enumerate(sorted(figs)):
            with cols[i % 3]:
                st.image(
                    os.path.join(figures_dir, fig_name),
                    caption=fig_name.replace('_', ' ').replace('.png', ''),
                    width=None
                )