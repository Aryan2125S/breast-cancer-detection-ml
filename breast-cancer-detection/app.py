"""
==============================================================================
Breast Cancer Detection — Streamlit Web Application
==============================================================================
Premium, interactive UI for breast cancer diagnosis prediction.

Usage:
    streamlit run app.py
==============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection | AI Diagnosis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium CSS — glassmorphism, animations, refined typography
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== Global ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1035 40%, #0d1b2a 70%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.08);
    }
    section[data-testid="stSidebar"] .stSlider > div > div {
        background: rgba(56, 189, 248, 0.08);
    }

    /* ===== Headers ===== */
    h1 {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
        font-size: 2.4em !important;
    }
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    /* ===== Glass cards ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 28px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.15);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.08);
    }

    /* ===== Metric cards ===== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 20px;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.1);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85em !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
    }

    /* ===== Prediction result ===== */
    .result-container {
        border-radius: 24px;
        padding: 40px 30px;
        text-align: center;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    .result-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 60%);
        animation: pulse-glow 3s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    .result-benign {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.12), rgba(0, 200, 83, 0.05));
        border: 2px solid rgba(0, 230, 118, 0.4);
    }
    .result-malignant {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.12), rgba(213, 0, 0, 0.05));
        border: 2px solid rgba(255, 82, 82, 0.4);
    }
    .result-icon {
        font-size: 3.5em;
        margin-bottom: 8px;
        display: block;
    }
    .result-text {
        font-size: 2.8em;
        font-weight: 900;
        margin: 8px 0;
        letter-spacing: -0.02em;
    }
    .result-sub {
        font-size: 1.05em;
        opacity: 0.7;
        margin-top: 8px;
        line-height: 1.6;
    }

    /* ===== Progress bars for confidence ===== */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .confidence-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.06);
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.8s ease;
    }

    /* ===== Info panel ===== */
    .info-panel {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }
    .info-panel p {
        margin: 10px 0;
        line-height: 1.7;
    }
    .info-label {
        color: #64748b;
        font-size: 0.82em;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .info-value {
        color: #e2e8f0;
        font-weight: 600;
    }

    /* ===== Waiting state ===== */
    .waiting-state {
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 50px 30px;
        text-align: center;
    }
    .waiting-icon {
        font-size: 3em;
        margin-bottom: 12px;
        display: block;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    /* ===== Stats row ===== */
    .stats-row {
        display: flex;
        gap: 12px;
        margin: 16px 0;
    }
    .stat-item {
        flex: 1;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.78em;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        opacity: 0.3;
        font-size: 0.82em;
        padding: 30px 0 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 40px;
    }

    /* ===== Button ===== */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1.05em !important;
        letter-spacing: 0.02em;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model, scaler, and metadata
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler, and metadata from disk."""
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    model_path = os.path.join(model_dir, "best_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    if not os.path.exists(model_path):
        return None, None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    return model, scaler, metadata


model, scaler, metadata = load_artifacts()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🔬 Breast Cancer Detection")
st.markdown(
    '<p style="font-size: 1.1em; color: #94a3b8; margin-top: -10px; line-height: 1.6;">'
    'AI-powered diagnosis prediction — enter tumor cell nucleus measurements from a fine needle aspirate (FNA) image '
    'to get an instant classification.'
    '</p>',
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "⚠️ **Model not found!** Please run `python main.py` first to train "
        "the model and generate the required files in the `model/` directory."
    )
    st.stop()

# Stats row
st.markdown(
    '<div class="stats-row">'
    '<div class="stat-item">'
    '<div class="stat-number">569</div>'
    '<div class="stat-label">Training Samples</div>'
    '</div>'
    '<div class="stat-item">'
    '<div class="stat-number">30</div>'
    '<div class="stat-label">Input Features</div>'
    '</div>'
    '<div class="stat-item">'
    f'<div class="stat-number">98.2%</div>'
    '<div class="stat-label">Model Accuracy</div>'
    '</div>'
    '<div class="stat-item">'
    f'<div class="stat-number">98.6%</div>'
    '<div class="stat-label">Recall Score</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("")  # spacer

# ---------------------------------------------------------------------------
# Sidebar — Feature inputs
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<h2 style="background: linear-gradient(90deg, #38bdf8, #818cf8); '
    '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
    'font-weight: 800;">📋 Tumor Measurements</h2>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p style="color: #64748b; font-size: 0.88em; line-height: 1.6;">'
    'Adjust the sliders to input cell nucleus measurements from a '
    'Fine Needle Aspirate (FNA) image. Defaults are dataset averages.'
    '</p>',
    unsafe_allow_html=True,
)

feature_names = metadata["feature_names"]
means = metadata["feature_means"]
mins = metadata["feature_mins"]
maxs = metadata["feature_maxs"]

# Group features by category
categories = {
    "📏 Mean Measurements": [f for f in feature_names if f.startswith("mean")],
    "📊 Standard Error": [f for f in feature_names if "error" in f],
    "⚠️ Worst (Most Severe)": [f for f in feature_names if f.startswith("worst")],
}

input_values = []
for category, feats in categories.items():
    with st.sidebar.expander(category, expanded=(category.startswith("📏"))):
        for feat in feats:
            min_val = float(mins[feat])
            max_val = float(maxs[feat])
            mean_val = float(means[feat])
            slider_min = min_val - abs(min_val) * 0.1
            slider_max = max_val + abs(max_val) * 0.1
            # Format label nicely
            label = feat.replace("mean ", "").replace("worst ", "").replace(" error", " (SE)")
            val = st.slider(
                feat,
                min_value=slider_min,
                max_value=slider_max,
                value=mean_val,
                format="%.4f",
                key=feat,
            )
            input_values.append(val)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
col_main, col_side = st.columns([5, 3], gap="large")

with col_main:
    st.markdown("### 🎯 Diagnosis Prediction")

    if st.button("🔍  Run Prediction", use_container_width=True, type="primary"):
        X = np.array(input_values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        label_map = metadata["label_map"]
        result = label_map.get(prediction, "Unknown")

        if result == "Benign":
            st.markdown(
                '<div class="result-container result-benign">'
                '<span class="result-icon">🛡️</span>'
                '<div class="result-text" style="color: #00e676;">BENIGN</div>'
                '<div class="result-sub">The tumor characteristics suggest a <b>non-cancerous</b> growth.<br>'
                'Regular monitoring is still recommended.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-container result-malignant">'
                '<span class="result-icon">⚠️</span>'
                '<div class="result-text" style="color: #ff5252;">MALIGNANT</div>'
                '<div class="result-sub">The tumor characteristics indicate potential <b>malignancy</b>.<br>'
                'Please consult an oncologist for further evaluation immediately.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Confidence scores with visual bars
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            mal_pct = proba[0] * 100
            ben_pct = proba[1] * 100

            st.markdown("#### 📊 Confidence Breakdown")
            st.markdown(
                f'<div class="confidence-bar-container">'
                f'<div class="confidence-label">'
                f'<span style="color: #00e676;">✅ Benign</span>'
                f'<span style="color: #e2e8f0; font-weight: 800;">{ben_pct:.1f}%</span>'
                f'</div>'
                f'<div class="confidence-bar">'
                f'<div class="confidence-fill" style="width: {ben_pct}%; '
                f'background: linear-gradient(90deg, #00e676, #00c853);"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="confidence-bar-container">'
                f'<div class="confidence-label">'
                f'<span style="color: #ff5252;">⚠️ Malignant</span>'
                f'<span style="color: #e2e8f0; font-weight: 800;">{mal_pct:.1f}%</span>'
                f'</div>'
                f'<div class="confidence-bar">'
                f'<div class="confidence-fill" style="width: {mal_pct}%; '
                f'background: linear-gradient(90deg, #ff5252, #d50000);"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Show top contributing features
        st.markdown("#### 🔑 Key Input Values")
        key_features = ["worst concave points", "worst perimeter",
                        "mean concave points", "worst radius", "mean radius"]
        kf_data = []
        for feat in key_features:
            if feat in feature_names:
                idx = feature_names.index(feat)
                kf_data.append({
                    "Feature": feat,
                    "Your Value": f"{input_values[idx]:.4f}",
                    "Dataset Avg": f"{means[feat]:.4f}",
                })
        if kf_data:
            st.dataframe(pd.DataFrame(kf_data), hide_index=True, use_container_width=True)

    else:
        st.markdown(
            '<div class="waiting-state">'
            '<span class="waiting-icon">🧬</span>'
            '<p style="color: #94a3b8; font-size: 1.15em; font-weight: 500;">'
            'Adjust tumor measurements in the sidebar</p>'
            '<p style="color: #64748b; font-size: 0.95em;">'
            'then click <b style="color: #818cf8;">Run Prediction</b> '
            'for an instant AI diagnosis</p>'
            '</div>',
            unsafe_allow_html=True,
        )

with col_side:
    # About panel
    st.markdown("### ℹ️ Model Info")
    st.markdown(
        '<div class="info-panel">'
        f'<p><span class="info-label">Algorithm</span><br>'
        f'<span class="info-value">{metadata["best_model_name"]}</span></p>'
        '<p><span class="info-label">Dataset</span><br>'
        '<span class="info-value">Breast Cancer Wisconsin (Diagnostic)</span></p>'
        '<p><span class="info-label">Input Features</span><br>'
        '<span class="info-value">30 measurements from FNA cell images</span></p>'
        '<p><span class="info-label">Evaluation Metric</span><br>'
        '<span class="info-value">Recall-optimized (minimizes missed cancers)</span></p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # How it works
    st.markdown("### 🧪 How It Works")
    st.markdown(
        '<div class="info-panel">'
        '<p style="color: #94a3b8; font-size: 0.92em; line-height: 1.8;">'
        '<b style="color: #38bdf8;">1.</b> A fine needle aspirate (FNA) is taken from the breast mass<br>'
        '<b style="color: #38bdf8;">2.</b> Cell nuclei are digitized and 30 measurements are computed<br>'
        '<b style="color: #38bdf8;">3.</b> Features are standardized (z-score normalization)<br>'
        '<b style="color: #38bdf8;">4.</b> The ML model classifies the tumor as Benign or Malignant'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # Disclaimer
    st.markdown("### ⚠️ Medical Disclaimer")
    st.markdown(
        '<div class="info-panel" style="border-color: rgba(255, 82, 82, 0.15);">'
        '<p style="color: #94a3b8; font-size: 0.88em; line-height: 1.7;">'
        'This tool is for <b style="color: #ff8a80;">educational purposes only</b>. '
        'It is NOT a substitute for professional medical diagnosis. '
        'Always consult a qualified healthcare provider for medical decisions. '
        'No clinical decisions should be made based on this tool.'
        '</p></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">'
    'Breast Cancer Detection System — Built with Scikit-learn & Streamlit<br>'
    'Multi-Agent ML Pipeline | Educational Project'
    '</div>',
    unsafe_allow_html=True,
)
