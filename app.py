import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# ==========================================================
# 🌌 STUNNING DEEP DARK GLOW THEME CONFIGURATION (Strict Force)
# ==========================================================
st.set_page_config(
    page_title="Informatics Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injector to force dark backgrounds, glowing vibrant buttons, and bright headers
st.markdown("""
    <style>
        /* Force dark background for the entire app */
        .stApp {
            background: linear-gradient(135deg, #0d0f12 0%, #151922 100%) !important;
            color: #e2e8f0 !important;
        }
        
        /* Force Sidebar Theme */
        section[data-testid="stSidebar"] {
            background-color: #0b0c10 !important;
            border-right: 1px solid #1f293d;
        }
        
        /* 🔥 FIX: Force all sidebar labels, paragraph texts, and widget headers to be brightly visible */
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
            color: #e2e8f0 !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        
        /* Fixed Neon Blue/Purple Glowing Header Cards */
        .main-header-card {
            background: linear-gradient(90deg, #1e1b4b 0%, #311042 100%) !important;
            padding: 25px !important;
            border-radius: 12px !important;
            border-left: 6px solid #6366f1 !important;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25) !important;
            margin-top: 10px !important;
            margin-bottom: 25px !important;
            display: block !important;
        }
        
        /* Bright High-Contrast Title Text */
        .main-title {
            color: #ffffff !important;
            font-size: 30px !important;
            font-weight: 800 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Metric Card Stylings */
        .metric-card {
            background: #11141a !important;
            padding: 20px !important;
            border-radius: 10px !important;
            border: 1px solid #1f293d !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
            text-align: center !important;
        }
        
        /* Input section header titles */
        h3 {
            color: #38bdf8 !important;
            font-weight: 700 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 📥 PIPELINE CORE INITIALIZATION
# ==========================================================
@st.cache_resource
def load_production_pipeline():
    try:
        return joblib.load('model_pipeline.joblib')
    except:
        return None

pipeline = load_production_pipeline()

# ==========================================================
# 🏛️ INTERACTIVE DASHBOARD HEADER LAYOUT
# ==========================================================
st.markdown("""
    <div class="main-header-card">
        <h1 class="main-title">🧠 Behavioral Informatics & Risk Prediction Dashboard</h1>
        <p style="color: #cbd5e1 !important; font-size: 15px !important; margin-top: 8px !important; margin-bottom: 0 !important;">
            An advanced, end-to-end MLOps pipeline evaluating user lifestyle habits using LightGBM and K-Means Personas.
        </p>
    </div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ 'model_pipeline.joblib' file not found in directory. Please run your final notebook cell and push the saved binary file to this repository.")
    st.stop()

# ==========================================================
# 🎛️ SIDEBAR CONTROL PANEL: LIVE SIMULATOR INPUTS
# ==========================================================
st.sidebar.markdown("<h2 style='color: #6366f1; font-weight:800; margin-bottom: 0;'>🛠️ Configuration</h2>", unsafe_allow_html=True)
st.sidebar.write("Adjust variables below to simulate real-time behavioral profiles:")

# High-Vibrancy Numerical Feature Inputs
sleep_hours = st.sidebar.slider("💤 Sleep Duration (Hours)", 3.0, 10.0, 6.5, 0.5)
social_media_hours = st.sidebar.slider("📱 Daily Social Media (Hours)", 0.0, 12.0, 4.5, 0.5)
screen_pre_sleep = st.sidebar.slider("🌙 Pre-Bed Screen Time (Hours)", 0.0, 4.0, 1.5, 0.5)

st.sidebar.markdown("---")

# Categorical Feature Inputs
gender = st.sidebar.selectbox("👤 Gender Identity", ["male", "female"])
platform = st.sidebar.selectbox("🎯 Primary Platform Used", ["Instagram", "TikTok", "Both", "YouTube", "Twitter"])
interaction = st.sidebar.selectbox("🤝 Social Interaction Level", ["medium", "high", "low"])

# Psychological Scale Fields
stress = st.sidebar.slider("🔥 Self-Reported Stress Score", 1, 10, 5)
anxiety = st.sidebar.slider("⚡ Self-Reported Anxiety Score", 1, 10, 5)
addiction = st.sidebar.slider("⛓️ Device Addiction Rating", 1, 10, 4)
academic = st.sidebar.slider("📚 Academic Grade Point Average", 1.0, 4.0, 3.0, 0.1)
activity = st.sidebar.slider("🏃 Weekly Physical Activity (Hours)", 0.0, 7.0, 2.5, 0.5)
age = st.sidebar.number_input("🎂 User Age Profile", min_value=12, max_value=25, value=16)

# ==========================================================
# 🧮 BACKGROUND FEATURE INFERENCE GENERATION
# ==========================================================
if social_media_hours >= 7.0 or screen_pre_sleep >= 2.5:
    inferred_segment = "2"   
elif sleep_hours >= 7.5 and social_media_hours <= 3.0:
    inferred_segment = "1"   
else:
    inferred_segment = "0"   

input_row = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'daily_social_media_hours': social_media_hours,
    'platform_usage': platform,
    'sleep_hours': sleep_hours,
    'screen_time_before_sleep': screen_pre_sleep,
    'academic_performance': academic,
    'physical_activity': activity,
    'social_interaction_level': interaction,
    'stress_level': stress,
    'anxiety_level': anxiety,
    'addiction_level': addiction,
    'User_Segment': inferred_segment
}])

raw_prediction = pipeline.predict(input_row)[0]
raw_probabilities = pipeline.predict_proba(input_row)[0]

# ==========================================================
# 📊 CENTRAL DATA DASHBOARD DISPLAY
# ==========================================================
col_metrics, col_graph = st.columns([1, 2], gap="large")

with col_metrics:
    st.markdown("### 📊 Inference Engine Outputs")
    
    if raw_prediction == 1:
        st.markdown(f"""
            <div style="background-color: #7f1d1d; border: 2px solid #ef4444; padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);">
                <h2 style="color: #fca5a5 !important; margin: 0; font-size: 24px;">🚨 HIGH RISK CONDITION</h2>
                <p style="color: #fca5a5; font-size: 15px; margin-top: 8px; margin-bottom: 0;">The classifier identifies significant behavioral anomalies indicative of a clinical depressive state.</p>
                <div style="font-size: 40px; font-weight: 800; color: #ffffff; margin-top: 10px;">{(raw_probabilities[1]*100):.1f}% Confidence</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: #064e3b; border: 2px solid #10b981; padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);">
                <h2 style="color: #a7f3d0 !important; margin: 0; font-size: 24px;">✅ LOW RISK STABLE</h2>
                <p style="color: #a7f3d0; font-size: 15px; margin-top: 8px; margin-bottom: 0;">Lifestyle habits and psychological flags indicate balanced structural patterns.</p>
                <div style="font-size: 40px; font-weight: 800; color: #ffffff; margin-top: 10px;">{(raw_probabilities[0]*100):.1f}% Confidence</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    persona_names = {
        "1": "🟦 Balanced Routine Archetype",
        "2": "🟥 High-Exposure Screen Dependent",
        "0": "🟨 Standard Mixed Baseline"
    }
    st.markdown(f"""
        <div class="metric-card">
            <span style="color: #94a3b8; font-size: 13px; text-transform: uppercase; font-weight:600;">Inferred Unsupervised Persona</span>
            <h4 style="margin: 5px 0 0 0; color: #ffffff; font-size: 18px;">{persona_names[inferred_segment]}</h4>
        </div>
    """, unsafe_allow_html=True)

with col_graph:
    st.markdown("### 🎨 Real-Time Habit Proximity Visualization")
    
    radar_data = pd.DataFrame({
        'Metrics': ['Sleep Hours', 'Social Media Hours', 'Screen Before Bed', 'Stress Level', 'Anxiety Level'],
        'Value': [sleep_hours, social_media_hours, screen_pre_sleep * 2.5, stress, anxiety]
    })
    
    fig = px.line_polar(
        radar_data, 
        r='Value', 
        theta='Metrics', 
        line_close=True,
        template="plotly_dark"
    )
    fig.update_traces(fill='toself', fillcolor='rgba(99, 102, 241, 0.3)', line_color='#6366f1', line_width=3)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#1f293d", range=[0, 10]),
            angularaxis=dict(gridcolor="#1f293d")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 📊 LOWER ANALYSIS ROW: PIPELINE METRICS ACCORDION
# ==========================================================
st.markdown("---")
with st.expander("📊 View Synthetic Dataset Retrospective & Design Principles"):
    st.markdown("""
    * **The Pipeline Strategy:** This dashboard runs on a synchronized, multi-step pipeline structure. Inputs are captured dynamically, passed to mathematical custom logic handlers, and split into numerical standard deviations and categorical arrays completely on the fly without manual processing script bottlenecks.
    * **Synthetic Data Warning:** As thoroughly mapped during initial exploratory validation steps, this dashboard interfaces with an AI-generated dataset pattern structure containing tightly coupled closed-loop feature parameters. This explains why the validation matrices achieve near-absolute predictive metrics, turning this workspace into an educational blueprint showing production-level interface engineering rather than an operational clinical medical engine.
    """)
