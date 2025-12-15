import streamlit as st
import sys
import os
import json
import pandas as pd

# Add src to python path to allow importing retraining_script
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from retraining_script import run_wfa_simulation

st.set_page_config(page_title="FXATM MLOps Dashboard", layout="wide")

st.title("🤖 Intelligent Forex Trading: MLOps Dashboard")
st.markdown("### Adaptive Machine Learning Framework (Thesis Simulation)")

# --- Sidebar Controls ---
st.sidebar.header("MLOps Pipeline Controls")

start_full_wfa = st.sidebar.button("🚀 RUN FULL WFA SIMULATION (Chapter 4 Data)")
start_single_train = st.sidebar.button("⚙️ Train Single Model (Centroid Update)")

# --- Main Dashboard Area ---

# 1. Pipeline Execution Status
if start_full_wfa:
    with st.spinner("Executing Full Walk-Forward Analysis (WFA) and saving results..."):
        results = run_wfa_simulation(mode='full')
        st.success(results.get('message', 'Done'))
        
        st.subheader("WFA Results Summary")
        if 'wfa_metrics' in results:
            col1, col2 = st.columns(2)
            col1.metric("Avg Sharpe Ratio", results['wfa_metrics']['sharpe_ratio_avg'])
            col2.metric("Recovery Factor", results['wfa_metrics']['recovery_factor'])

if start_single_train:
    with st.spinner("Training GMM on the latest data window and updating centroids..."):
        results = run_wfa_simulation(mode='single_train')
        st.success(results.get('message', 'Done'))
        
        st.subheader("New GMM Cluster Centroids")
        if 'centroids' in results:
            df = pd.DataFrame(results['centroids'])
            st.dataframe(df)

# 2. Existing Artifact Visualization (Placeholder for now)
st.markdown("---")
st.subheader("Current Production Model State")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config/trade_params.json')
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as f:
            trade_params = json.load(f)
        
        st.markdown("**Active CPO Logic Table (Regime -> Parameters)**")
        st.json(trade_params)
    except Exception as e:
        st.error(f"Error reading trade params: {e}")
else:
    st.warning("No active trade parameters found (Run simulation to generate).")
