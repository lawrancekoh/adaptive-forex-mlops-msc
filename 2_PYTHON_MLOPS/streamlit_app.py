import streamlit as st
import pandas as pd
import json
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.retraining_script import run_wfa_simulation
except ImportError:
    # Fallback if running directly from root
    from retraining_script import run_wfa_simulation

# Page Config
st.set_page_config(
    page_title="Adaptive Forex MLOps Dashboard",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("🤖 Adaptive Forex MLOps: Validation Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Control Panel")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Market Data (CSV)", type=['csv'], help="Export M15 Bars from MT5")
with st.sidebar.expander("Data Export Instructions"):
    st.markdown("""
    1. Open MetaTrader 5
    2. Go to **View -> Symbols** (Ctrl+U)
    3. Select **EURUSD** -> **Bars**
    4. Set Timeframe to **M15**
    5. Request history (e.g. 2020-2024)
    6. Click **Export Bars** (CSV)
    """)

# Simulation Controls
st.sidebar.subheader("Simulation Triggers")

if st.sidebar.button("Train Single Model (Quick)"):
    if uploaded_file is not None:
        with st.spinner("Processing Data & Training GMM..."):
            # Pass the uploaded file object specifically
            result = run_wfa_simulation(data_source=uploaded_file, mode='single_train')
            
            if result['status'] == 'success':
                st.success(result.get('message', "Training Complete"))
                
                # Display Results
                st.subheader("📊 Market Regimes (GMM Clusters)")
                
                centroids = result.get('centroids', [])
                trade_params = result.get('trade_params', {})
                
                if centroids:
                    # Convert to DataFrame for nice display
                    df_clusters = pd.DataFrame(centroids)
                    # Reorder columns
                    cols = ['cluster_id', 'hurst', 'atr', 'adx']
                    available_cols = [c for c in cols if c in df_clusters.columns]
                    df_clusters = df_clusters[available_cols]
                    
                    # Add Labels from trade params map
                    labels = []
                    rationales = []
                    for cid in df_clusters['cluster_id']:
                         p = trade_params.get(str(int(cid)), {})
                         labels.append(p.get('regime_label', 'Unknown'))
                         rationales.append(p.get('rationale', ''))
                    
                    df_clusters['Label'] = labels
                    df_clusters['Rationale'] = rationales
                    
                    st.dataframe(df_clusters.style.format({
                        'hurst': '{:.4f}',
                        'atr': '{:.6f}',
                        'adx': '{:.2f}'
                    }), use_container_width=True)
                    
                    st.markdown("##### Feature Distribution")
                    st.json(trade_params, expanded=False)
                    
            else:
                st.error(f"Training Failed: {result.get('message')}")
    else:
        st.warning("Please upload a CSV file first.")

if st.sidebar.button("Run Full WFA Loop (Slow)"):
    if uploaded_file is not None:
        with st.spinner("Running Walk-Forward Analysis... This may take several minutes."):
            uploaded_file.seek(0)
            result = run_wfa_simulation(data_source=uploaded_file, mode='full')
            
            if result['status'] == 'success':
                st.success(result.get('message', "WFA Complete"))
                
                # Display Aggregate Stats
                st.subheader("📈 WFA Aggregate Results")
                agg = result.get('aggregate', {})
                
                if agg:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Iterations", agg.get('total_iterations', 0))
                    col2.metric("Avg. Stability", f"{agg.get('avg_stability_ratio', 0):.2%}")
                    col3.metric("Avg. Gen. Gap", f"{agg.get('avg_generalization_gap', 0):.4f}")
                    
                    st.markdown("##### Regime Frequency (Dominant per OOS period)")
                    regime_freq = agg.get('regime_frequency', {})
                    df_freq = pd.DataFrame([
                        {'Regime': f"R{k}", 'Count': v} 
                        for k, v in regime_freq.items()
                    ])
                    st.bar_chart(df_freq.set_index('Regime'))
                
                # Display Cluster Centroids (final model)
                st.subheader("📊 Final Model Clusters")
                centroids = result.get('centroids', [])
                trade_params = result.get('trade_params', {})
                
                if centroids:
                    df_clusters = pd.DataFrame(centroids)
                    cols = ['cluster_id', 'hurst', 'atr', 'adx']
                    available_cols = [c for c in cols if c in df_clusters.columns]
                    df_clusters = df_clusters[available_cols]
                    
                    labels = []
                    for cid in df_clusters['cluster_id']:
                         p = trade_params.get(str(int(cid)), {})
                         labels.append(p.get('regime_label', 'Unknown'))
                    
                    df_clusters['Label'] = labels
                    st.dataframe(df_clusters.style.format({
                        'hurst': '{:.4f}',
                        'atr': '{:.6f}',
                        'adx': '{:.2f}'
                    }), use_container_width=True)
                
                # Show path to full results
                st.info(f"Full WFA metrics saved to: `{result.get('wfa_metrics_path', 'N/A')}`")
                
            else:
                st.error(f"WFA Failed: {result.get('message')}")
    else:
        st.warning("Please upload a CSV file first.")

# Data Preview
if uploaded_file is not None:
    st.markdown("### Data Preview")
    uploaded_file.seek(0)
    try:
        # Just peak at first few lines raw to avoid messing up the pointer for the script if handled poorly
        # Try same aggressive list as data_loader
        try:
             df_preview = pd.read_csv(uploaded_file, sep='\t', encoding='utf-16', nrows=5)
        except:
             uploaded_file.seek(0)
             df_preview = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', nrows=5)
             
        st.dataframe(df_preview)
    except Exception as e:
        uploaded_file.seek(0)
        st.error(f"Preview unavailable. Error reading file: {e}")
        st.caption("Common issues: File is open in Excel, different encoding, or corrupted.")

st.markdown("---")
st.caption("MSc Thesis Project - Adaptive Algorithmic Trading System")
