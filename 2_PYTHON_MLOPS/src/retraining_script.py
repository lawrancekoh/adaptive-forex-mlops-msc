import time
import os
import yaml
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Import local modules
# Assuming src is in python path or relative import
try:
    from src.data_loader import load_data
    from src.feature_engineering import build_feature_matrix
except ImportError:
    from data_loader import load_data
    from feature_engineering import build_feature_matrix

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '../config/config.yaml')
ARTIFACTS_DIR = os.path.join(BASE_DIR, '../../3_ML_ARTIFACTS')
TRADE_PARAMS_PATH = os.path.join(BASE_DIR, '../config/trade_params.json')

def train_gmm(df, n_components=4):
    """
    Trains GMM on the feature matrix.
    Args:
        df (pd.DataFrame): Dataframe with features [hurst, volatility_atr, trend_adx].
    Returns:
        model: Trained GMM model
        scaler: Fitted scaler (if used - logically we might want to scale)
    """
    X = df[['hurst', 'volatility_atr', 'trend_adx']].values
    
    # Check for inf/nan
    if np.any(np.isinf(X)) or np.any(np.isnan(X)):
        print("Warning: Input contains NaNs or Infs. Cleaning...")
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        X = df_clean[['hurst', 'volatility_atr', 'trend_adx']].values
    
    # Note: GMM works better with scaling, but Thesis 3.4.1 doesn't explicitly mandate it.
    # However, Hurst (0-1), ATR (small), ADX (0-100) have vastly different scales.
    # Standard Scaling is recommended for GMM convergence.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    
    return gmm, scaler

def derive_cpo_params(gmm, scaler, feature_cols=['hurst', 'volatility_atr', 'trend_adx']):
    """
    Maps Mixture Components to Economic Regimes and Parameters.
    Logic: Sort clusters by Mean Hurst Exponent.
    Low Hurst -> Mean Reverting -> Aggressive Params
    High Hurst -> Trending -> Defensive Params
    """
    # Get centroids (means) in original scale
    means_scaled = gmm.means_
    means_original = scaler.inverse_transform(means_scaled)
    
    centroids = []
    for i in range(gmm.n_components):
        c = {
            'cluster_id': i,
            'hurst': means_original[i][0],
            'atr': means_original[i][1],
            'adx': means_original[i][2]
        }
        centroids.append(c)
        
    # Sort by Hurst (Low to High)
    # 0 = Lowest Hurst (Strongest Mean Reversion)
    # 3 = Highest Hurst (Strongest Trend)
    centroids_sorted = sorted(centroids, key=lambda x: x['hurst'])
    
    # Define CPO Map (Hardcoded Logic from Thesis Table 1/Section 3.5.1)
    # Refined Mapping:
    # Rank 0 (Lowest H): Ranging/Safe -> Aggressive Grid (Dist=1.2, Lot=1.5)
    # Rank 1: Weak Trend/Volatile -> Moderate (Dist=1.5, Lot=1.3)
    # Rank 2: Trending -> Defensive (Dist=2.0, Lot=1.2)
    # Rank 3 (Highest H): Strong Trend -> Max Defense (Dist=2.5, Lot=1.1)
    
    cpo_map = {}
    
    # Strict mapping based on Rank
    mappings = [
        {"label": "Ranging (Safe)", "dist": 1.2, "lot": 1.5, "rationale": "Mean Reverting (Low Hurst)"},
        {"label": "Choppy / Weak Trend", "dist": 1.5, "lot": 1.3, "rationale": "Moderate Hurst"},
        {"label": "Trending", "dist": 2.0, "lot": 1.2, "rationale": "High Hurst"},
        {"label": "Strong Trend / Breakout", "dist": 2.5, "lot": 1.1, "rationale": "Extreme Hurst (Persistence)"}
    ]
    
    for rank, cluster_obj in enumerate(centroids_sorted):
        original_id = cluster_obj['cluster_id']
        param_set = mappings[rank]
        
        cpo_map[str(original_id)] = {
            "regime_label": param_set['label'],
            "distance_multiplier": param_set['dist'],
            "lot_multiplier": param_set['lot'],
            "rationale": f"{param_set['rationale']} (H={cluster_obj['hurst']:.2f}, ADX={cluster_obj['adx']:.1f})",
            "feature_means": cluster_obj
        }
        
    return cpo_map, centroids

def run_wfa_simulation(data_source=None, mode='single_train'):
    """
    Orchestrates the training/simulation process.
    Args:
        data_source: File path or UploadedFile.
        mode (str): 'single_train' or 'full'.
    """
    timestamp = int(time.time())
    result = {"status": "success", "mode": mode, "timestamp": timestamp}
    
    # 1. Load Data
    if data_source:
        df = load_data(data_source)
    else:
        # Fallback for testing without upload - try local file
        test_path = os.path.join(BASE_DIR, '../data/EURUSD_M15.csv')
        if os.path.exists(test_path):
            df = load_data(test_path)
        else:
            return {"status": "error", "message": "No data provided and no local file found."}
            
    if df.empty:
        return {"status": "error", "message": "Loaded dataframe is empty."}

    # 2. Feature Engineering
    try:
        df_features = build_feature_matrix(df)
        if df_features.empty:
             return {"status": "error", "message": "Feature matrix is empty (not enough data?)."}
    except Exception as e:
        return {"status": "error", "message": f"Feature Engineering failed: {e}"}

    if mode == 'single_train':
        print("Training GMM on latest data...")
        
        # Train
        try:
            model, scaler = train_gmm(df_features, n_components=4)
        except Exception as e:
            return {"status": "error", "message": f"GMM Training failed: {e}"}
            
        # CPO Derivation
        trade_params, centroids = derive_cpo_params(model, scaler)
        
        # Save Artifacts
        if not os.path.exists(ARTIFACTS_DIR):
            os.makedirs(ARTIFACTS_DIR)
            
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'gmm_model.pkl'))
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
        
        with open(TRADE_PARAMS_PATH, 'w') as f:
            json.dump(trade_params, f, indent=4)
            
        # Also update the config/trade_params for the inference server to pick up
        result['message'] = "Model trained. Artifacts saved. CPO Params updated."
        result['centroids'] = centroids
        result['trade_params'] = trade_params

    elif mode == 'full':
        # Placeholder for full WFA loop logic
        result['message'] = "Full WFA Simulation not fully implemented yet."
        
    return result

if __name__ == "__main__":
    # Test
    res = run_wfa_simulation()
    print(json.dumps(res, indent=2, default=str))
