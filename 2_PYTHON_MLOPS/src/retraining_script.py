import time
import os
import yaml
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

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
WFA_METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'wfa_metrics.json')

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


def run_wfa_full(df_features, lookback_months=6, step_weeks=1, n_components=4):
    """
    Runs the full Walk-Forward Analysis simulation.
    
    Args:
        df_features (pd.DataFrame): Feature matrix with datetime index.
        lookback_months (int): IS window size in months.
        step_weeks (int): OOS window size / step in weeks.
        n_components (int): Number of GMM clusters.
        
    Returns:
        dict: WFA metrics and per-period results.
    """
    results = []
    
    # Define time deltas
    lookback = pd.DateOffset(months=lookback_months)
    step = pd.DateOffset(weeks=step_weeks)
    
    # Get date range
    start_date = df_features.index.min()
    end_date = df_features.index.max()
    
    # First IS window starts at beginning, first OOS starts after lookback
    current_oos_start = start_date + lookback
    
    iteration = 0
    total_iterations = 0
    
    # Count total iterations for progress
    temp_date = current_oos_start
    while temp_date + step <= end_date:
        total_iterations += 1
        temp_date += step
    
    print(f"Starting WFA Simulation:")
    print(f"  Data Range: {start_date.date()} to {end_date.date()}")
    print(f"  IS Window: {lookback_months} months, OOS Step: {step_weeks} week(s)")
    print(f"  Total Iterations: {total_iterations}")
    print("-" * 60)
    
    while current_oos_start + step <= end_date:
        iteration += 1
        
        # Define IS and OOS periods
        is_start = current_oos_start - lookback
        is_end = current_oos_start
        oos_start = current_oos_start
        oos_end = current_oos_start + step
        
        # Slice data
        is_data = df_features[(df_features.index >= is_start) & (df_features.index < is_end)]
        oos_data = df_features[(df_features.index >= oos_start) & (df_features.index < oos_end)]
        
        # Skip if insufficient data
        if len(is_data) < 100 or len(oos_data) < 10:
            print(f"  [{iteration}/{total_iterations}] Skipping - insufficient data (IS: {len(is_data)}, OOS: {len(oos_data)})")
            current_oos_start += step
            continue
        
        try:
            # Train GMM on IS data
            gmm, scaler = train_gmm(is_data, n_components=n_components)
            
            # Predict on OOS data
            X_oos = oos_data[['hurst', 'volatility_atr', 'trend_adx']].values
            X_oos_scaled = scaler.transform(X_oos)
            oos_regimes = gmm.predict(X_oos_scaled)
            
            # Also predict on IS for comparison (training fit)
            X_is = is_data[['hurst', 'volatility_atr', 'trend_adx']].values
            X_is_scaled = scaler.transform(X_is)
            is_regimes = gmm.predict(X_is_scaled)
            
            # Calculate metrics
            oos_regime_counts = np.bincount(oos_regimes, minlength=n_components)
            is_regime_counts = np.bincount(is_regimes, minlength=n_components)
            
            # Regime stability: count transitions
            oos_transitions = np.sum(np.diff(oos_regimes) != 0)
            stability_ratio = 1.0 - (oos_transitions / max(1, len(oos_regimes) - 1))
            
            # Dominant regime
            dominant_regime = int(np.argmax(oos_regime_counts))
            
            # Average log-likelihood (model fit quality)
            oos_log_likelihood = gmm.score(X_oos_scaled)
            is_log_likelihood = gmm.score(X_is_scaled)
            
            # Get CPO params for this iteration
            cpo_map, _ = derive_cpo_params(gmm, scaler)
            
            period_result = {
                'iteration': iteration,
                'is_start': str(is_start.date()),
                'is_end': str(is_end.date()),
                'oos_start': str(oos_start.date()),
                'oos_end': str(oos_end.date()),
                'is_samples': len(is_data),
                'oos_samples': len(oos_data),
                'oos_regime_distribution': oos_regime_counts.tolist(),
                'is_regime_distribution': is_regime_counts.tolist(),
                'dominant_regime': dominant_regime,
                'dominant_regime_label': cpo_map[str(dominant_regime)]['regime_label'],
                'oos_transitions': int(oos_transitions),
                'stability_ratio': round(stability_ratio, 4),
                'oos_log_likelihood': round(oos_log_likelihood, 4),
                'is_log_likelihood': round(is_log_likelihood, 4),
                'generalization_gap': round(is_log_likelihood - oos_log_likelihood, 4)
            }
            
            results.append(period_result)
            
            print(f"  [{iteration}/{total_iterations}] OOS: {oos_start.date()} | "
                  f"Dominant: R{dominant_regime} ({cpo_map[str(dominant_regime)]['regime_label'][:15]:15s}) | "
                  f"Stability: {stability_ratio:.2f}")
            
        except Exception as e:
            print(f"  [{iteration}/{total_iterations}] Error: {e}")
            
        # Roll forward
        current_oos_start += step
    
    print("-" * 60)
    print(f"WFA Complete: {len(results)} valid iterations")
    
    # Aggregate statistics
    if results:
        all_dominant = [r['dominant_regime'] for r in results]
        all_stability = [r['stability_ratio'] for r in results]
        all_gap = [r['generalization_gap'] for r in results]
        
        aggregate = {
            'total_iterations': len(results),
            'regime_frequency': {
                str(i): all_dominant.count(i) for i in range(n_components)
            },
            'avg_stability_ratio': round(np.mean(all_stability), 4),
            'std_stability_ratio': round(np.std(all_stability), 4),
            'avg_generalization_gap': round(np.mean(all_gap), 4),
            'std_generalization_gap': round(np.std(all_gap), 4)
        }
    else:
        aggregate = {}
    
    return {
        'wfa_params': {
            'lookback_months': lookback_months,
            'step_weeks': step_weeks,
            'n_components': n_components,
            'data_start': str(start_date.date()),
            'data_end': str(end_date.date())
        },
        'aggregate': aggregate,
        'periods': results
    }


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
        print("Running Full Walk-Forward Analysis...")
        
        try:
            wfa_results = run_wfa_full(
                df_features, 
                lookback_months=6, 
                step_weeks=1, 
                n_components=4
            )
            
            # Save WFA metrics
            if not os.path.exists(ARTIFACTS_DIR):
                os.makedirs(ARTIFACTS_DIR)
                
            with open(WFA_METRICS_PATH, 'w') as f:
                json.dump(wfa_results, f, indent=2)
                
            result['message'] = f"WFA Complete: {wfa_results['aggregate'].get('total_iterations', 0)} iterations."
            result['wfa_metrics_path'] = WFA_METRICS_PATH
            result['aggregate'] = wfa_results['aggregate']
            
            # Also train final model on all data for deployment
            print("\nTraining final model on full dataset...")
            model, scaler = train_gmm(df_features, n_components=4)
            trade_params, centroids = derive_cpo_params(model, scaler)
            
            joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'gmm_model.pkl'))
            joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
            
            with open(TRADE_PARAMS_PATH, 'w') as f:
                json.dump(trade_params, f, indent=4)
                
            result['centroids'] = centroids
            result['trade_params'] = trade_params
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"WFA Simulation failed: {e}"}
        
    return result


def generate_backtest_cheatsheet(data_source=None):
    """
    Generate a CSV cheatsheet for Strategy Tester backtesting.
    
    This function:
    1. Loads the full historical dataset
    2. Applies the trained GMM model to predict regime for every bar
    3. Maps regimes to CPO parameters using trade_params.json
    4. Saves as CSV: 3_ML_ARTIFACTS/backtest_cheatsheet.csv
    
    CSV Format: Time (epoch),RegimeID,DistMult,LotMult
    
    Args:
        data_source: File path to OHLC data, or None to use default.
        
    Returns:
        dict: Status and path to generated CSV.
    """
    print("=" * 60)
    print("Generating Backtest Cheatsheet")
    print("=" * 60)
    
    # 1. Load Data
    if data_source:
        df = load_data(data_source)
    else:
        # Try default local file
        test_path = os.path.join(BASE_DIR, '../data/EURUSD_M15.csv')
        if os.path.exists(test_path):
            df = load_data(test_path)
        else:
            return {"status": "error", "message": "No data provided and no local file found."}
    
    if df.empty:
        return {"status": "error", "message": "Loaded dataframe is empty."}
    
    print(f"  Loaded {len(df)} rows of price data")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # 2. Build Feature Matrix
    try:
        df_features = build_feature_matrix(df)
        if df_features.empty:
            return {"status": "error", "message": "Feature matrix is empty (not enough data?)."}
        print(f"  Built feature matrix: {len(df_features)} rows")
    except Exception as e:
        return {"status": "error", "message": f"Feature Engineering failed: {e}"}
    
    # 3. Load trained model and scaler
    model_path = os.path.join(ARTIFACTS_DIR, 'gmm_model.pkl')
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return {"status": "error", "message": "Model or scaler not found. Run training first."}
    
    gmm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"  Loaded GMM model and scaler")
    
    # 4. Load trade params
    if not os.path.exists(TRADE_PARAMS_PATH):
        return {"status": "error", "message": "trade_params.json not found. Run training first."}
    
    with open(TRADE_PARAMS_PATH, 'r') as f:
        trade_params = json.load(f)
    print(f"  Loaded trade params: {len(trade_params)} regimes")
    
    # 5. Predict regime for every bar
    X = df_features[['hurst', 'volatility_atr', 'trend_adx']].values
    X_scaled = scaler.transform(X)
    regimes = gmm.predict(X_scaled)
    
    print(f"  Predicted regimes for {len(regimes)} bars")
    print(f"  Regime distribution: {np.bincount(regimes, minlength=4)}")
    
    # 6. Build output DataFrame
    # Time as epoch seconds (for MQL5 datetime compatibility)
    times = df_features.index.astype('int64') // 10**9  # Convert to Unix epoch
    
    dist_mults = []
    lot_mults = []
    for r in regimes:
        params = trade_params.get(str(r), trade_params.get("0", {}))
        dist_mults.append(params.get("distance_multiplier", 1.5))
        lot_mults.append(params.get("lot_multiplier", 1.2))
    
    cheatsheet_df = pd.DataFrame({
        'Time': times,
        'RegimeID': regimes,
        'DistMult': dist_mults,
        'LotMult': lot_mults
    })
    
    # Sort by time to ensure proper binary search in MQL5
    cheatsheet_df = cheatsheet_df.sort_values('Time').reset_index(drop=True)
    
    # 7. Save CSV
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    csv_path = os.path.join(ARTIFACTS_DIR, 'backtest_cheatsheet.csv')
    cheatsheet_df.to_csv(csv_path, index=False)
    
    print(f"  ✅ Saved cheatsheet to: {csv_path}")
    print(f"  Total rows: {len(cheatsheet_df)}")
    print(f"  Time range: {cheatsheet_df['Time'].min()} to {cheatsheet_df['Time'].max()}")
    print("=" * 60)
    
    return {
        "status": "success",
        "path": csv_path,
        "rows": len(cheatsheet_df),
        "time_range": [int(cheatsheet_df['Time'].min()), int(cheatsheet_df['Time'].max())]
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cheatsheet":
        # Generate backtest cheatsheet
        res = generate_backtest_cheatsheet()
    else:
        # Default: run WFA simulation
        res = run_wfa_simulation()
    
    print(json.dumps(res, indent=2, default=str))

