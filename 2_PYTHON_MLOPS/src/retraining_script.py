import time
import os
import yaml
import json

# Setup paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '../config/config.yaml')
# Artifacts path relative to the script execution or defined in config
ARTIFACTS_DIR = os.path.join(BASE_DIR, '../../3_ML_ARTIFACTS')

def run_wfa_simulation(mode='single_train'):
    """
    Orchestrates the training/simulation process.
    
    Args:
        mode (str): 'single_train' for immediate model update (Centroid Update),
                    'full' for the full Walk-Forward Analysis (Chapter 4 data).
    
    Returns:
        dict: A dictionary containing status and results summary.
    """
    print(f"Starting Simulation in Mode: {mode}")
    timestamp = int(time.time())
    
    # 1. Load Config
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        return {"status": "error", "message": "Config file not found."}

    # Simulate Process Delays (Data Ingestion, Feature Engineering, Training)
    # In a real implementation, this would call GMM training logic.
    time.sleep(2) # Placeholder for computation time
    
    result = {
        "status": "success",
        "mode": mode,
        "timestamp": timestamp,
        "config_used": config
    }

    if mode == 'full':
        # Simulate Full WFA Loop
        print("Running Walk-Forward Analysis...")
        # Placeholder for iterating through historical windows
        result['message'] = "Full WFA Complete. Results saved to artifacts."
        result['wfa_metrics'] = {
            "sharpe_ratio_avg": 1.25,
            "recovery_factor": 3.4
        }
        # In reality, this would save a CSV/JSON time series to ARTIFACTS_DIR

    elif mode == 'single_train':
        # Simulate Single Model Retraining
        print("Retraining GMM on latest window...")
        # Placeholder for GMM fitting
        result['message'] = "Single Model Retrained. centroids updated."
        result['centroids'] = [
            {"regime": 0, "H": 0.45, "ATR": 0.0012},
            {"regime": 1, "H": 0.65, "ATR": 0.0025},
            {"regime": 2, "H": 0.55, "ATR": 0.0035},
            {"regime": 3, "H": 0.52, "ATR": 0.0009}
        ]
        # In reality, this would save model.pkl to ARTIFACTS_DIR

    return result

if __name__ == "__main__":
    # Allow command line execution for testing
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'single_train'
    res = run_wfa_simulation(mode)
    print(res)
