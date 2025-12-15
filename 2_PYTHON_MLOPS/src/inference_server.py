import zmq
import time
import json
import yaml
import os
import joblib
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '../config/config.yaml')
PARAMS_PATH = os.path.join(BASE_DIR, '../config/trade_params.json')
ARTIFACTS_DIR = os.path.join(BASE_DIR, '../../3_ML_ARTIFACTS')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'gmm_model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')

print(f"Loading config from {os.path.abspath(CONFIG_PATH)}")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# ============================================================
# 2. LOAD ML ARTIFACTS
# ============================================================
def load_artifacts():
    """Load GMM model, scaler, and trade params."""
    global gmm_model, scaler, trade_params
    
    try:
        if os.path.exists(MODEL_PATH):
            gmm_model = joblib.load(MODEL_PATH)
            print(f"✅ Loaded GMM model from {MODEL_PATH}")
        else:
            gmm_model = None
            print(f"⚠️ Model not found at {MODEL_PATH}. Will use default regime.")
            
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Loaded Scaler from {SCALER_PATH}")
        else:
            scaler = None
            print(f"⚠️ Scaler not found at {SCALER_PATH}.")
            
        with open(PARAMS_PATH, 'r') as f:
            trade_params = json.load(f)
            print(f"✅ Loaded trade params: {len(trade_params)} regimes defined.")
            
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        gmm_model = None
        scaler = None
        trade_params = {"0": {"distance_multiplier": 1.5, "lot_multiplier": 1.2}}

# Initialize globals
gmm_model = None
scaler = None
trade_params = {}
load_artifacts()

# ============================================================
# 3. PREDICTION LOGIC
# ============================================================
def predict_regime(features):
    """
    Predict market regime from feature vector.
    
    Args:
        features: dict with keys ['hurst', 'volatility_atr', 'trend_adx']
                  OR list/array of [hurst, atr, adx]
    Returns:
        int: Cluster/Regime ID (0-3)
    """
    if gmm_model is None:
        print("Model not loaded. Returning default regime 0.")
        return 0
        
    try:
        # Convert dict to array if needed
        if isinstance(features, dict):
            X = np.array([[
                features.get('hurst', 0.5),
                features.get('volatility_atr', 0.001),
                features.get('trend_adx', 25.0)
            ]])
        else:
            X = np.array(features).reshape(1, -1)
        
        # Scale if scaler exists
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
            
        # Predict
        regime_id = gmm_model.predict(X_scaled)[0]
        return int(regime_id)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0

# ============================================================
# 4. ZMQ SERVER
# ============================================================
ZMQ_HOST = config.get('MLOps / IPC Configuration', {}).get('ZMQ_HOST', "tcp://*:5555")

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(ZMQ_HOST)

print(f"🚀 ZMQ Inference Server started on {ZMQ_HOST}")
print(f"   Model loaded: {'Yes' if gmm_model else 'No'}")
print(f"   Scaler loaded: {'Yes' if scaler else 'No'}")

while True:
    try:
        # Wait for request from MQL5 EA
        message = socket.recv_json()
        timestamp = int(time.time())
        
        action = message.get('action', '')
        
        if action == 'GET_PARAMS':
            print(f"📥 Received request: {message}")
            
            # Check if features are provided in the request
            features = message.get('features', None)
            
            if features:
                # Use provided features for prediction
                regime_id = predict_regime(features)
            else:
                # No features provided - use a default or last known regime
                # In production, MQL5 would send the features
                regime_id = 0  # Default to safest regime
                print("   No features in request, using default regime 0")
            
            # Get parameters for this regime
            params = trade_params.get(str(regime_id), trade_params.get('0', {}))
            
            # Build response
            response = {
                "status": "OK",
                "timestamp": timestamp,
                "symbol": message.get('symbol', 'EURUSD'),
                "regime_id": regime_id,
                "regime_label": params.get('regime_label', 'Unknown'),
                "params": {
                    "distance_multiplier": params.get('distance_multiplier', 1.5),
                    "lot_multiplier": params.get('lot_multiplier', 1.2)
                },
                "error_msg": ""
            }
            
            socket.send_json(response)
            print(f"📤 Sent response: Regime {regime_id} ({params.get('regime_label', 'Unknown')})")
            
        elif action == 'RELOAD_MODEL':
            # Allow hot-reloading of model without restart
            load_artifacts()
            socket.send_json({"status": "OK", "message": "Model reloaded."})
            
        else:
            print(f"❓ Unknown action: {action}")
            socket.send_json({"status": "ERROR", "error_msg": f"Unknown action: {action}"})
    
    except zmq.error.ZMQError as e:
        print(f"ZMQ Error: {e}")
        time.sleep(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(1)
