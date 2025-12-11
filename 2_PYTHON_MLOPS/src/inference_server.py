import zmq
import time
import json
import yaml
import os

# 1. Load configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, '../config/config.yaml')
PARAMS_PATH = os.path.join(BASE_DIR, '../config/trade_params.json')

print(f"Loading config from {os.path.abspath(CONFIG_PATH)}")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
with open(PARAMS_PATH, 'r') as f:
    trade_params = json.load(f)

ZMQ_HOST = config.get('MLOps / IPC Configuration', {}).get('ZMQ_HOST', "tcp://*:5555")

# Placeholder: Load a default regime (Regime ID 0 for the test)
# In a real system, a model would be loaded here:
# model = load_model(config['MODEL_ARTIFACT_PATH'])

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(ZMQ_HOST)

print(f"ZMQ Inference Server started on {ZMQ_HOST}")

# Simulate the adaptive logic for testing
current_regime_id = "0" 

while True:
    try:
        # Wait for the request from MQL5 EA
        message = socket.recv_json()
        timestamp = int(time.time())

        # Check for simple ping/heartbeat if needed, or structured requests
        if message.get('action') == 'GET_PARAMS':
            print(f"Received request: {message}")
            
            # In a real system:
            # 1. Fetch live data (via MT5 library or other)
            # 2. Extract features
            # 3. Predict regime: current_regime_id = str(model.predict(features)[0])
            
            # --- Get the adaptive parameters ---
            # Default to regime 0 if not found
            params_to_send = trade_params.get(current_regime_id, trade_params['0'])
            
            # --- Construct the JSON response (matching the Thesis schema 2.5.2) ---
            response_payload = {
                "regime_id": int(current_regime_id),
                "timestamp": timestamp,
                "symbol": message.get('symbol', 'EURUSD'),
                "params": {
                    "distance_multiplier": params_to_send['distance_multiplier'],
                    "lot_multiplier": params_to_send['lot_multiplier']
                },
                "status": "OK"
            }

            socket.send_json(response_payload)
            print(f"Sent adaptive params for Regime {current_regime_id}: {params_to_send}")
            
            # Rotate regime for a quick test of adaptation
            # This simulates the model changing its mind next time
            current_regime_id = str((int(current_regime_id) + 1) % len(trade_params))

        else:
            print(f"Received unknown action: {message}")
            socket.send_json({"status": "ERROR", "message": "Invalid action."})
    
    except zmq.error.ZMQError as e:
        print(f"ZMQ Error: {e}")
        time.sleep(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        time.sleep(1)
