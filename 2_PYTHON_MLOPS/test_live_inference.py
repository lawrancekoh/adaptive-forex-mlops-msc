"""
Test script for live inference API.
Simulates sending OHLC data as the MQL5 EA would.
"""

import requests
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configuration
API_URL = "http://localhost:8000/predict"
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/EURUSD_M15.csv')

def load_sample_data():
    """Load sample OHLC data from local CSV."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} rows from {DATA_PATH}")
        return df
    else:
        # Generate random data if no file
        print("No CSV found, generating random data...")
        n = 300
        base_time = int((datetime.now() - timedelta(days=5)).timestamp())
        return pd.DataFrame({
            'time': [base_time + i * 900 for i in range(n)],  # 15-min intervals
            'open': np.random.rand(n) + 1.1,
            'high': np.random.rand(n) + 1.2,
            'low': np.random.rand(n) + 1.0,
            'close': np.random.rand(n) + 1.1
        })

def test_live_inference():
    """Test the live inference endpoint."""
    print("=" * 60)
    print("Testing Live Inference API")
    print("=" * 60)
    
    # Load data
    df = load_sample_data()
    
    # Prepare 300 bars for inference (take recent ones)
    df_sample = df.tail(300).copy()
    
    # Convert to the expected format
    ohlc_data = []
    for _, row in df_sample.iterrows():
        # Handle different column name conventions
        time_val = row.get('time', row.get('Time', 0))
        if isinstance(time_val, str):
            time_val = int(pd.Timestamp(time_val).timestamp())
        
        ohlc_data.append({
            "time": int(time_val),
            "open": float(row.get('open', row.get('Open', 1.1))),
            "high": float(row.get('high', row.get('High', 1.2))),
            "low": float(row.get('low', row.get('Low', 1.0))),
            "close": float(row.get('close', row.get('Close', 1.1)))
        })
    
    # Build request
    request_body = {
        "action": "GET_PARAMS",
        "symbol": "EURUSD",
        "magic": 123456,
        "ohlc_data": ohlc_data
    }
    
    print(f"Sending {len(ohlc_data)} bars to {API_URL}")
    print(f"Payload size: {len(json.dumps(request_body))} bytes")
    
    try:
        response = requests.post(API_URL, json=request_body, timeout=30)
        
        print(f"\nResponse Code: {response.status_code}")
        print(f"Response Body:\n{json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("inference_mode") == "live":
                print("\n✅ LIVE INFERENCE SUCCESSFUL!")
            else:
                print("\n⚠️ Fallback mode used (check server logs)")
        else:
            print("\n❌ Request failed")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Is the server running?")
        print("   Start with: uvicorn src.inference_server:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_fallback():
    """Test that fallback works when no data is sent."""
    print("\n" + "=" * 60)
    print("Testing Fallback (no OHLC data)")
    print("=" * 60)
    
    request_body = {
        "action": "GET_PARAMS",
        "symbol": "EURUSD",
        "magic": 123456
    }
    
    try:
        response = requests.post(API_URL, json=request_body, timeout=10)
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        data = response.json()
        if data.get("inference_mode") == "fallback":
            print("\n✅ Fallback mode working correctly")
        else:
            print("\n⚠️ Unexpected inference mode")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_live_inference()
    test_fallback()
