"""
FastAPI Inference Server for Adaptive Trading Parameters
=========================================================
Serves ML-predicted trading parameters to the MQL5 EA via HTTP POST.

Endpoints:
    POST /predict - Returns regime-based trading parameters (with live inference)
    GET  /health  - Health check endpoint
    POST /reload  - Hot-reload trade parameters
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import numpy as np
import pandas as pd
import joblib

# Import feature engineering
try:
    from src.feature_engineering import build_feature_matrix
except ImportError:
    from feature_engineering import build_feature_matrix

# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, '../config/trade_params.json')
ARTIFACTS_DIR = os.path.join(BASE_DIR, '../../3_ML_ARTIFACTS')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'gmm_model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')

# ============================================================
# 2. LOAD ARTIFACTS AT STARTUP
# ============================================================
trade_params = {}
gmm_model = None
scaler = None

def load_trade_params():
    """Load trade parameters from JSON file."""
    global trade_params
    try:
        with open(PARAMS_PATH, 'r') as f:
            trade_params = json.load(f)
            print(f"✅ Loaded trade params: {len(trade_params)} regimes defined.")
    except Exception as e:
        print(f"❌ Error loading trade params: {e}")
        trade_params = {
            "0": {"distance_multiplier": 1.5, "lot_multiplier": 1.2, "regime_label": "Default"}
        }

def load_model():
    """Load GMM model and scaler from pkl files."""
    global gmm_model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            gmm_model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Loaded GMM model and scaler from {ARTIFACTS_DIR}")
        else:
            print(f"⚠️ Model files not found. Live inference will use fallback.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        gmm_model = None
        scaler = None

# Load on module import
load_trade_params()
load_model()

# ============================================================
# 3. PYDANTIC MODELS
# ============================================================
class OHLCBar(BaseModel):
    """Single OHLC bar from MQL5."""
    time: int       # Unix epoch seconds
    open: float
    high: float
    low: float
    close: float

class PredictRequest(BaseModel):
    action: str
    symbol: str
    magic: Optional[int] = None
    ohlc_data: Optional[List[OHLCBar]] = None  # Live data for inference

class ParamsResponse(BaseModel):
    distance_multiplier: float
    lot_multiplier: float

class PredictResponse(BaseModel):
    regime_id: int
    params: ParamsResponse
    status: str
    regime_label: Optional[str] = None
    error_msg: Optional[str] = None
    inference_mode: Optional[str] = None  # 'live' or 'fallback'

# ============================================================
# 4. INFERENCE LOGIC
# ============================================================
def predict_regime_from_ohlc(ohlc_data: List[OHLCBar]) -> tuple:
    """
    Calculate features from OHLC data and predict regime.
    
    Returns:
        tuple: (regime_id, regime_label, error_message)
    """
    global gmm_model, scaler, trade_params
    
    if gmm_model is None or scaler is None:
        return None, None, "Model not loaded"
    
    if not ohlc_data or len(ohlc_data) < 150:
        return None, None, f"Insufficient data: {len(ohlc_data) if ohlc_data else 0} bars (need 150+)"
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([{
            'time': pd.to_datetime(bar.time, unit='s'),
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close
        } for bar in ohlc_data])
        
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Build feature matrix (this calculates Hurst, ATR, ADX)
        df_features = build_feature_matrix(df)
        
        if df_features.empty:
            return None, None, "Feature matrix is empty after calculation"
        
        # Get the latest feature row
        latest_features = df_features.iloc[-1:][['hurst', 'volatility_atr', 'trend_adx']].values
        
        # Scale and predict
        features_scaled = scaler.transform(latest_features)
        regime_id = int(gmm_model.predict(features_scaled)[0])
        
        # Get label from trade_params
        regime_info = trade_params.get(str(regime_id), {})
        regime_label = regime_info.get("regime_label", f"Regime {regime_id}")
        
        print(f"🔮 Live inference: Regime {regime_id} ({regime_label}) | "
              f"H={latest_features[0][0]:.3f}, ATR={latest_features[0][1]:.4f}, ADX={latest_features[0][2]:.1f}")
        
        return regime_id, regime_label, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Inference error: {str(e)}"

# ============================================================
# 5. FASTAPI APP
# ============================================================
app = FastAPI(
    title="Adaptive Trading Inference Server",
    description="ML-powered regime detection and parameter server for FXATM",
    version="3.0.0"  # Updated version for live inference
)

@app.on_event("startup")
async def startup_event():
    """Reload params and model on startup."""
    load_trade_params()
    load_model()
    print("🚀 FastAPI Inference Server started on http://0.0.0.0:8000")
    print(f"   Trade params loaded: {len(trade_params)} regimes")
    print(f"   Live inference: {'ENABLED' if gmm_model is not None else 'DISABLED (model not loaded)'}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "regimes_loaded": len(trade_params),
        "model_loaded": gmm_model is not None
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Return trading parameters for the current predicted regime.
    
    If ohlc_data is provided, performs live inference using GMM model.
    Otherwise, returns a fallback regime (Regime 0).
    """
    action = request.action
    
    if action == "GET_PARAMS":
        regime_id = None
        regime_label = None
        error_msg = None
        inference_mode = "fallback"
        
        # Try live inference if data is provided
        if request.ohlc_data and len(request.ohlc_data) >= 150:
            regime_id, regime_label, error_msg = predict_regime_from_ohlc(request.ohlc_data)
            if regime_id is not None:
                inference_mode = "live"
        
        # Fallback to default regime if live inference failed
        if regime_id is None:
            regime_id = 0
            regime_label = "Default (Fallback)"
            if error_msg:
                print(f"⚠️ Fallback: {error_msg}")
        
        # Get params for the regime
        params = trade_params.get(str(regime_id), 
                                   trade_params.get("0", {
                                       "distance_multiplier": 1.5, 
                                       "lot_multiplier": 1.2
                                   }))
        
        return PredictResponse(
            regime_id=regime_id,
            params=ParamsResponse(
                distance_multiplier=params.get("distance_multiplier", 1.5),
                lot_multiplier=params.get("lot_multiplier", 1.2)
            ),
            status="OK",
            regime_label=regime_label or params.get("regime_label", "Unknown"),
            error_msg=error_msg or "",
            inference_mode=inference_mode
        )
    else:
        return PredictResponse(
            regime_id=0,
            params=ParamsResponse(distance_multiplier=1.5, lot_multiplier=1.2),
            status="ERROR",
            error_msg=f"Unknown action: {action}",
            inference_mode="error"
        )

@app.post("/reload")
async def reload_params():
    """Hot-reload trade parameters and model without restarting the server."""
    load_trade_params()
    load_model()
    return {
        "status": "OK", 
        "message": "Parameters and model reloaded", 
        "regimes": len(trade_params),
        "model_loaded": gmm_model is not None
    }

# ============================================================
# 6. RUN WITH UVICORN (for direct execution)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
