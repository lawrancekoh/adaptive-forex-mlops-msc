"""
FastAPI Inference Server for Adaptive Trading Parameters
=========================================================
Serves ML-predicted trading parameters to the MQL5 EA via HTTP POST.

Endpoints:
    POST /predict - Returns regime-based trading parameters
    GET  /health  - Health check endpoint
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, '../config/trade_params.json')

# ============================================================
# 2. LOAD TRADE PARAMS AT STARTUP
# ============================================================
trade_params = {}

def load_trade_params():
    """Load trade parameters from JSON file."""
    global trade_params
    try:
        with open(PARAMS_PATH, 'r') as f:
            trade_params = json.load(f)
            print(f"✅ Loaded trade params: {len(trade_params)} regimes defined.")
    except Exception as e:
        print(f"❌ Error loading trade params: {e}")
        # Fallback defaults
        trade_params = {
            "0": {"distance_multiplier": 1.5, "lot_multiplier": 1.2, "regime_label": "Default"}
        }

# Load on module import
load_trade_params()

# ============================================================
# 3. PYDANTIC MODELS
# ============================================================
class PredictRequest(BaseModel):
    action: str
    symbol: str
    magic: Optional[int] = None

class ParamsResponse(BaseModel):
    distance_multiplier: float
    lot_multiplier: float

class PredictResponse(BaseModel):
    regime_id: int
    params: ParamsResponse
    status: str
    regime_label: Optional[str] = None
    error_msg: Optional[str] = None

# ============================================================
# 4. FASTAPI APP
# ============================================================
app = FastAPI(
    title="Adaptive Trading Inference Server",
    description="ML-powered regime detection and parameter server for FXATM",
    version="2.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Reload params on startup."""
    load_trade_params()
    print("🚀 FastAPI Inference Server started on http://0.0.0.0:8000")
    print(f"   Trade params loaded: {len(trade_params)} regimes")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "regimes_loaded": len(trade_params)}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Return trading parameters for the current predicted regime.
    
    For now, this returns a static regime (regime_id=0).
    In a full implementation, this would:
    1. Accept real-time features from the EA
    2. Use the GMM model to predict the current regime
    3. Return the corresponding parameters
    
    Since the EA can't easily calculate Hurst/ATR/ADX in MQL5 for the model,
    and to keep things simple, we return the most common/safest regime.
    The backtest cheatsheet provides the historical lookup.
    """
    action = request.action
    
    if action == "GET_PARAMS":
        # Default to regime 0 (most conservative) for live trading
        # The full WFA analysis shows regime distribution across time
        # For live, we use a sensible default
        default_regime_id = 0
        
        params = trade_params.get(str(default_regime_id), 
                                   trade_params.get("0", {
                                       "distance_multiplier": 1.5, 
                                       "lot_multiplier": 1.2
                                   }))
        
        return PredictResponse(
            regime_id=default_regime_id,
            params=ParamsResponse(
                distance_multiplier=params.get("distance_multiplier", 1.5),
                lot_multiplier=params.get("lot_multiplier", 1.2)
            ),
            status="OK",
            regime_label=params.get("regime_label", "Unknown"),
            error_msg=""
        )
    else:
        return PredictResponse(
            regime_id=0,
            params=ParamsResponse(distance_multiplier=1.5, lot_multiplier=1.2),
            status="ERROR",
            error_msg=f"Unknown action: {action}"
        )

@app.post("/reload")
async def reload_params():
    """Hot-reload trade parameters without restarting the server."""
    load_trade_params()
    return {"status": "OK", "message": "Parameters reloaded", "regimes": len(trade_params)}

# ============================================================
# 5. RUN WITH UVICORN (for direct execution)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
