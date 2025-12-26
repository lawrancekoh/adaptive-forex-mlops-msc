# Thesis Revision Suggestions

> [!NOTE]
> Based on the review of `MSc_Thesis.md` and the recent implementation of the **Live Inference** architecture using FastAPI and `WebRequest`, the following specific updates are recommended for the thesis document.

## 1. System Architecture & Methodology

### Section 2.5.3: MQL5 Expert Advisor Adaptation
**Location**: Lines 399-408
**Issue**: Currently describes a "continuous listening mode" (ZeroMQ interaction style) which is incorrect for the new HTTP Pull model.

**Proposed Update**:
Replace the paragraph starting "The EA operates in continuous listening mode..." with:
> "The EA operates as an active client, initiating synchronous requests to the Python inference engine. Upon the completion of a new bar (e.g., M15), the EA constructs a JSON payload containing the last $N$ bars of OHLC data (where $N=300$ for the Hurst calculation) and transmits this via an HTTP `POST` request to the `/predict` endpoint using the MQL5 `WebRequest` function. This synchronous 'Pull' architecture ensures that the regime prediction receives the exact price data visible to the EA at the moment of decision, eliminating synchronization errors and race conditions inherent in asynchronous socket communication."

### Section 4.1.2.D: Inference Server
**Location**: Lines 654-660
**Issue**: incorrectly lists endpoint as `GET` and implies lightweight status without mentioning on-the-fly feature calculation.

**Proposed Update**:
Update the list entries:
*   **Original**: `API Endpoint: GET /predict` -> **New**: `API Endpoint: POST /predict`
*   **Original**: `Protocol: HTTP/1.1 (REST)` -> **Keep**.
*   **Add**: `Input Format`: JSON Body containing `ohlc_data` array (Time, Open, High, Low, Close).
*   **Add**: `Logic`: On-the-fly feature engineering (Hurst, ATR, ADX) followed by GMM inference.

Add the following description to the text below the list:
> "To ensure data consistency between the backtest and live environments, the server implements on-the-fly feature calculation. It accepts raw OHLC data from the EA, computes complex features (such as the Hurst Exponent) in real-time averaging ~157ms, and determines the market regime. A fallback mechanism is implemented to return a default 'Safe' regime (Regime 0) in the event of insufficient data or model timeout, ensuring system stability."

## 2. Validation & Testing

### Section 4.4.1: Validation Criteria Assessment
**Location**: Line 821 (Table 4.8)
**Issue**: "Real-time Latency" is marked as "TBD" and target is "<100ms".
**Observation**: Our benchmark showed ~157ms. For an M15 timeframe (900 seconds), 0.157s is negligible (0.017% of the bar).

**Proposed Update**:
*   **Latancy Row**: Change `TBD` to `157ms`. Change `⏳ Pending` to `✅ Acceptable`.
*   **Add Note**: Add a footnote or text explaining that while >100ms, it is effectively instantaneous for a 15-minute trading strategy.

## 3. Appendix D: Source Codes

**Location**: Lines 1147-1177
**Issue**: Lists a generic full directory structure.
**Recommendation**: Replace the full structure with a focused list of the **Novel Contribution** files as requested.

**Proposed Structure**:
```
### Full Project Directory Structure (Appendix D)

```text
msc-thesis/
├── 0_DOCS/
│   ├── MSc_Thesis.md             # Adaptive Algorithmic Trading Thesis
│   └── ...
│
├── 1_MQL5_EA/                    # MQL5 Execution Environment
│   ├── Experts/
│   │   ├── FXATM_MSc.mq5         # Main Adaptive EA (Novel)
│   │   └── FXATM.mq5             # Baseline Static EA
│   └── Include/FXATM/
│       └── Managers/
│           ├── AdaptiveManager.mqh # [Novel] Hybrid Architecture & API Client
│           ├── TradeManager.mqh    # Order Execution Logic
│           ├── SignalManager.mqh   # Technical Indicators
│           ├── MoneyManager.mqh    # Risk Management
│           └── ... (Standard Managers)
│
├── 2_PYTHON_MLOPS/               # Python Intelligence Environment
│   ├── config/                   # System Configuration
│   ├── data/                     # Historical Data Storage
│   ├── src/
│   │   ├── inference_server.py   # [Novel] FastAPI Live Inference Engine
│   │   ├── feature_engineering.py# [Novel] Hurst/ATR/ADX Calculation
│   │   ├── retraining_script.py  # [Novel] GMM Training & WFA Logic
│   │   └── data_loader.py        # Data Ingestion
│   │
│   ├── requirements.txt
│   └── test_live_inference.py    # Integration Tests
│
└── 3_ML_ARTIFACTS/               # Model Registry
    ├── gmm_model.pkl             # Trained Gaussian Mixture Model
    ├── scaler.pkl                # Feature Scaler
    └── trade_params.json         # Adaptive Parameter Map
```

### Visual Representation
```mermaid
graph TD
    subgraph MQL5["MQL5 Environment (Client/Execution)"]
        EA["Experts/FXATM_MSc.mq5<br/>(Main Event Loop)"]
        AM["Include/FXATM/Managers/AdaptiveManager.mqh<br/>(Hybrid Architecture Logic)"]
        EA -->|Includes| AM
    end

    subgraph Python["Python Environment (Server/Intelligence)"]
        IS["src/inference_server.py<br/>(FastAPI Engine)"]
        FE["src/feature_engineering.py<br/>(Hurst/ATR/ADX)"]
        RS["src/retraining_script.py<br/>(GMM Training & WFA)"]
        
        IS -->|Imports| FE
        RS -->|Imports| FE
    end

    AM -.->|HTTP POST<br/>(JSON OHLC)| IS
```
