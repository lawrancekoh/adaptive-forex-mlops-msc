# Technical Specification: Adaptive Forex MLOps System

**Version:** 1.0  
**Status:** Approved for Implementation  
**Related Documents:** [PRD.md](PRD.md), [msc_thesis.md](msc_thesis.md)

---

## 1. Data Structures & Schema

### 1.1. Market Data (Pandas DataFrame)
**Source:** User Upload via `streamlit_app.py` (`st.file_uploader`) -> CSV Parser.
**Format:** Standard MT5 Export CSV (Header: `<DATE>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>`).

The internal DataFrame must strictly adhere to this schema after preprocessing:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `time` | `datetime64[ns]` | Timestamp of the bar open. Index. |
| `open` | `float64` | |
| `high` | `float64` | |
| `low` | `float64` | |
| `close` | `float64` | |
| `tick_volume` | `int64` | |
| `log_returns` | `float64` | $ln(P_t / P_{t-1})$ |
| `volatility_atr` | `float64` | Normalized ATR (14). |
| `trend_adx` | `float64` | ADX (14). |
| `hurst` | `float64` | Rolling Hurst Exponent (100). |
| `regime_cluster`| `int32` | GMM Cluster Label (0-3). |

### 1.2. Feature Matrix ($X$)
The input vector for the GMM Model (`model.predict(X)`).
*   **Columns:** `['hurst', 'volatility_atr', 'trend_adx']` (Order is critical).
*   **Scaling:** Standard Scaler (Z-Score) applied before clustering? **No** (Thesis implies raw/normalized features, but typically scaling is needed for GMM. *Decision: Monitor raw values first as H is 0-1, ADX is 0-100, ATR is small. We may need to normalize ADX/100.*)

---

## 2. IPC Protocol (ZMQ / JSON)

**Transport:** TCP Port 5555  
**Pattern:** REP/REQ (Python=REP, MQL5=REQ)

### 2.1. Request (MQL5 -> Python)
Triggered on new candle event.

```json
{
  "action": "GET_PARAMS",
  "symbol": "EURUSD",
  "magic_number": 123456
}
```

### 2.2. Response (Python -> MQL5)
Calculated based on the `trade_params.json` lookup for the predicted regime.

```json
{
  "status": "OK",
  "timestamp": 1715420000,
  "regime_id": 2,
  "regime_label": "High Volatility", 
  "params": {
    "distance_multiplier": 2.5,
    "lot_multiplier": 1.2
  },
  "error_msg": "" 
}
```

*Error Case:* `{"status": "ERROR", "error_msg": "Model not loaded"}`

---

## 3. Artifacts & Persistence

### 3.1. File Locations
All artifacts are stored in `3_ML_ARTIFACTS/`.

| Filename | Format | Content | Creator | Consumer |
| :--- | :--- | :--- | :--- | :--- |
| `gmm_model.pkl` | `pickle` | Trained `sklearn.mixture.GaussianMixture` object. | `retraining_script.py` | `inference_server.py` |
| `trade_params.json` | `json` | Lookup Map (Regime ID -> Multipliers). | `retraining_script.py` | `inference_server.py` & `streamlit_app.py` |
| `wfa_metrics.json` | `json` | List of dicts: `[{date, sharpe, recovery, regime_counts...}]` | `retraining_script.py` | `streamlit_app.py` |
| `scaler.pkl` | `pickle` | (Optional) `StandardScaler` if used. | `retraining_script.py` | `inference_server.py` |

### 3.2. `trade_params.json` Structure
This file couples the abstract "Cluster ID" to concrete "EA Settings".

```json
{
  "0": {
    "regime_label": "Ranging",
    "distance_multiplier": 1.2,
    "lot_multiplier": 1.5,
    "rationale": "Low H, Low Vol"
  },
  "1": {
    "regime_label": "Trending",
    "distance_multiplier": 2.0,
    "lot_multiplier": 1.1,
    "rationale": "High H, High ADX"
  }
  // ... maps for all k=4 clusters
}
```

---

## 4. Simulation Logic Specification

### 4.1. The `retraining_script.py`
Must implement two modes callable by Streamlit:

1.  **`run_wfa_simulation(mode='single_train')`**:
    *   Load latest 5000 bars from MT5 (or CSV).
    *   Calc Features.
    *   Fit GMM (k=4).
    *   Save `gmm_model.pkl`.
    *   Update `trade_params.json` (Logic: Sort clusters by Hurst mean; assigns Labels based on thresholds).

2.  **`run_wfa_simulation(mode='full')`** (Thesis Ch 4 validation):
    *   Iterate sliding window (e.g., Step=1 week, Lookback=6 months).
    *   For each step: Train -> Test on next week -> Record Equity Curve.
    *   Save `wfa_metrics.json`.
