# Adaptive Algorithmic Trading System (MSc Thesis)

**Author:** Lawrance Koh  
**Project:** Adaptive Forex MLOps Framework (EUR/USD)  
**Version:** 1.0 (Simulation Phase)

## 📌 Project Overview
This repository contains the source code and documentation for an **Adaptive Algorithmic Trading System** designed to handle non-stationary financial markets. The system uses a **Hybrid Architecture** combining:
1.  **MQL5 Expert Advisor (Client):** Handles low-latency execution, risk management, and market interaction.
2.  **Python MLOps Layer (Server):** Performs market regime classification (GMM), Walk-Forward Analysis (WFA), and dynamic parameter optimization (CPO).
3.  **Streamlit Dashboard:** Provides a "Human-in-the-loop" interface to monitor regimes and trigger retraining simulations.

## 🏗️ System Architecture
The system operates via a **ZeroMQ (ZMQ)** IPC bridge between MetaTrader 5 and Python.

*   **Online Loop (Inference):** The EA requests updated parameters -> Python Server classifies current regime -> Returns optimal `Distance` and `Lot` multipliers.
*   **Offline Loop (Retraining):** A simulated weekly pipeline retrains the GMM model and updates the parameter lookup table (`trade_params.json`).

## 📂 Repository Structure

```text
.
├── 0_DOCS/                 # Documentation (PRD, Thesis, Notes)
├── 1_MQL5_EA/             
│   └── Experts/            # MQL5 Source Code (FXATM.mq5)
├── 2_PYTHON_MLOPS/         # Python Source & Configs
│   ├── config/             # Configuration (yaml) & Regime Map (json)
│   ├── src/                # Core Logic
│   │   ├── inference_server.py  # ZMQ Server (Online)
│   │   └── retraining_script.py # WFA & Training Logic (Offline)
│   ├── streamlit_app.py    # Dashboard Entry Point
│   ├── requirements.txt    # Python Dependencies
│   └── Dockerfile          # Container Definition
├── 3_ML_ARTIFACTS/         # Generated Models (.pkl) & Metrics
└── .venv/                  # Local Python Virtual Environment
```

## 🚀 Setup & Installation

### 1. Prerequisites
*   **OS:** Windows (for MT5 Client) or Linux (for Python Server/Dev).
*   **MetaTrader 5:** Installed and configured (Allow Import of DLLs for ZMQ).
*   **Python:** Version 3.9+.

### 2. Python Environment Setup
```bash
# Clone repository
git clone <repo_url>
cd msc-thesis

# Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install Dependencies
pip install -r 2_PYTHON_MLOPS/requirements.txt
```
*> **Note:** If on Linux, `MetaTrader5` and `TA-Lib` binaries may require specific system libraries or can be omitted if only running the ZMQ Server component.*

## ⚡ Usage

### A. Start the Inference Server (ZMQ)
This script listens for requests from the MQL5 EA on port `5555`.
```bash
python 2_PYTHON_MLOPS/src/inference_server.py
```

### B. Launch the Dashboard (MLOps Control)
The dashboard allows you to visualize the regime structure and trigger the **Walk-Forward Analysis** simulation manually.
```bash
streamlit run 2_PYTHON_MLOPS/streamlit_app.py
```

### C. MetaTrader 5 (MQL5 EA)
1.  Copy `1_MQL5_EA/Experts/FXATM.mq5` to your MT5 `Experts/` folder.
2.  Ensure `mql5-zmq.dll` (or equivalent zmq wrapper) is in `Libraries/`.
3.  Attach the EA to **EURUSD M15** chart.
4.  Ensure "Allow DLL imports" is checked in EA Common settings.

## ⚠️ Disclaimer
This software is for **educational and research purposes only** as part of an MSc Thesis. It involves significant financial risk if used in live trading. The author assumes no responsibility for any trading losses.

---
*Copyright © 2025 Lawrance Koh.*