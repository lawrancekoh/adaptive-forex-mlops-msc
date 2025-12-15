# Adaptive Algorithmic Trading System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MQL5](https://img.shields.io/badge/MQL5-MetaTrader%205-orange.svg)](https://www.mql5.com/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#disclaimer)

**Author:** Lawrance Koh  
**Project Type:** MSc Thesis  
**Domain:** Quantitative Finance / MLOps

---

## 🎯 Overview

An end-to-end **machine learning system** that dynamically adapts forex trading strategy parameters based on real-time market regime detection. The system addresses the challenge of **non-stationary markets** by automatically adjusting DCA (Dollar-Cost Averaging) grid trading parameters according to current market conditions.

### Key Innovation
Traditional algorithmic trading uses static parameters optimized for historical data. This system introduces **Cluster Parameter Optimization (CPO)** — a novel approach that maps unsupervised market regime clusters to context-aware trading parameters.

---

## 🧠 How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MetaTrader 5  │     │  Python ML Layer │     │   Streamlit     │
│   Expert Advisor│◄───►│  Inference Server│◄────│   Dashboard     │
│   (Execution)   │ ZMQ │  (GMM Classifier)│     │   (Monitoring)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │  Feature Engine  │
         │              │  • Hurst Exponent│
         │              │  • Normalized ATR│
         │              │  • ADX           │
         └──────────────┴──────────────────┘
```

1. **Regime Detection**: A Gaussian Mixture Model (GMM) classifies market conditions into 4 regimes using Hurst Exponent, ATR, and ADX features
2. **Parameter Mapping**: Each regime maps to optimized DCA parameters (grid spacing & position sizing)
3. **Real-time Adaptation**: ZeroMQ IPC enables sub-100ms parameter updates between Python and MQL5

---

## 📊 Key Results (Walk-Forward Analysis)

The system was validated using **Walk-Forward Analysis (WFA)** on EUR/USD M15 data from December 2021 to December 2024.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Regime Stability** | 87.75% | Regimes persist; low noise |
| **Generalization Gap** | 0.11 | Minimal overfitting |
| **WFA Iterations** | 133 | Sufficient statistical power |
| **Data Points** | 76,188 bars | ~3 years of M15 data |

### Regime Distribution
| Regime | Occurrence | Trading Behavior |
|--------|------------|------------------|
| Trending | 36.8% | Wide grids, conservative sizing |
| Strong Trend | 30.1% | Widest grids, minimal sizing |
| Choppy | 17.3% | Moderate grids, balanced sizing |
| Ranging | 15.8% | Tight grids, aggressive sizing |

---

## 🏗️ Architecture

```
msc-thesis/
├── 0_DOCS/                     # Thesis & Documentation
│   ├── PRD.md                  # Product Requirements
│   └── thesis_chapters_*.md    # Thesis Content
│
├── 1_MQL5_EA/                  # Trading Execution Layer
│   └── Experts/
│       ├── FXATM.mq5           # Baseline EA (Static)
│       └── FXATM_MSc.mq5       # Adaptive EA (ML-Integrated)
│
├── 2_PYTHON_MLOPS/             # Machine Learning Layer
│   ├── src/
│   │   ├── inference_server.py # ZMQ REP Server
│   │   ├── retraining_script.py# GMM Training & WFA
│   │   ├── feature_engineering.py
│   │   └── data_loader.py
│   ├── config/
│   │   ├── config.yaml         # System Configuration
│   │   └── trade_params.json   # CPO Parameter Mapping
│   └── streamlit_app.py        # Dashboard UI
│
└── 3_ML_ARTIFACTS/             # Trained Models
    ├── gmm_model.pkl           # GMM Classifier
    ├── scaler.pkl              # Feature Normalizer
    └── wfa_metrics.json        # Validation Results
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Trading Platform | MetaTrader 5 (MQL5) |
| ML Framework | Scikit-learn (GMM) |
| IPC Communication | ZeroMQ |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- MetaTrader 5 (Windows)
- ZeroMQ libraries

### Installation
```bash
# Clone and setup
git clone <repo_url>
cd msc-thesis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r 2_PYTHON_MLOPS/requirements.txt
```

### Usage
```bash
# Start inference server
python 2_PYTHON_MLOPS/src/inference_server.py

# Launch dashboard
streamlit run 2_PYTHON_MLOPS/streamlit_app.py
```

---

## 📈 Future Enhancements

- [ ] Multi-pair validation (GBP/USD, XAU/USD)
- [ ] Deep learning regime detection (LSTM/Transformer)
- [ ] Reinforcement learning for CPO optimization
- [ ] Live trading pilot

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only** as part of an MSc Thesis. It involves significant financial risk if used in live trading. The author assumes no responsibility for any trading losses.

---

*© 2025 Lawrance Koh*