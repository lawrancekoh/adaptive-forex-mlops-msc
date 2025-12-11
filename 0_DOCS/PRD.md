## Product Requirements Document (PRD)

### Intelligent Forex Trading: Adaptive Machine Learning Framework

**Version:** 1.1 (Incorporating Streamlit & Simulated MLOps)
**Date:** December 11, 2025
**Author:** Lawrance Koh Chee Hng

---

### 1. Introduction and Goals

The core objective is to develop and validate an Adaptive Algorithmic Trading System that overcomes the challenge of **market non-stationarity** in the Forex market. This is achieved by dynamically adjusting the parameters of a Dollar-Cost Averaging (DCA) strategy based on real-time market regime classification performed by a Machine Learning (ML) model.

The ultimate goal is to prove, via a controlled simulation (Walk-Forward Analysis), that this **Adaptive ML-Driven DCA** system yields superior risk-adjusted performance (Sharpe Ratio, Recovery Factor) compared to a static, fixed-parameter strategy.

---

### 2. Project Architecture and Layer Segregation

The system is a **Hybrid Adaptive Trading System** segregated into four functional layers, emphasizing separation between analytical complexity, low-latency execution, and monitoring.

| Layer | Environment | Key Components | Core Responsibility |
| :--- | :--- | :--- | :--- |
| **Execution Layer (Client)** | MQL5 Expert Advisor | `1_MQL5_EA/Experts/FXATM.mq5` | **Low-Latency Trade Management:** Receives adaptive parameters, executes market orders, handles trade lifecycle (SL/TP, DCA). |
| **Analytical Layer (Server/Inference)**| Python ML Service | `2_PYTHON_MLOPS/src/inference_server.py` (Docker/Cloud Run Target) | **Real-Time Regime Classification:** Fetches data, extracts structural features (Hurst, ATR, ADX), runs GMM model prediction, and performs Conditional Parameter Optimization (CPO) lookup. |
| **MLOps Layer (Offline/Training)** | **Simulated Pipeline** | `2_PYTHON_MLOPS/src/retraining_script.py` | **Continuous Adaptation (Simulated):** Orchestrates **Walk-Forward Analysis (WFA)**. Performs data ingestion, GMM retraining, CPO, validation, and artifact generation on an *as-needed* basis, simulating a fixed weekly schedule. |
| **Presentation Layer (Monitoring)** | Streamlit Application | `2_PYTHON_MLOPS/streamlit_app.py` | **Visualization & Validation:** Interactive dashboard to trigger WFA/Training, display MLOps metrics, GMM cluster structure, and performance results for thesis documentation. |

The Execution and Analytical Layers communicate via **ZeroMQ (ZMQ)** for robust, low-latency Inter-Process Communication (IPC) using **JSON** data serialization.

---

### 3. Scope and Technical Stack

#### 3.1. Scope and Delimitations (Constraints)

1.  **Single-Asset Focus:** Exclusively focused on the **EUR/USD** currency pair.
2.  **Strategy Constraint:** Adaptive mechanism is restricted to controlling two critical variables of the Dollar-Cost Averaging (DCA) strategy: **Distance Multiplier** and **Lot Multiplier**.
3.  **Entry Logic Decoupling:** Trade entry signals are based on a **Static MACD** indicator to isolate the performance impact of the *adaptive risk management* layer (CPO).
4.  **Exclusion of Fundamental Data:** ML model relies strictly on Technical/Complexity Features (Hurst Exponent, ATR, ADX).
5.  **Simulated MLOps Orchestration (Constraint):** The weekly retraining schedule is executed by directly running the `retraining_script.py` via a **Streamlit front-end trigger** or local shell command, *simulating* a production cloud scheduler (e.g., Google Cloud Scheduler).
6.  **MLOps Presentation (Addition):** Inclusion of a **Streamlit** dashboard to visualize GMM cluster properties, WFA time series, and comparative performance metrics for evidence generation.

#### 3.2. Technical Stack

| Category | Component | Key Libraries |
| :--- | :--- | :--- |
| **Development** | Python 3.9+, MQL5 | numpy, pandas |
| **Machine Learning** | GMM Clustering, Feature Engineering (Hurst, ATR, ADX) | scikit-learn, TA-Lib, hurst, joblib, pyyaml |
| **Communication** | IPC (ZMQ), Data Serialization (JSON) | pyzmq, json |
| **MLOps/Infrastructure** | Containerization (Docker), Artifacts (PKL, JSON) | Docker, joblib, yaml |
| **Presentation** | Interactive Visualization | Streamlit, Plotly/Altair |

---

### 4. Operational Workflow

The system operates on a dual-mode workflow: the **Online Cycle** for real-time trading logic, and the **Offline Cycle** for continuous adaptation and validation.

#### A. Online Cycle (Inference - Runs 24/5)

| Step | Process | Responsibility |
| :--- | :--- | :--- |
| 1. Trigger | New M15 Bar Check | MQL5 EA |
| 2. Request | EA sends `GET_PARAMS` JSON request | MQL5 EA (REQ Client) |
| 3. Analysis | Server loads latest model, predicts Regime ID (0-3) | Python Server (ZMQ REP) |
| 4. Prescribe | Server fetches adaptive parameters from **CPO Logic Table** | Python Server (CPO Lookup) |
| 5. Respond | Server sends JSON response with `distance_multiplier` and `lot_multiplier` | Python Server (ZMQ REP) |
| 6. Execute | EA updates internal DCA variables and executes next trade/recovery using the new settings. | MQL5 EA |

#### B. Offline Cycle (Continuous Training - Simulated Trigger)

This cycle is primarily orchestrated by the `retraining_script.py`, with the **Streamlit front-end serving as the manual trigger and final display layer.**

| Step | Process | Responsibility |
| :--- | :--- | :--- |
| 1. Trigger | **Manual Trigger** (Button Click) | Streamlit Front-End |
| 2. Data Pull | Acquire 180 days of new market data (simulating continuity). | `retraining_script.py` |
| 3. Retrain/Segment | Execute GMM on new data, classify historical segments. | `retraining_script.py` (GMM) |
| 4. CPO & WFA Loop | Iteratively run the Walk-Forward Optimization (IS/OOS) over the entire historical window to derive a new optimal $\mathbf{P}_i$ for each of the $k=4$ segments. | `retraining_script.py` (Optimization Subprocess) |
| 5. Validation Capture | **Capture and Save**: WFA metrics (Sharpe/Recovery time series) and the final GMM Cluster Centroids. | `retraining_script.py` (Artifacts) |
| 6. Deployment | Generate and save the updated $\mathbf{P}_i$ CPO Logic Table (`trade_params.json`). | `retraining_script.py` (Artifacts) |
| 7. Monitor & Analyze | Streamlit application loads the newly saved artifacts to visualize comparative performance (Adaptive vs. Static) and GMM structure. | Streamlit Front-End |

---

### 5. Implementation Status

| Component | Status | Notes |
| :--- | :--- | :--- |
| **MQL5 Expert Advisor** | **Existing** | Located in `1_MQL5_EA/Experts/FXATM.mq5`. Needs verification against Thesis Appendix A. |
| **Inference Server** | **Pending** | `2_PYTHON_MLOPS/src/inference_server.py` to be implemented. |
| **Retraining Script** | **Pending** | `2_PYTHON_MLOPS/src/retraining_script.py` to be implemented. |
| **Streamlit App** | **Pending** | `2_PYTHON_MLOPS/streamlit_app.py` to be implemented. |
| **Docker Environment** | **Partial** | `Dockerfile` exists but requirements need update. |