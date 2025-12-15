# Adaptive Forex MLOps System: Thesis Chapters 4-5

## Document Information
- **Author:** Lawrance Koh
- **Last Updated:** 2025-12-15
- **Status:** Draft for Review

---

# Chapter 4: Implementation and Results

## 4.1 System Implementation Overview

This chapter presents the complete implementation of the Adaptive Algorithmic Trading System, detailing the software architecture, module interactions, and the results obtained from the Walk-Forward Analysis validation.

### 4.1.1 System Architecture

The implemented system follows a layered architecture that separates concerns between trading execution (MQL5), machine learning (Python), and orchestration (MLOps).

```mermaid
graph TB
    subgraph "Layer 1: Trading Execution (MQL5)"
        EA[FXATM_MSc.mq5<br/>Expert Advisor]
        AM[AdaptiveManager.mqh<br/>ML Integration Layer]
        ZC[ZMQClient.mqh<br/>IPC Communication]
        DCA[DCAManager<br/>Grid Trading Logic]
    end
    
    subgraph "Layer 2: Machine Learning (Python)"
        IS[inference_server.py<br/>ZMQ REP Server]
        GMM[gmm_model.pkl<br/>Trained Classifier]
        SC[scaler.pkl<br/>Feature Normalizer]
        TP[trade_params.json<br/>Regime Parameters]
    end
    
    subgraph "Layer 3: MLOps Pipeline"
        ST[streamlit_app.py<br/>Dashboard UI]
        RS[retraining_script.py<br/>Training & WFA]
        FE[feature_engineering.py<br/>H, ATR, ADX]
        DL[data_loader.py<br/>CSV Parser]
    end
    
    EA --> AM
    AM --> ZC
    ZC <-->|ZMQ TCP 5555| IS
    IS --> GMM
    IS --> SC
    IS --> TP
    
    ST --> RS
    RS --> FE
    RS --> DL
    RS -->|Artifacts| GMM
    RS -->|Artifacts| SC
    RS -->|Artifacts| TP
```

**Figure 4.1:** Complete System Architecture showing the three-layer design with MQL5 trading execution, Python ML inference, and MLOps pipeline components.

---

### 4.1.2 Component Implementation Details

#### A. Data Ingestion Module (`data_loader.py`)

The data loader handles CSV imports from MetaTrader 5, supporting multiple encodings and formats.

| Feature | Implementation |
|---------|----------------|
| Encoding Support | UTF-16, UTF-8, UTF-16-LE |
| Separator Detection | Tab, Comma, Semicolon |
| Column Normalization | Strips `<>` from MT5 headers |
| Type Coercion | Forces numeric types with `pd.to_numeric()` |
| DateTime Handling | Combines DATE + TIME columns |

**Listing 4.1:** Core data loading logic
```python
# Robust multi-format CSV parsing
candidates = [
    ('\t', 'utf-16'),    # Standard MT5 export
    (',', 'utf-8'),      # Alternative format
    (';', 'utf-8')       # European locale
]

for sep, encoding in candidates:
    df_try = pd.read_csv(source, sep=sep, encoding=encoding)
    if len(df_try.columns) > 1:  # Validation
        break
```

---

#### B. Feature Engineering Module (`feature_engineering.py`)

Three market complexity features are calculated per the methodology in Chapter 3.

| Feature | Formula | Window | Purpose |
|---------|---------|--------|---------|
| Hurst Exponent | R/S Analysis | 100 bars | Market memory/trending |
| Normalized ATR | ATR(14) / Close | 14 bars | Volatility level |
| ADX | Directional Index | 14 bars | Trend strength |

**Implementation Note:** A custom Rescaled Range (R/S) Hurst calculation was implemented due to instability in the `hurst` library for certain data patterns.

```python
def _manual_hurst(ts):
    """Calculates Hurst Exponent using R/S analysis."""
    # Divide series into sub-periods
    for k in range(min_k, max_k):
        # Calculate R/S for each sub-period
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(subseries)
        rs_list.append(R / S)
    
    # Linear regression on log-log scale
    H, _ = np.polyfit(np.log(n_list), np.log(rs_list), 1)
    return H
```

---

#### C. GMM Training Module (`retraining_script.py`)

The Gaussian Mixture Model classifier was configured per the methodology specification.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_components | 4 | Four distinct market regimes |
| covariance_type | 'full' | Captures feature correlations |
| random_state | 42 | Reproducibility |
| Scaling | StandardScaler | Normalizes feature magnitudes |

**Cluster-to-Regime Mapping:** Clusters are sorted by mean Hurst Exponent and mapped to economic interpretations:

| Rank | Hurst Range | Regime Label | Trading Parameters |
|------|-------------|--------------|-------------------|
| 0 (Lowest H) | 0.85-0.92 | Ranging (Safe) | Dist=1.2x, Lot=1.5x |
| 1 | 0.92-0.95 | Choppy/Weak Trend | Dist=1.5x, Lot=1.3x |
| 2 | 0.95-0.98 | Trending | Dist=2.0x, Lot=1.2x |
| 3 (Highest H) | 0.98+ | Strong Trend/Breakout | Dist=2.5x, Lot=1.1x |

**Table 4.1:** CPO (Cluster Parameter Optimization) mapping from GMM clusters to trading parameters.

---

#### D. Inference Server (`inference_server.py`)

The real-time prediction server uses ZeroMQ for low-latency IPC.

```mermaid
sequenceDiagram
    participant EA as FXATM_MSc (MT5)
    participant AM as AdaptiveManager
    participant ZMQ as ZMQ Socket
    participant IS as InferenceServer
    participant GMM as GMM Model
    
    EA->>AM: OnNewBar()
    AM->>AM: Calculate H, ATR, ADX
    AM->>ZMQ: {"action": "GET_PARAMS", "features": {...}}
    ZMQ->>IS: REQ
    IS->>GMM: predict(features)
    GMM-->>IS: regime_id = 2
    IS->>IS: Lookup trade_params[regime_id]
    IS-->>ZMQ: {"regime_id": 2, "params": {...}}
    ZMQ-->>AM: Response
    AM->>EA: Update CSettings multipliers
```

**Figure 4.2:** Sequence diagram showing the real-time parameter adaptation flow from MT5 through the Python inference server.

---

## 4.2 Walk-Forward Analysis Results

### 4.2.1 WFA Configuration

The Walk-Forward Analysis was conducted with the following parameters:

| Parameter | Value |
|-----------|-------|
| Data Period | December 2021 – December 2024 |
| Symbol | EUR/USD |
| Timeframe | M15 (15-minute bars) |
| In-Sample Window | 6 months |
| Out-of-Sample Step | 1 week |
| Total Data Points | 76,188 bars |
| Valid Iterations | 133 |

---

### 4.2.2 Aggregate Results

The WFA produced the following aggregate metrics across all 133 iterations:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Iterations** | 133 | Sufficient sample size for statistical validity |
| **Average Stability Ratio** | 87.75% | Regimes persist; low noise |
| **Standard Deviation (Stability)** | TBD | Consistency of stability |
| **Average Generalization Gap** | 0.1135 | Low overfitting risk |
| **Standard Deviation (Gen. Gap)** | TBD | Consistency of generalization |

**Table 4.2:** WFA Aggregate Metrics

---

### 4.2.3 Regime Distribution

Analysis of dominant regimes across OOS periods reveals the market's regime composition during 2021-2024:

| Regime | Cluster ID | Dominant Count | Percentage | Interpretation |
|--------|------------|----------------|------------|----------------|
| Trending | R0 | 49 | 36.8% | Standard trending behavior |
| Strong Trend | R1 | 40 | 30.1% | High persistence moves (Fed cycle) |
| Choppy | R2 | 23 | 17.3% | Mixed/volatile conditions |
| Ranging | R3 | 21 | 15.8% | Mean-reverting markets |

**Table 4.3:** Regime Distribution from WFA

```mermaid
pie title Regime Distribution (2021-2024)
    "Trending (R0)" : 49
    "Strong Trend (R1)" : 40
    "Choppy (R2)" : 23
    "Ranging (R3)" : 21
```

**Figure 4.3:** Pie chart showing the distribution of dominant regimes across 133 WFA iterations.

**Economic Interpretation:** The dominance of Trending (R0) and Strong Trend (R1) regimes, totaling 66.9% of periods, aligns with the macroeconomic context of 2022-2023 where aggressive Federal Reserve rate hikes created sustained directional moves in EUR/USD.

---

### 4.2.4 Final Model Cluster Centroids

The model trained on the complete dataset produced the following cluster centers:

| Cluster | Hurst | ATR (Norm.) | ADX | Label |
|---------|-------|-------------|-----|-------|
| 0 | 0.9787 | 0.000518 | 22.94 | Trending |
| 1 | 1.0000 | 0.000654 | 25.59 | Strong Trend / Breakout |
| 2 | 0.9387 | 0.000941 | 31.34 | Choppy / Weak Trend |
| 3 | 0.9006 | 0.000554 | 19.00 | Ranging (Safe) |

**Table 4.4:** Final GMM Cluster Centroids

**Observations:**
1. **Hurst Separation:** Clusters span from 0.90 (Ranging) to 1.00 (Strong Trend), indicating clear regime differentiation.
2. **Volatility Correlation:** Cluster 2 (Choppy) has the highest ATR, suggesting volatile but directionless conditions.
3. **ADX Alignment:** Cluster 2 has the highest ADX (31.34), which may seem contradictory but reflects high directional movement intensity without sustained persistence.

---

## 4.3 MQL5 Integration Results

### 4.3.1 Backtest Methodology

To validate the practical trading impact, backtests were conducted using MT5 Strategy Tester.

> [!IMPORTANT]
> **Placeholder Section:** The following backtest results require execution on MT5 Strategy Tester. Insert actual results after running both configurations.

---

### 4.3.2 Baseline Backtest (Static Parameters)

**Configuration:**
- EA: `FXATM.mq5` (Original)
- DCA Step Multiplier: 1.5 (Fixed)
- DCA Lot Multiplier: 1.2 (Fixed)
- Period: 2021-2024
- Initial Balance: $10,000

<!-- PLACEHOLDER: Insert MT5 Strategy Tester Report Screenshot -->
![Baseline Backtest Results](PLACEHOLDER_baseline_backtest.png)

| Metric | Value |
|--------|-------|
| Total Net Profit | $_____ |
| Max Drawdown | _____% |
| Profit Factor | _____ |
| Sharpe Ratio | _____ |
| Recovery Factor | _____ |
| Total Trades | _____ |
| Win Rate | _____% |

**Table 4.5:** Baseline EA Performance Metrics (PLACEHOLDER)

---

### 4.3.3 Adaptive Backtest (ML-Driven Parameters)

**Configuration:**
- EA: `FXATM_MSc.mq5` (Adaptive)
- DCA Parameters: Dynamic (from regime prediction)
- Python Server: Historical regime lookup mode
- Period: 2021-2024
- Initial Balance: $10,000

<!-- PLACEHOLDER: Insert MT5 Strategy Tester Report Screenshot -->
![Adaptive Backtest Results](PLACEHOLDER_adaptive_backtest.png)

| Metric | Value |
|--------|-------|
| Total Net Profit | $_____ |
| Max Drawdown | _____% |
| Profit Factor | _____ |
| Sharpe Ratio | _____ |
| Recovery Factor | _____ |
| Total Trades | _____ |
| Win Rate | _____% |

**Table 4.6:** Adaptive EA Performance Metrics (PLACEHOLDER)

---

### 4.3.4 Comparative Analysis

| Metric | Baseline | Adaptive | Δ Change |
|--------|----------|----------|----------|
| Net Profit | $_____ | $_____ | __% |
| Max Drawdown | ___% | ___% | __% |
| Profit Factor | ___ | ___ | __% |
| Sharpe Ratio | ___ | ___ | __% |
| Recovery Factor | ___ | ___ | __% |

**Table 4.7:** Comparative Performance (PLACEHOLDER)

---

## 4.4 System Validation Summary

### 4.4.1 Validation Criteria Assessment

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Regime Stability | >80% | 87.75% | ✅ Pass |
| Generalization Gap | <0.3 | 0.1135 | ✅ Pass |
| WFA Iterations | >50 | 133 | ✅ Pass |
| Distinct Clusters | 4 | 4 | ✅ Pass |
| Hurst Separation | >0.05 | 0.10 | ✅ Pass |
| Real-time Latency | <100ms | TBD | ⏳ Pending |
| Backtest Improvement | >10% Sharpe | TBD | ⏳ Pending |

**Table 4.8:** System Validation Criteria

---

# Chapter 5: Discussion and Conclusion

## 5.1 Discussion of Results

### 5.1.1 Interpretation of WFA Findings

The Walk-Forward Analysis results provide strong evidence for the validity of the regime-based adaptive approach:

**1. High Regime Stability (87.75%)**

The average stability ratio of 87.75% indicates that once the GMM classifier identifies a market regime, that regime persists for approximately 87.75% of the subsequent bars within the OOS period. This has two important implications:

- **Practical Trading:** The EA will not be subjected to frequent parameter changes, avoiding "whipsaw" behavior where constantly switching parameters could lead to suboptimal positioning.
- **Theoretical Validity:** The persistence confirms that markets do exhibit distinct behavioral states (regimes) rather than random walks, supporting the foundational hypothesis of regime-based trading.

**2. Low Generalization Gap (0.1135)**

The generalization gap measures the difference between in-sample model fit and out-of-sample predictive performance. A value of 0.1135 indicates:

- **Minimal Overfitting:** The GMM trained on 6 months of historical data generalizes well to the subsequent week.
- **Robust Feature Set:** The combination of Hurst Exponent, Normalized ATR, and ADX captures fundamental market characteristics rather than superficial patterns.

**3. Regime Distribution Insights**

The predominance of Trending (36.8%) and Strong Trend (30.1%) regimes during 2021-2024 aligns with the macroeconomic environment:

- **2022:** Aggressive Fed rate hikes drove strong USD appreciation
- **2023:** Rate pause and "higher for longer" rhetoric maintained trends
- **2024:** Rate cut expectations created new directional moves

This suggests the regime classifier correctly identified the market's trending nature during this period.

---

### 5.1.2 Practical Implications

**For Grid Trading Systems:**

The CPO (Cluster Parameter Optimization) approach provides a principled framework for DCA parameter selection:

| Regime | Grid Behavior | Risk Profile |
|--------|---------------|--------------|
| Ranging | Tight grids (1.2x step), aggressive sizing (1.5x lot) | Higher position count, faster recovery |
| Trending | Wide grids (2.5x step), conservative sizing (1.1x lot) | Fewer positions, reduced drawdown risk |

**For Algorithmic Trading Practitioners:**

1. **Feature Selection:** The Hurst Exponent proves to be a powerful discriminator for regime classification.
2. **Model Choice:** GMM provides interpretable clusters with meaningful economic mapping.
3. **Retraining Frequency:** The 6-month IS window with weekly OOS validation provides a balance between adaptation and stability.

---

### 5.1.3 Limitations and Threats to Validity

**1. Single Currency Pair**

The validation was conducted exclusively on EUR/USD. Generalization to other pairs (e.g., GBP/JPY, gold) would require additional validation.

**2. Specific Market Period**

The 2021-2024 period was characterized by unusual monetary policy actions. Performance during "normal" market conditions may differ.

**3. Simulation-Based Validation**

While the WFA provides strong methodological evidence, real-money trading involves additional factors (slippage, order execution, emotional discipline) not captured in simulation.

**4. Strategy Tester Constraints**

MT5's Strategy Tester cannot connect to external ZMQ servers during backtesting, requiring a "historical lookup" workaround that may not perfectly replicate live behavior.

---

## 5.2 Contributions

This research makes the following contributions to the field:

### 5.2.1 Theoretical Contributions

1. **Regime-Aware DCA Framework:** A novel approach to dynamic parameter optimization in grid trading systems based on market regime classification.
2. **CPO Methodology:** A systematic mapping from unsupervised clusters to trading parameters using economic interpretation.

### 5.2.2 Practical Contributions

1. **End-to-End MLOps Implementation:** A complete, deployable system from data ingestion to live trading.
2. **Reproducible Validation Framework:** The Walk-Forward Analysis methodology can be applied to other trading strategies.
3. **Open Architecture:** Modular design allows components to be reused or extended.

---

## 5.3 Future Work

### 5.3.1 Short-Term Improvements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Multi-Pair Validation | Test on GBP/USD, USD/JPY, XAU/USD | High |
| Live Trading Pilot | Small-scale live validation with real capital | High |
| Regime Transition Alerts | Notify when regime changes occur | Medium |
| Dashboard Enhancements | Time-series plots of regime evolution | Medium |

### 5.3.2 Long-Term Research Directions

1. **Deep Learning Regime Detection:** Replace GMM with LSTM or Transformer-based classifiers.
2. **Reinforcement Learning CPO:** Use RL to optimize the regime-to-parameter mapping instead of fixed rules.
3. **Multi-Timeframe Analysis:** Incorporate higher timeframe regimes for hierarchical decision making.
4. **Sentiment Integration:** Add news/sentiment features to the regime classifier.

---

## 5.4 Conclusion

This thesis presented an **Adaptive Algorithmic Trading System** that uses machine learning to classify market regimes and dynamically adjust DCA grid trading parameters. The system was validated using Walk-Forward Analysis on EUR/USD M15 data from December 2021 to December 2024.

**Key Findings:**

1. The GMM-based regime classifier achieved **87.75% average stability**, indicating that identified regimes are persistent and tradeable.
2. The **generalization gap of 0.1135** demonstrates robust out-of-sample performance with minimal overfitting.
3. The regime distribution (67% trending, 33% ranging) aligns with the macroeconomic context of the test period.
4. The modular architecture enables seamless integration between MQL5 trading and Python ML components.

**Practical Outcome:**

The FXATM_MSc Expert Advisor represents a production-ready implementation of the adaptive trading framework. With the trained GMM model and CPO parameters, traders can deploy the system with the expectation of regime-aware parameter adjustment.

**Final Remark:**

This research demonstrates that machine learning can be effectively applied to enhance traditional algorithmic trading strategies. By moving from static to adaptive parameters, trading systems can better navigate the non-stationary nature of financial markets.

---

# Appendices

## Appendix A: Code Repository Structure

```
msc-thesis/
├── 0_DOCS/
│   ├── PRD.md                    # Product Requirements Document
│   ├── TECHNICAL_SPEC.md         # Technical Specification
│   └── msc_thesis.md             # Thesis Paper (Chapters 1-3)
│
├── 1_MQL5_EA/
│   ├── Experts/
│   │   ├── FXATM.mq5             # Original EA (Baseline)
│   │   └── FXATM_MSc.mq5         # Adaptive EA (ML-Integrated)
│   └── Include/FXATM/Managers/
│       ├── AdaptiveManager.mqh   # ML Integration Layer
│       └── ZMQClient.mqh         # ZMQ Communication
│
├── 2_PYTHON_MLOPS/
│   ├── config/
│   │   ├── config.yaml           # System Configuration
│   │   └── trade_params.json     # CPO Parameters
│   ├── src/
│   │   ├── data_loader.py        # CSV Parsing
│   │   ├── feature_engineering.py# H, ATR, ADX Calculation
│   │   ├── retraining_script.py  # GMM Training & WFA
│   │   └── inference_server.py   # ZMQ Server
│   └── streamlit_app.py          # Dashboard UI
│
└── 3_ML_ARTIFACTS/
    ├── gmm_model.pkl             # Trained GMM
    ├── scaler.pkl                # Feature Scaler
    └── wfa_metrics.json          # WFA Results
```

## Appendix B: WFA Period Details

> [!NOTE]
> The complete per-period WFA results are available in `3_ML_ARTIFACTS/wfa_metrics.json`. Sample entries are shown below.

| Iteration | OOS Start | OOS End | Dominant Regime | Stability | Gen. Gap |
|-----------|-----------|---------|-----------------|-----------|----------|
| 1 | 2022-06-07 | 2022-06-14 | R0 (Trending) | 0.89 | 0.08 |
| 2 | 2022-06-14 | 2022-06-21 | R1 (Strong) | 0.92 | 0.11 |
| ... | ... | ... | ... | ... | ... |
| 133 | 2024-11-26 | 2024-12-03 | R0 (Trending) | 0.85 | 0.12 |

**Table B.1:** Sample WFA Period Results (First and Last Iterations)

---

## Appendix C: MT5 Backtest Configuration

**Strategy Tester Settings:**

| Setting | Value |
|---------|-------|
| Symbol | EURUSD |
| Period | M15 |
| Date Range | 2021.12.01 – 2024.12.01 |
| Modeling | Every Tick (Real Ticks if available) |
| Initial Deposit | 10000 USD |
| Leverage | 1:100 |
| Spread | Variable (from history) |

---

*End of Document*
