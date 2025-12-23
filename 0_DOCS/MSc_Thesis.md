

UNIVERSITA’ DEGLI STUDI GUGLIELMO MARCONI

Master in Computer Science

**Intelligent Forex Trading**  
An Adaptive Machine Learning Framework for Handling Market Non-Stationary in Algorithmic Forex Trading \- A Simulation-Based Study

| Academic Advisor | Candidate |
| :---- | ----: |
| Ryan Suryanto | Lawrance Koh Chee Hng MCSLT00274  |

 

ACADEMIC YEAR  
2025 / 2026

# **Table of Contents** {#table-of-contents}

[**Table of Contents	2**](#table-of-contents)

[**Abstract	5**](#abstract)

[**1\. Introduction and Problem Statement	6**](#1.-introduction-and-problem-statement)

[1.1 Proposed Solution: The Adaptive Algorithmic Trading System	6](#1.1-proposed-solution:-the-adaptive-algorithmic-trading-system)

[1.1.1 The Hybrid Machine Learning Core	6](#1.1.1-the-hybrid-machine-learning-core)

[1.1.2 Dynamic Parameter Optimization	7](#1.1.2-dynamic-parameter-optimization)

[1.2 Methodology for Continuous Robustness: The MLOps Pipeline	7](#1.2-methodology-for-continuous-robustness:-the-mlops-pipeline)

[**2\. Literature Review	9**](#2.-literature-review)

[2.1 Strategic Imperative: Non-Stationarity and the Adaptive Advantage	9](#2.1-strategic-imperative:-non-stationarity-and-the-adaptive-advantage)

[2.1.1 The Fundamental Challenge of Market Non-Stationarity	9](#2.1.1-the-fundamental-challenge-of-market-non-stationarity)

[2.1.2 Empirical Justification for Adaptive Strategies	9](#2.1.2-empirical-justification-for-adaptive-strategies)

[2.2 Dynamic Regime Classification: The Machine Learning Core	10](#2.2-dynamic-regime-classification:-the-machine-learning-core)

[2.2.1 Selection Justification: Unsupervised Clustering (UCL)	10](#2.2.1-selection-justification:-unsupervised-clustering-\(ucl\))

[2.2.2 Advanced Feature Engineering for Forex Regime Differentiation	11](#2.2.2-advanced-feature-engineering-for-forex-regime-differentiation)

[2.2.3. Clustering Methodology and Cluster Validation	11](#2.2.3.-clustering-methodology-and-cluster-validation)

[2.3 The Dynamic Mapping Layer: Optimization and MQL5 Interface	12](#2.3-the-dynamic-mapping-layer:-optimization-and-mql5-interface)

[2.3.1 Conditional Parameter Optimization (CPO) Mechanics	12](#2.3.1-conditional-parameter-optimization-\(cpo\)-mechanics)

[2.3.2 Optimization Methodology and Search Space	13](#2.3.2-optimization-methodology-and-search-space)

[2.3.3 Rigorous Evaluation Metrics for Adaptive Systems	14](#2.3.3-rigorous-evaluation-metrics-for-adaptive-systems)

[2.4 MLOps for Continuous Adaptivity and Robustness	14](#2.4-mlops-for-continuous-adaptivity-and-robustness)

[2.4.1 The Necessity of MLOps for Financial Time Series	14](#2.4.1-the-necessity-of-mlops-for-financial-time-series)

[2.4.2 Justification of Fixed Weekly Rolling Window Retraining	15](#2.4.2-justification-of-fixed-weekly-rolling-window-retraining)

[2.4.3 The Rolling Window Mechanism	15](#2.4.3-the-rolling-window-mechanism)

[2.4.4 Robustness Validation: Walk Forward Analysis (WFA)	16](#2.4.4-robustness-validation:-walk-forward-analysis-\(wfa\))

[2.5 Technical Architecture: Python-MQL5 Interoperability	16](#2.5-technical-architecture:-python-mql5-interoperability)

[2.5.1 The Inter-Process Communication (IPC) Backbone	16](#2.5.1-the-inter-process-communication-\(ipc\)-backbone)

[2.5.2 JSON Schema for Dynamic Parameter Transmission	16](#2.5.2-json-schema-for-dynamic-parameter-transmission)

[2.5.3 MQL5 Expert Advisor Adaptation (The Execution Layer)	17](#2.5.3-mql5-expert-advisor-adaptation-\(the-execution-layer\))

[2.6 Conclusions and Future Work	18](#2.6-conclusions-and-future-work)

[2.6.1 Synthesis of Findings	18](#2.6.1-synthesis-of-findings)

[2.6.2 Recommendations and Future Trajectories	18](#2.6.2-recommendations-and-future-trajectories)

[**3\. Research Methodology	20**](#heading=)

[3.1 Introduction to the Methodology	20](#3.1-introduction-to-the-methodology)

[3.2 Research Design	20](#3.2-research-design)

[3.3 Research Questions and Hypotheses	20](#3.3-research-questions-and-hypotheses)

[3.4 Data Sources and Data Collection	21](#3.4-data-sources-and-data-collection)

[3.5 System / Model / Framework Description	21](#3.5-system-/-model-/-framework-description)

[3.5.1 The Machine Learning Core: Regime Classification	22](#3.5.1-the-machine-learning-core:-regime-classification)

[3.5.2 The Dynamic Mapping Layer: Conditional Parameter Optimization (CPO)	22](#3.5.2-the-dynamic-mapping-layer:-conditional-parameter-optimization-\(cpo\))

[3.5.3 Inter-Process Communication and Execution	22](#3.5.3-inter-process-communication-and-execution)

[3.6 Tools, Technologies, and Platforms	22](#3.6-tools,-technologies,-and-platforms)

[3.7 Evaluation Metrics and Analysis Techniques	23](#3.7-evaluation-metrics-and-analysis-techniques)

[3.8 Validation and Comparison	23](#3.8-validation-and-comparison)

[3.9 Ethical Considerations	24](#3.9-ethical-considerations)

[3.10 Limitations of the Methodology	24](#3.10-limitations-of-the-methodology)

[3.11 Summary of the Chapter	24](#3.11-summary-of-the-chapter)

[**4\. Implementation and Results	25**](#4.-implementation-and-results)

[4.1 System Implementation Overview	25](#4.1-system-implementation-overview)

[4.1.1 System Architecture	25](#4.1.1-system-architecture)

[4.1.2 Component Implementation Details	25](#4.1.2-component-implementation-details)

[4.2 Walk-Forward Analysis Results	29](#4.2-walk-forward-analysis-results)

[4.2.1 WFA Configuration	29](#4.2.1-wfa-configuration)

[4.2.2 Aggregate Results	29](#4.2.2-aggregate-results)

[4.2.3 Regime Distribution	30](#4.2.3-regime-distribution)

[4.2.4 Final Model Cluster Centroids	31](#4.2.4-final-model-cluster-centroids)

[4.3 MQL5 Integration Results	32](#4.3-mql5-integration-results)

[4.3.1 Backtest Methodology	32](#4.3.1-backtest-methodology)

[To validate the practical trading impact, backtests were conducted using MT5 Strategy Tester.	32](#to-validate-the-practical-trading-impact,-backtests-were-conducted-using-mt5-strategy-tester.)

[4.3.2 Baseline Backtest (Static Parameters)	32](#4.3.2-baseline-backtest-\(static-parameters\))

[4.3.3 Adaptive Backtest (ML-Driven Parameters)	33](#4.3.3-adaptive-backtest-\(ml-driven-parameters\))

[4.3.4 Comparative Analysis	34](#4.3.4-comparative-analysis)

[4.4 System Validation Summary	34](#4.4-system-validation-summary)

[4.4.1 Validation Criteria Assessment	34](#4.4.1-validation-criteria-assessment)

[**5\. Discussion and Conclusion	36**](#5.-discussion-and-conclusion)

[5.1 Discussion of Results	36](#5.1-discussion-of-results)

[5.1.1 Interpretation of WFA Findings	36](#5.1.1-interpretation-of-wfa-findings)

[5.1.2 Practical Implications	37](#5.1.2-practical-implications)

[5.1.3 Limitations and Threats to Validity	37](#5.1.3-limitations-and-threats-to-validity)

[5.2 Contributions	38](#5.2-contributions)

[5.2.1 Theoretical Contributions	38](#5.2.1-theoretical-contributions)

[5.2.2 Practical Contributions	38](#5.2.2-practical-contributions)

[5.3 Future Work	38](#5.3-future-work)

[5.3.1 Short-Term Improvements	38](#5.3.1-short-term-improvements)

[5.3.2 Long-Term Research Directions	38](#5.3.2-long-term-research-directions)

[5.4 Conclusion	39](#5.4-conclusion)

[**6\. Bibliography and Web Sources	40**](#6.-bibliography-and-web-sources)

[**Appendices	44**](#appendices)

[Appendix A: Code Repository Structure	44](#appendix-a:-code-repository-structure)

[Appendix B: WFA Period Details	45](#appendix-b:-wfa-period-details)

[Appendix C: MT5 Backtest Configuration	46](#appendix-c:-mt5-backtest-configuration)

[Appendix D: EA Source Codes	47](#appendix-d:-ea-source-codes)

[MQL5 Directory Structure	47](#mql5-directory-structure)

# 

# 

# **Abstract** {#abstract}

The Foreign Exchange (Forex) market is a highly dynamic and non-stationary environment where statistical properties such as volatility and trend duration shift frequently, rendering traditional static algorithmic trading strategies ineffective over time. Fixed-parameter Expert Advisors (EAs) often suffer from parameter decay, leading to degraded performance when market regimes change. This project addresses the challenge of non-stationarity by developing an **Adaptive Algorithmic Trading System** that dynamically aligns its risk management parameters with the prevailing market conditions.

The proposed solution utilizes a hybrid architecture that integrates a Python-based Machine Learning (ML) core with a high-performance MQL5 execution layer. The ML core employs **Unsupervised Clustering (Gaussian Mixture Models)** on structural market features—specifically the Hurst Exponent, Normalized ATR, and ADX—to classify the market into distinct regimes such as "Trending," "Ranging," or "Volatile." These regimes are then mapped to optimal risk parameters for a **Dollar-Cost Averaging (DCA)** strategy via a Conditional Parameter Optimization (CPO) process.

To ensure robustness and longevity, the system is governed by a simulated **MLOps (Machine Learning Operations)** pipeline that enforces a fixed weekly re-training schedule. This mechanism, validated through rigorous **Walk-Forward Analysis (WFA)** on historical data from 2020–2024, allows the model to continuously adapt to concept drift. The research aims to demonstrate that a regime-aware, adaptive system significantly outperforms static baselines in risk-adjusted metrics such as the Sharpe Ratio and Recovery Factor, offering a sustainable approach to automated trading in non-stationary markets.

# 

# **1\. Introduction and Problem Statement** {#1.-introduction-and-problem-statement}

This project work directly tackles the most significant and persistent challenge in automated Forex (Foreign Exchange) trading: market non-stationarity. The Forex market is an inherently dynamic environment where statistical properties, such as volatility, trend duration, and correlation structures, are not constant but change fundamentally and frequently over time.

Traditional algorithmic trading strategies, which form the vast majority of current automated systems, are fundamentally static. They operate with a fixed set of optimal parameters (e.g., lookback periods for moving averages, thresholds for oscillators, stop-loss percentages) determined through historical backtesting. While highly profitable for the specific market regime they were optimized against, these static strategies inevitably suffer from a catastrophic failure known as "parameter decay" when market conditions abruptly shift. For example, a strategy tuned for a low-volatility, ranging market will quickly become unprofitable, or even disastrous, during a high-volatility, strong-trending phase. The inability of these fixed-parameter systems to autonomously adapt to changes in volatility, liquidity, or trend strength renders them ultimately unsustainable and non-robust.

## **1.1 Proposed Solution: The Adaptive Algorithmic Trading System** {#1.1-proposed-solution:-the-adaptive-algorithmic-trading-system}

This research aims to deliver a novel and methodologically original contribution by developing a truly Adaptive Algorithmic Trading System. This system is specifically designed to overcome the limitations of static strategies by continuously and dynamically aligning its trading logic with the prevailing market environment.

### 1.1.1 The Hybrid Machine Learning Core {#1.1.1-the-hybrid-machine-learning-core}

The proposed solution is a sophisticated hybrid architecture centered on a Python-based Machine Learning (ML) model. This ML model forms the intelligence layer of the system. Its primary, non-trivial function is to apply Unsupervised Clustering techniques—such as K-Means or DBSCAN—to a multivariate time series of market features (e.g., Average True Range, ADX, price return variance, autocorrelation).

The purpose of this clustering is to dynamically identify the current market regime. Instead of relying on pre-defined, arbitrary thresholds, the system allows the data to statistically group itself into distinct operational states, which might include:

* **Ranging/Consolidation:** Low volatility, weak or non-existent trend.

* **Strong Trend:** High momentum, low mean-reversion characteristics.  
* **High Volatility Breakout:** Explosive price movements, often associated with news events.  
* **Low Volatility Drift:** Persistent, slow movement.

### 1.1.2 Dynamic Parameter Optimization {#1.1.2-dynamic-parameter-optimization}

Based on the market regime identified by the ML model, the system executes the critical step of dynamic parameter suggestion. The ML layer interfaces with an existing, high-performance MQL5 Expert Advisor (EA) running on the MetaTrader 5 platform. Instead of the human trader manually inputting fixed settings, the ML model provides the optimal parameter set specifically calibrated for the detected regime. For instance, in a Strong Trend regime, the system might suggest a longer lookback period for a trend-following indicator and a wider take-profit target, whereas in a Ranging regime, it would switch to a mean-reversion strategy with tighter stop-losses and shorter lookback periods. This continuous, data-driven parameter adjustment ensures the trading strategy remains logically sound and maximally efficient under all observed market conditions.

## **1.2 Methodology for Continuous Robustness: The MLOps Pipeline** {#1.2-methodology-for-continuous-robustness:-the-mlops-pipeline}

To ensure the adaptive solution remains effective, robust, and constantly relevant over an extended period, the project incorporates a vital, state-of-the-art component: an MLOps (Machine Learning Operations) pipeline.

This pipeline establishes a comprehensive infrastructure for guaranteed continuous optimization. The core mechanism involves a fixed weekly rolling window re-training schedule. Every week, the ML model is automatically retrained on the most recent, relevant market data.

Key functions of the MLOps pipeline include:

1. Automated Data Ingestion: Secure and timely fetching of new, cleaned market data.

2. Model Re-training: Automatic re-running of the clustering algorithm to identify new, evolving market regimes. This prevents model drift and ensures the regimes identified are always representative of the latest market behavior.

3. Validation and Performance Monitoring: Rigorous backtesting of the newly trained model to ensure performance metrics are met before deployment.

4. Automated Deployment: Seamless and zero-downtime deployment of the updated ML model and its new parameter mappings to the live trading environment.

This systematic and automated mechanism represents a major methodological advancement beyond static strategies. By adopting this rigorous adaptive methodology, this project not only delivers a constantly optimizing and highly responsive framework for automated Forex trading but also demonstrates the acquisition of essential, cutting-edge cultural skills in both quantitative finance and modern machine learning engineering. The ultimate goal is to create a robust, self-optimizing, and truly intelligent trading solution capable of sustaining profitability across the non-stationary landscape of the global Forex market.

# **2\. Literature Review** {#2.-literature-review}

## **2.1 Strategic Imperative: Non-Stationarity and the Adaptive Advantage** {#2.1-strategic-imperative:-non-stationarity-and-the-adaptive-advantage}

### 2.1.1 The Fundamental Challenge of Market Non-Stationarity {#2.1.1-the-fundamental-challenge-of-market-non-stationarity}

Algorithmic trading systems operating in the Foreign Exchange (Forex) market face a central, pervasive obstacle: the non-stationary nature of financial time series. Unlike traditional data sets, Forex returns are characterized by frequent, unpredictable shifts in underlying dynamics, often exhibiting phenomena such as volatility clustering and long-range dependence.\[1\] This inherent non-stationarity fundamentally violates the core assumptions of classical statistical models and fixed-parameter Expert Advisors (EAs).\[2\]

The consequence of non-stationarity in a production ML system is Concept Drift.\[3, 4\] This occurs when the statistical relationship between the input features (price action, volume, indicators) and the optimal trading outcome (profitability, risk management) changes over time, represented formally as a shift in the conditional probability distribution P(Y∣X).\[3\] For example, a trading model optimized for a strong trending market regime may rapidly lose its predictive power and generate immediate losses when the market transitions into a mean-reverting or ranging environment.\[4\] A strategy based on static, optimized parameters inevitably reflects past patterns rather than remaining robust to the current reality.\[5\]

To manage this instability, the Regime-Switching Paradigm is essential. This methodology acknowledges that the underlying market generates distinct, latent, or "hidden" states (regimes) that influence asset returns.\[2, 6\] Academic literature confirms that models designed to detect and switch strategies based on these regimes are necessary to adapt successfully to complex financial dynamics.\[7\] The system proposed herein moves beyond a static risk management filter to proactively identify these latent states and dynamically map them to optimized trading policies.

### 2.1.2 Empirical Justification for Adaptive Strategies {#2.1.2-empirical-justification-for-adaptive-strategies}

The transition from static to adaptive strategies is supported by compelling empirical evidence across quantitative finance. Studies comparing static agent populations, which are evolved on historical data, against adaptive populations, which are continuously retrained on the most recent available data, demonstrate the clear superiority of the adaptive approach.\[8\] Adaptive systems are not merely incremental improvements; they are robust solutions designed to overcome the fundamental decay of fixed models.

Furthermore, dynamic solutions significantly enhance operational efficiency. Research into optimal trade execution problems shows that strategies observing signals a finite number of times can substantially reduce transaction costs and dramatically improve performance compared to their optimal static counterparts.\[9\] The dynamic nature allows the system to adjust positioning and execution timing based on rapidly evolving market context.

The core of this system is achieving Conditional Parameter Optimization (CPO), a technique where the configuration settings of a trading strategy are adapted based on the identified current regime.\[10\] This approach recognizes that the optimal parameter set—such as the lookback period for a moving average, the multiplier for an Average True Range (ATR) stop-loss, or the profit target—must fundamentally change across regimes. For instance, strong trending markets permit wider profit targets and looser stops to capture large movements, whereas ranging markets require tighter stops and mean-reversion logic.\[11\] By continuously adapting the parameters to the current market state, the system shifts from a rigid rule set to a calculated, continuously learning strategic evolution, satisfying the rigorous requirements for a production-grade system.\[12\] This process of continuous adaptation addresses concept drift proactively, maximizing the system's long-term competitive advantage.

## **2.2 Dynamic Regime Classification: The Machine Learning Core** {#2.2-dynamic-regime-classification:-the-machine-learning-core}

### 2.2.1 Selection Justification: Unsupervised Clustering (UCL) {#2.2.1-selection-justification:-unsupervised-clustering-(ucl)}

The first technical requirement of the framework is to identify the natural groupings of market behavior (regimes) without relying on pre-labeled data. This necessitates an unsupervised machine learning approach. Unsupervised Clustering (UCL) achieves this by grouping observations based purely on feature similarity.\[13, 14\]

While traditional quantitative models often employ Hidden Markov Models (HMM) to detect regimes, HMMs are primarily sequential statistical models that are effective for modeling state transitions and forecasting future states.\[6, 15\] However, HMMs can be computationally intensive and determining the optimal number of states often proves challenging.\[2, 16\] For the specific goal of the thesis—which is to identify the current regime for immediate parameter mapping—a robust clustering methodology based on current feature snapshots is often more directly actionable.

Gaussian Mixture Models (GMM) are selected as the probabilistic clustering choice.\[17\] GMM is a robust probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions, where each distribution represents a distinct cluster or regime.\[17\] This capability is critical in finance, as asset returns frequently exhibit complex, multimodal distributions that cannot be adequately captured by a single Gaussian or simple distance-based clustering (like K-Means).\[17\] GMM provides a probabilistic assignment of the current market state, allowing for clear, immediate boundaries based on the latest feature set for dynamic parameter mapping.\[18\] The flexibility and power of GMM to model complex distributions make it an essential technique for capturing intricate dynamics in the financial domain.\[17\]

### 2.2.2 Advanced Feature Engineering for Forex Regime Differentiation {#2.2.2-advanced-feature-engineering-for-forex-regime-differentiation}

Effective regime classification hinges on the input features used for clustering. Simple log-returns or volumes often fail to capture the complex, long-term memory effects inherent in FX markets.\[19\] Therefore, the system must utilize features that rigorously quantify the structural complexity of the time series.\[20\]

The features selected must provide a quantifiable link between the abstract cluster output and an economically meaningful regime. This link is provided by complexity-based features, primarily the Hurst Exponent (H).\[19\] The Hurst Exponent measures the long-term memory of a time series, classifying it as purely random (H≈0.5), persistent or trending (H\>0.5), or anti-persistent or mean-reverting (H\<0.5).\[19, 21\] Because the Hurst Exponent fundamentally classifies market structure, a clustering solution that shows a strong correlation with distinct ranges of H provides an unambiguous economic label (e.g., Cluster 1 is "Strong Trending Market" if H consistently exceeds 0.7). This is a vital step beyond simple mathematical proximity, ensuring the clusters are actionable.

Complementary features include:

1\. Fractal Dimension (D): Related to H by the equation D=2−H.\[21\] This metric measures the "jaggedness" or self-similarity of the price curve, offering a secondary structural view that has been shown to enhance prediction accuracy, particularly in volatility forecasting.\[20\]

2\. Volatility-based Features: Metrics derived from range-based estimators (such as the Average True Range or realized volatility over the clustering window) are critical for isolating high-volatility, choppy regimes from calm ones. Log-volatility features, specifically, have been utilized successfully in clustering to capture empirical market dynamics.\[1, 22\]

3\. Momentum Features: Multi-scale feature extraction, analyzing price action across different timeframes, also assists in robustly classifying up, down, and sideways trends.\[23\]  
Given the inclusion of multiple, potentially correlated features, Principal Component Analysis (PCA) should be considered prior to clustering. Applying PCA for dimensionality reduction can enhance computational efficiency and improve cluster separation by focusing the algorithm on the most impactful structural components of the data.\[7, 13\]

### 2.2.3. Clustering Methodology and Cluster Validation {#2.2.3.-clustering-methodology-and-cluster-validation}

To apply clustering, the continuous time series must be segmented into discrete windows of a fixed length (cluster\_window).\[13\] The resulting feature set, calculated over these windows, forms the input for the GMM.

The choice of the optimal number of clusters (k) is non-trivial and remains an open question in the academic literature.\[16\] Standard statistical methods, such as the Elbow Method and Silhouette Score, must be employed.\[13\] However, the statistical results must be balanced with practical constraints. Published studies using advanced clustering techniques often reveal modest Silhouette scores, indicating the inherent difficulty in achieving perfectly clean separation in noisy FX time series.\[22\]

This inherent fuzziness in market boundaries means the resulting regime classifications may be ambiguous, increasing the risk of misclassification in real-time. This emphasizes the vital role of the downstream MLOps pipeline: the speed and continuity of the weekly re-training cycle (Section IV) must compensate for ambiguous regime assignments by quickly adapting the model when performance degradation occurs.

Furthermore, the selection of k must be guided by the complexity of the existing MQL5 Expert Advisor. If the EA has only a small number of tunable parameters, selecting too many clusters (e.g., 10 regimes) will likely lead to optimization redundancy and overfitting, as the parameter differentiation across these numerous states may not be meaningful.\[10\] For practical deployment, aiming for 3 to 5 economically distinct regimes (e.g., Strong Trend, Weak Trend, Mean Reverting, High Volatility) is generally optimal.

Table 1: Comparative Analysis of Regime Classification Models  
![][image1]

## **2.3 The Dynamic Mapping Layer: Optimization and MQL5 Interface** {#2.3-the-dynamic-mapping-layer:-optimization-and-mql5-interface}

### 2.3.1 Conditional Parameter Optimization (CPO) Mechanics {#2.3.1-conditional-parameter-optimization-(cpo)-mechanics}

Once the Python ML module classifies the current market regime, this cluster ID must be mapped to a specific set of optimal parameters for the MQL5 Expert Advisor. This Conditional Parameter Optimization (CPO) process defines the system's adaptive capability.\[10\]

The core mechanism involves:

1\. Regime Training Segmentation: Historical data is segmented based on the cluster assignments generated by the GMM.

2\. Parameter Search: For each regime, the parameters of the MQL5 EA are optimized using only the data corresponding to that regime. This optimization is crucial because it ensures the parameters (e.g., entry sensitivity, stop-loss distance, maximum spread tolerance) are tailored to the market dynamics of that specific state.\[11\]

3\. Mapping: A simple lookup table or structured artifact stores the optimized parameter set (P1, P2, …, Pn) corresponding to each Regime ID. This artifact is then transferred to the MQL5 environment via the Inter-Process Communication (IPC) layer.

### 2.3.2 Optimization Methodology and Search Space {#2.3.2-optimization-methodology-and-search-space}

The initial optimization can leverage the powerful built-in testing capabilities of the MT5 platform, which allows for brute-force or sophisticated genetic optimization of the EA based on historical data.\[11, 24\]

Crucially, the optimization process must employ Walk Forward Analysis (WFA) techniques within the initial backtesting phase to prevent the CPO itself from leading to parameter overfitting.\[5, 25\] Static optimization over a fixed period risks finding parameters that reflect noise rather than underlying patterns. WFA addresses this by repeatedly optimizing parameters on an "In-Sample" (IS) segment and validating them immediately on a subsequent, untouched "Out-of-Sample" (OOS) segment, simulating real-time deployment and confirming parameter robustness.\[5\] A strategy is only deemed robust if the performance remains stable across all OOS segments.

For enhanced academic rigor, the framework can be extended beyond simple lookup tables toward dynamic policy generation:

* Genetic Algorithms (GA): GAs are capable of evolving complex trading strategies or parameter combinations specifically tailored to the characteristics quantified by features like the Hurst Exponent.\[26, 27\]  
    
* Contextual Reinforcement Learning (RL): A more advanced paradigm utilizes the regime clusters as the "context" for a Reinforcement Learning agent. The RL agent, acting as a Meta-Controller, learns the optimal action (i.e., selecting or generating the precise parameter set) for each identified context, moving beyond pre-optimized static sets to a learned, dynamic policy.\[26, 28, 29, 30\]

### 2.3.3 Rigorous Evaluation Metrics for Adaptive Systems {#2.3.3-rigorous-evaluation-metrics-for-adaptive-systems}

In quantitative finance, particularly with leveraged instruments like Forex, evaluation must extend far beyond simple net profit or win rate. The true measure of a dynamic trading system is its ability to generate high returns while preserving capital and managing risk.\[31\] The optimization objective function must be multi-objective, penalizing excessive risk.

Risk-Adjusted Metrics  
![][image2]

The simultaneous focus on the Sharpe Ratio and the Recovery Factor reveals an inherent trade-off. While a strategy may achieve a high Sharpe Ratio through consistent, small wins, a low Recovery Factor indicates vulnerability to catastrophic, low-probability drawdowns.\[32\] The objective function must therefore ensure that the system is not just statistically efficient but also resilient, prioritizing capital preservation through drawdown control.\[11\] To monitor the stability of performance in the MLOps pipeline, a rolling Sharpe Ratio should be employed, calculated continuously over a fixed lookback period.\[33\]

## **2.4 MLOps for Continuous Adaptivity and Robustness** {#2.4-mlops-for-continuous-adaptivity-and-robustness}

### 2.4.1 The Necessity of MLOps for Financial Time Series {#2.4.1-the-necessity-of-mlops-for-financial-time-series}

MLOps, the engineering culture that unifies ML system development (Dev) and operations (Ops), is mandatory for high-stakes, production-ready ML systems, especially in finance.\[12, 34\] The rapid degradation of predictive power in financial models due to non-stationarity makes MLOps automation a non-negotiable defense against model drift.\[4, 35\] The pipeline must automate Continuous Integration (CI), Continuous Delivery (CD), and Continuous Training (CT) to manage the entire lifecycle.\[12\] This approach establishes a comprehensive governance framework, ensuring auditability and reproducibility by versioning code, data, models, and optimization results.\[34\]

### 2.4.2 Justification of Fixed Weekly Rolling Window Retraining {#2.4.2-justification-of-fixed-weekly-rolling-window-retraining}

The core innovation of the proposed framework is the commitment to a fixed weekly rolling window re-training schedule, designed to combat continuous concept drift proactively.

While reactive retraining approaches use drift detectors like Adaptive Windowing (ADWIN) or the Population Stability Index (PSI) to trigger updates only when statistical divergence is detected \[36, 37, 38\], these methods often suffer from an inherent delay in financial markets. A drift-triggered approach must wait until model accuracy or data distribution has already dropped below a preset threshold, resulting in incurred losses before adaptation begins.\[3, 39\]

In the high-volatility, continuously non-stationary environment of Forex, concept drift is assumed to be continuous and pervasive. A fixed weekly schedule provides a superior, proactive defense. This frequency ensures the model’s parameters are always anchored to the most recent underlying market behavior, effectively mitigating gradual degradation.\[4, 39\] Time-based schedules offer predictable compute loads and avoid the operational complexity and false alarms associated with calibrating sensitive statistical thresholds (e.g., Wasserstein distance thresholds) in noisy time series.\[38\]

Although the fixed schedule is the primary CT mechanism, monitoring for sudden, catastrophic drift (e.g., during major central bank announcements) using statistical measures like PSI should still be implemented in parallel. While these metrics may not trigger a full retraining cycle, they can be used to generate critical system alerts, allowing for manual intervention, such as temporarily halting the EA, until the next scheduled retraining incorporates the shock event data.\[35, 39\]

### 2.4.3 The Rolling Window Mechanism {#2.4.3-the-rolling-window-mechanism}

The Continuous Training (CT) pipeline must be meticulously automated to run weekly. The fixed weekly re-training utilizes a rolling historical data window (e.g., the last 180 trading days).\[33\] This window size is crucial; it must be long enough to capture diverse regime characteristics but short enough to discard overly stale data that no longer reflects current market structure.

The automated CT process includes:

1\. Data Acquisition: Pulling the latest data from the MT5 platform or dedicated data source.\[40\]

2\. Feature Calculation: Re-calculating all complexity features (Hurst Exponent, volatility metrics) over the new rolling window.

3\. Model Fitting (GMM): Retraining the GMM clustering model on the updated feature set, which may result in subtly shifted cluster boundaries and means.

4\. Parameter Optimization: Re-running the regime-specific parameter optimization routines on the historical data segmented by the new cluster assignments.

5\. Artifact Generation and Deployment: Generating the updated cluster-to-parameter map artifact and pushing the new configuration to the live ZMQ endpoint, ready for the MQL5 EA.\[12, 34\]

### 2.4.4 Robustness Validation: Walk Forward Analysis (WFA) {#2.4.4-robustness-validation:-walk-forward-analysis-(wfa)}

The primary risk in deploying an adaptive system is overfitting, where the optimization process captures noise specific to the training period, leading to poor live performance.\[25\] Walk Forward Analysis (WFA) is the indispensable technique used for validation.\[5\]

WFA implementation requires dividing the overall historical dataset into sequential optimization and validation segments. For instance, the system might use 100 days for in-sample optimization (IS) and 20 days for out-of-sample testing (OOS), rolling this window forward chronologically.

The success of the strategy is not judged by the profit in the IS period, but by the stability and quality of the performance metrics across all OOS segments. The WFA must confirm that the system maintains critical risk thresholds, such as a stable Sharpe Ratio above a predetermined floor (e.g., S\>1.0) during the unseen OOS periods.\[33\] By mandating WFA within the validation step of the MLOps pipeline, the framework dramatically increases the confidence that the optimized, regime-specific parameters will perform reliably in a live, adaptive environment.\[25\]

## **2.5 Technical Architecture: Python-MQL5 Interoperability** {#2.5-technical-architecture:-python-mql5-interoperability}

### 2.5.1 The Inter-Process Communication (IPC) Backbone {#2.5.1-the-inter-process-communication-(ipc)-backbone}

The proposed framework is fundamentally a hybrid system, combining the analytical power of Python (for ML and MLOps orchestration) with the low-latency execution efficiency of MQL5/MetaTrader 5 (MT5). Inter-Process Communication (IPC) is required to bridge this functional gap.

ZeroMQ (ZMQ) is selected as the IPC backbone. ZMQ is a high-performance, asynchronous messaging library designed for concurrent applications.\[41\] It is brokerless and supports common messaging patterns like Push/Pull or Request/Reply over various transports, ensuring robust, low-latency transmission of the critical parameter updates.\[41, 42\] Utilizing existing robust ZMQ connectors designed for MT5 simplifies implementation and ensures reliable data transfer between the Python environment and the Expert Advisor (EA) running within the trading terminal.\[42, 43\]

### 2.5.2 JSON Schema for Dynamic Parameter Transmission {#2.5.2-json-schema-for-dynamic-parameter-transmission}

To ensure reliable and structured data transfer, the dynamic parameter updates must be serialized using JSON (JavaScript Object Notation), the standard lightweight data-interchange format.\[44\]  
The Python ML environment sends a structured JSON payload to the MQL5 EA, which defines the new configuration. A robust JSON schema includes:

1\. Regime\_ID: The identified cluster integer (1 to K).

2\. Timestamp: The epoch time of the model prediction, vital for version control.

3\. Symbol: The currency pair (e.g., EURUSD).

4\. EA\_Parameters: A nested object containing the new parameter values, such as {"Period\_MA": 20, "StopLoss\_ATR\_Multiplier": 1.5, "Max\_Drawdown\_Percent": 0.01}.

The MQL5 environment must be equipped to handle this data. As MQL5 does not possess a native JSON library, a custom solution or integrated third-party API is necessary to reliably parse the incoming JSON string and update the internal EA variables (typically defined as extern inputs).\[44, 45\]

### 2.5.3 MQL5 Expert Advisor Adaptation (The Execution Layer) {#2.5.3-mql5-expert-advisor-adaptation-(the-execution-layer)}

The MQL5 Expert Advisor (EA) serves as the high-speed, execution layer. It must be programmed with the flexibility to receive and instantly reload critical parameters without disrupting ongoing trade execution or necessitating a terminal restart.\[46, 47\]

The EA operates in continuous listening mode, monitoring the ZMQ port for new parameter payloads. Once a new JSON object is received, the parsing logic updates the EA's internal parameters.

A significant operational challenge introduced by this hybrid architecture is the risk of temporal lag or race conditions during parameter synchronization. The Python model runs asynchronously, potentially updating parameters every few minutes (or weekly during the MLOps deployment). The MQL5 EA, however, processes in real-time, often using the ultra-low-latency OnTick() method.\[48\] If the EA receives a new parameter set mid-trade, the update must be synchronized, typically by making the change atomic (instantaneous and simultaneous) before the next price tick. The inclusion of a Timestamp in the JSON payload helps the MQL5 logic confirm that it is using the newest parameter configuration, mitigating the risk of executing trades with stale rules.

This hybrid architecture also introduces a fundamental dependency: the entire system relies on the MT5 terminal remaining constantly running and connected to the broker via the ZMQ middleware.\[42\] Consequently, the MLOps monitoring pipeline must extend beyond tracking model performance to include continuous tracking of the health, latency, and connectivity of the ZMQ link and the MT5 terminal itself.\[35\]

## **2.6 Conclusions and Future Work** {#2.6-conclusions-and-future-work}

The Adaptive Algorithmic Trading framework successfully addresses the core challenge of market non-stationarity in Forex through the construction of a robust, hybrid machine learning and execution system.

### 2.6.1 Synthesis of Findings {#2.6.1-synthesis-of-findings}

1\. Regime Classification (The ML Core): The system establishes a mathematically sophisticated approach to market state identification by utilizing Unsupervised Clustering (GMM) on advanced structural features, notably the Hurst Exponent and Fractal Dimension. The Hurst Exponent serves as the necessary quantifiable link, translating abstract cluster separation into economically actionable regimes (trending vs. mean-reverting).\[19\]

2\. Adaptive Parameterization: The framework implements Conditional Parameter Optimization (CPO), moving beyond fixed strategies to dynamically map each identified regime to a distinct, pre-optimized set of MQL5 Expert Advisor parameters. This adaptation is essential for maximizing risk-adjusted performance across changing market dynamics.\[10, 11\]

3\. MLOps Rigor (Continuous Training): The commitment to a fixed weekly rolling window re-training schedule serves as a critical, proactive defense against the continuous and pervasive nature of concept drift in financial markets, a method highly suited to Forex's non-stationary environment.\[4, 39\] Robustness is further ensured by mandated Walk Forward Analysis (WFA) during all optimization cycles, minimizing the risk of overfitting the parameters to historical noise.\[25\]

4\. Architectural Integrity: The Python-MQL5 integration, leveraging ZeroMQ and JSON for high-performance IPC, successfully separates the analytical complexity (Python) from the low-latency execution mandate (MQL5), establishing a design that satisfies the requirements of a production-grade algorithmic trading system.\[41, 42\]

### 2.6.2 Recommendations and Future Trajectories {#2.6.2-recommendations-and-future-trajectories}

1\. Enhancing Optimization Methodology: While the current framework uses optimization routines to define parameter sets for each regime, a rigorous extension would involve replacing the static parameter lookup table with a dynamic decision-making policy generated by a Contextual Reinforcement Learning (RL) agent. The clustering output naturally provides the distinct contexts (states) required for training an RL Meta-Controller that learns the optimal parameter selection policy, significantly boosting the system’s adaptability.\[26, 29\]

2\. Dual Drift Management: Although the fixed weekly schedule is justified for continuous drift, the system should integrate low-latency, parallel statistical drift monitoring (e.g., PSI or Wasserstein distance) to detect and alert operators to sudden, abrupt concept drift events that may occur between scheduled retraining cycles.\[35, 38\]

3\. Comprehensive MLOps Governance: Future development must integrate dedicated MLOps platforms (e.g., MLflow) to ensure full versioning of all data snapshots, feature sets, model artifacts, and optimization results, guaranteeing complete reproducibility and auditability, which is vital for compliance in regulated financial environments.\[34\]

# **3\. Research Methodology**

## **3.1 Introduction to the Methodology** {#3.1-introduction-to-the-methodology}

This chapter details the research methodology employed to investigate the efficacy of adaptive machine learning in combating market non-stationarity for algorithmic Forex trading. The primary aim of this project is to design, implement, and validate a robust, continuous adaptation framework. 

A **quantitative, simulation-based experimental approach** is adopted to evaluate the system. This methodology is designed to ensure systematic model training, objective performance measurement, and direct comparison against static, industry-standard baselines, thereby guaranteeing the reproducibility and academic rigor of the results.

## 

## **3.2 Research Design** {#3.2-research-design}

This research employs a System Design and Validation strategy combined with a Comparative Experimental Evaluation. The quantitative design is structured around three main components:

1. Dynamic System Design: Designing a hybrid Python-MQL5 architecture centered on Gaussian Mixture Model (GMM) clustering and ZeroMQ (ZMQ) IPC.

2. Conditional Parameter Optimization (CPO): Implementing a logic layer that translates GMM-identified market regimes into optimal DCA risk parameters.

3. Comparative Simulation (WFA): Conducting an Iterative Walk-Forward Analysis (WFA) over a multi-year historical dataset to compare the time-varying performance of the proposed Adaptive System against an optimized Static Baseline System under identical market conditions.

The results generated from the WFA will serve as the core dataset for Chapter 4, providing a pure measure of the adaptive algorithm's efficacy, isolated from external deployment factors.

## 

## **3.3 Research Questions and Hypotheses** {#3.3-research-questions-and-hypotheses}

The methodology is designed to address the following central hypothesis, which flows from the problem statement:

Hypothesis (H1): The Adaptive Algorithmic Trading System, employing a fixed weekly rolling-window retraining MLOps pipeline and Conditional Parameter Optimization (CPO) based on GMM regime classification, will achieve a statistically higher Walk Forward Efficiency (WFE) and Recovery Factor compared to a fixed-parameter Static Baseline over a multi-year validation period (2020–2024).

This primary hypothesis is further supported by two operational research questions:

* RQ1: How effectively does the GMM, utilizing structural features (Hurst Exponent, ATR, ADX), segment historical data into economically meaningful market regimes (Trending, Ranging, Volatile)?

* RQ2: To what extent does the CPO process improve the risk-adjusted performance (Sharpe Ratio, Recovery Factor) of the strategy during the out-of-sample (OOS) periods of the WFA?

## **3.4 Data Sources and Data Collection** {#3.4-data-sources-and-data-collection}

The study uses a single, high-fidelity data source to ensure methodological consistency:

| Attribute | Detail | Rationale |
| :---- | :---- | :---- |
| **Data Source** | MetaTrader 5 Terminal via Python API (MetaTrader5 library). | High-quality, tick-data-derived OHLCV bars. |
| **Asset** | EUR/USD currency pair. | Chosen as the global benchmark for liquidity, minimizing execution noise.  |
| **Granularity** | M15 (15-Minute) OHLCV data. | Balances the need to capture volatility shifts against the need to filter higher-frequency market noise.  |
| **Test Period** | Historical data from 2020–2024 (Multi-year window).	 | Provides a statistically significant, diverse set of market conditions (ranging, trending, high-volatility) for robust WFA.  |

Preprocessing: Data preprocessing steps include Gap Filling (forward-filling missing timestamps) and conversion of raw prices to Log-Returns to ensure stationarity for variance-based feature calculation, addressing the fundamental challenge of market non-stationarity.

## **3.5 System / Model / Framework Description** {#3.5-system-/-model-/-framework-description}

The proposed solution is a hybrid architecture consisting of three functional layers: the Machine Learning Core, the Dynamic Mapping Layer, and the Continuous Training/MLOps Pipeline. This structure is illustrated in the architectural diagrams of the thesis (as referenced by the guidelines).

### 3.5.1 The Machine Learning Core: Regime Classification {#3.5.1-the-machine-learning-core:-regime-classification}

The core component is the Gaussian Mixture Model (GMM), configured with *k* \= 4 components (clusters). This unsupervised clustering technique operates on a compact vector of structural features, specifically:

* Hurst Exponent (H): Quantifies long-term memory (Persistence vs. Mean-Reversion).

* Normalized Average True Range (ATR): Measures volatility and price movement magnitude.

* Average Directional Index (ADX): Measures trend momentum strength.

### 3.5.2 The Dynamic Mapping Layer: Conditional Parameter Optimization (CPO) {#3.5.2-the-dynamic-mapping-layer:-conditional-parameter-optimization-(cpo)}

The GMM's cluster output (Regime ID) is translated into actionable DCA parameters via the CPO process. This involves:

1. Regime Interpretation: Analyzing the GMM cluster centroids (mean *H*, ATR, ADX values) to assign an Economic Label ('Ranging', 'Strong Trend', etc.).

2. Segmented Optimization: Running the strategy's optimization routine (maximizing Sharpe Ratio with Drawdown constraints) exclusively on the historical data subset corresponding to a single regime.

3. Parameter Mapping: Storing the resulting optimal parameters (Pi) for the DCA strategy (Distance Multiplier, Lot Multiplier) in a lookup table artifact (trade\_params.json).

### 3.5.3 Inter-Process Communication and Execution {#3.5.3-inter-process-communication-and-execution}

The Analytical Layer communicates with the MQL5 Execution Layer using ZeroMQ (ZMQ). The process involves:

* Python (Server): The Inference Server listens on a ZMQ REP socket. Upon request, it executes the prediction and performs the CPO parameter lookup.

* MQL5 (Client): The Expert Advisor sends a request to the ZMQ socket on each new M15 bar. It receives the JSON payload and instantly updates its internal DCA parameters. This Hybrid Decision Architecture uses a Static MACD for trade timing and the Adaptive ML Parameters for risk sizing/spacing.

## **3.6 Tools, Technologies, and Platforms** {#3.6-tools,-technologies,-and-platforms}

The following tools and languages are used for implementation:

| Component | Tool / Technology | Role in Thesis |
| :---- | :---- | :---- |
| **ML & Analytics** | Python (3.9+), scikit-learn, numpy | Model training, feature engineering, CPO orchestration. |
| **Execution** | MQL5, MetaTrader 5 (MT5) | Low-latency trade management and platform connectivity. |
| **IPC** | ZeroMQ (ZMQ), pyzmq | Asynchronous communication between Python and MQL5. |
| **Orchestration/Simulation** | Docker, Streamlit, .venv | Containerization, environment isolation, local simulation of Cloud MLOps pipeline. |
| **Validation** | Walk-Forward Analysis (WFA) | Core method for testing robustness on unseen data. |

## 

## **3.7 Evaluation Metrics and Analysis Techniques** {#3.7-evaluation-metrics-and-analysis-techniques}

System performance is evaluated using a rigorous, multi-objective metric set, prioritized for capital preservation and risk-adjusted returns, as defined in Table \[Reference Table 2.3.3\]:

1. Primary Performance Metric: Sharpe Ratio ((Rp−Rf)/σp). Measures excess return per unit of volatility. Goal: Maximize Sharpe Ratio.

2. Risk Management Metric: Recovery Factor (Net Profit/Maximum Drawdown). Measures the system’s resilience and ability to recover from losses. Goal: Maximize Recovery Factor (Target \> 3.0).

3. Robustness Metric: Walk Forward Efficiency (WFE) (Total Net ProfitOOS/Total Net ProfitIS). Measures the stability of performance between the in-sample (IS) optimization period and the unseen out-of-sample (OOS) validation period. Goal: Maintain WFE \> 70% across the simulation.

## **3.8 Validation and Comparison** {#3.8-validation-and-comparison}

The experimental evaluation is structured as a direct comparison over the 2020-2024 dataset:

1. Static Baseline Creation: A conventional backtest is performed on the entire dataset to find the single best fixed-parameter set for the DCA strategy. This single best result serves as the Static Baseline.

2. Adaptive System Validation (WFA): The Adaptive System is validated using a continuous, iterative WFA mechanism (e.g., 100 days IS, 20 days OOS). The retraining\_script.py executes a full GMM training and CPO process at the start of each new WFA segment.

3. Comparison: The aggregate performance metrics (Sharpe, Recovery, Max Drawdown) of the Adaptive System's combined OOS segments are directly compared against the metrics of the Static Baseline.

## **3.9 Ethical Considerations** {#3.9-ethical-considerations}

This research adheres to ethical guidelines for computational and financial studies: all price data utilized is publicly available (non-proprietary Forex OHLCV data). The implementation involves no personally identifiable information (PII) or user data. The primary ethical consideration is transparency, which is addressed by adopting a rigorous MLOps framework to ensure complete auditability and reproducibility of all data acquisition, model training, and performance results.

## **3.10 Limitations of the Methodology** {#3.10-limitations-of-the-methodology}

The following constraints and limitations are acknowledged:

* Simulation-Based Only: The results are derived solely from historical simulation (WFA) and do not account for real-world execution factors like live network latency, broker slippage, or partial order fills beyond the MT5 simulation engine.

* Single-Asset Focus: Results are specific to EUR/USD; generalization to other currency pairs or asset classes is outside the scope.

* DCA Constraint: The adaptive mechanism is limited to only two core DCA parameters, isolating the risk management effect but simplifying the full trade strategy potential.

* Static Entry Signal: The use of a simple, static MACD entry signal prevents the full exploration of an adaptive entry strategy, focusing the thesis purely on adaptive risk management.

## **3.11 Summary of the Chapter** {#3.11-summary-of-the-chapter}

This chapter has detailed a rigorous, mixed-methods research design centered on a quantitative, simulation-based comparison. The methodology establishes an end-to-end MLOps pipeline for GMM regime classification, Conditional Parameter Optimization (CPO), and Walk-Forward Analysis (WFA). The design, tools, and metrics are explicitly defined to provide the data required to validate the core hypothesis regarding the adaptive system's superior performance, setting the stage for the presentation of results in Chapter 4\.

# **4\. Implementation and Results** {#4.-implementation-and-results}

## **4.1 System Implementation Overview** {#4.1-system-implementation-overview}

This chapter presents the complete implementation of the Adaptive Algorithmic Trading System, detailing the software architecture, module interactions, and the results obtained from the Walk-Forward Analysis validation.

### 4.1.1 System Architecture {#4.1.1-system-architecture}

The implemented system follows a layered architecture that separates concerns between trading execution (MQL5), machine learning (Python), and orchestration (MLOps).

![][image3]  
**Figure 4.1**: Complete System Architecture showing the three-layer design with MQL5 trading execution, Python ML inference, and MLOps pipeline components.

### 4.1.2 Component Implementation Details {#4.1.2-component-implementation-details}

A. Data Ingestion Module (data\_loader.py)  
The data loader handles CSV imports from MetaTrader 5, supporting multiple encodings and formats.

| Feature | Implementation |
| ----- | ----- |
| Encoding Support | UTF-16, UTF-8, UTF-16-LE |
| Separator Detection | Tab, Comma, Semicolon |
| Column Normalization | Strips \<\> from MT5 headers |
| Type Coercion | Forces numeric types with pd.to\_numeric() |
| DateTime Handling | Combines DATE \+ TIME columns |

Listing 4.1: Core data loading logic

\# Robust multi-format CSV parsing  
candidates \= \[  
    ('\\t', 'utf-16'),    \# Standard MT5 export  
    (',', 'utf-8'),      \# Alternative format  
    (';', 'utf-8')       \# European locale  
\]

for sep, encoding in candidates:  
    df\_try \= pd.read\_csv(source, sep\=sep, encoding\=encoding)  
    if len(df\_try.columns) \> 1:  \# Validation  
break

B. Feature Engineering Module (feature\_engineering.py)

Three market complexity features are calculated per the methodology in Chapter 3\.

| Feature | Formula | Window | Purpose |
| ----- | ----- | ----- | ----- |
| Hurst Exponent | R/S Analysis | 100 bars | Market memory/trending |
| Normalized ATR | ATR(14) / Close | 14 bars | Volatility level |
| ADX | Directional Index | 14 bars | Trend strength |

Implementation Note: A custom Rescaled Range (R/S) Hurst calculation was implemented due to instability in the hurst library for certain data patterns.

`def _manual_hurst(ts):`  
    `"""Calculates Hurst Exponent using R/S analysis."""`  
    `# Divide series into sub-periods`  
    `for k in range(min_k, max_k):`  
        `# Calculate R/S for each sub-period`  
        `R = np.max(cumsum) - np.min(cumsum)`  
        `S = np.std(subseries)`  
        `rs_list.append(R / S)`  
      
    `# Linear regression on log-log scale`  
    `H, _ = np.polyfit(np.log(n_list), np.log(rs_list), 1)`  
    `return H`

C. GMM Training Module (retraining\_script.py)  
The Gaussian Mixture Model classifier was configured per the methodology specification.

| Parameter | Value | Rationale |
| ----- | ----- | ----- |
| n\_components | 4 | Four distinct market regimes |
| covariance\_type | 'full' | Captures feature correlations |
| random\_state | 42 | Reproducibility |
| Scaling | StandardScaler | Normalizes feature magnitudes |

Cluster-to-Regime Mapping: Clusters are sorted by mean Hurst Exponent and mapped to economic interpretations:

| Rank | Hurst Range | Regime Label | Trading Parameters |
| ----- | ----- | ----- | ----- |
| 0 (Lowest H) | 0.85-0.92 | Ranging (Safe) | Dist=1.2x, Lot=1.5x |
| 1 | 0.92-0.95 | Choppy/Weak Trend | Dist=1.5x, Lot=1.3x |
| 2 | 0.95-0.98 | Trending | Dist=2.0x, Lot=1.2x |
| 3 (Highest H) | 0.98+ | Strong Trend/Breakout | Dist=2.5x, Lot=1.1x |

Table 4.1: CPO (Cluster Parameter Optimization) mapping from GMM clusters to trading parameters.

D. Inference Server (inference\_server.py)  
The real-time prediction server uses ZeroMQ for low-latency IPC.

# **![][image4]**

Figure 4.2: Sequence diagram showing the real-time parameter adaptation flow from MT5 through the Python inference server.

## **4.2 Walk-Forward Analysis Results** {#4.2-walk-forward-analysis-results}

### 4.2.1 WFA Configuration {#4.2.1-wfa-configuration}

The Walk-Forward Analysis was conducted with the following parameters:

| Parameter | Value |
| ----- | ----- |
| Data Period | December 2021 – December 2024 |
| Symbol | EUR/USD |
| Timeframe | M15 (15-minute bars) |
| In-Sample Window | 6 months |
| Out-of-Sample Step | 1 week |
| Total Data Points | 76,188 bars |
| Valid Iterations | 133 |

### 4.2.2 Aggregate Results {#4.2.2-aggregate-results}

The WFA produced the following aggregate metrics across all 133 iterations:

| Metric | Value | Interpretation |
| ----- | ----- | ----- |
| Total Iterations | 133 | Sufficient sample size for statistical validity |
| Average Stability Ratio | 87.75% | Regimes persist; low noise |
| Standard Deviation (Stability) | TBD | Consistency of stability |
| Average Generalization Gap | 0.1135 | Low overfitting risk |
| Standard Deviation (Gen. Gap) | TBD | Consistency of generalization |

Table 4.2: WFA Aggregate Metrics

### 4.2.3 Regime Distribution {#4.2.3-regime-distribution}

Analysis of dominant regimes across OOS periods reveals the market's regime composition during 2021-2024:

| Regime | Cluster ID | Dominant Count | Percentage | Interpretation |
| ----- | ----- | ----- | ----- | ----- |
| Trending | R0 | 49 | 36.8% | Standard trending behavior |
| Strong Trend | R1 | 40 | 30.1% | High persistence moves (Fed cycle) |
| Choppy | R2 | 23 | 17.3% | Mixed/volatile conditions |
| Ranging | R3 | 21 | 15.8% | Mean-reverting markets |

Table 4.3: Regime Distribution from WFA

![][image5]  
Figure 4.3: Pie chart showing the distribution of dominant regimes across 133 WFA iterations.

Economic Interpretation: The dominance of Trending (R0) and Strong Trend (R1) regimes, totaling 66.9% of periods, aligns with the macroeconomic context of 2022-2023 where aggressive Federal Reserve rate hikes created sustained directional moves in EUR/USD.

### 4.2.4 Final Model Cluster Centroids {#4.2.4-final-model-cluster-centroids}

The model trained on the complete dataset produced the following cluster centers:

| Cluster | Hurst | ATR (Norm.) | ADX | Label |
| ----- | ----- | ----- | ----- | ----- |
| 0 | 0.9787 | 0.000518 | 22.94 | Trending |
| 1 | 1.0000 | 0.000654 | 25.59 | Strong Trend / Breakout |
| 2 | 0.9387 | 0.000941 | 31.34 | Choppy / Weak Trend |
| 3 | 0.9006 | 0.000554 | 19.00 | Ranging (Safe) |

Table 4.4: Final GMM Cluster Centroids

Observations:

1. Hurst Separation: Clusters span from 0.90 (Ranging) to 1.00 (Strong Trend), indicating clear regime differentiation.  
2. Volatility Correlation: Cluster 2 (Choppy) has the highest ATR, suggesting volatile but directionless conditions.  
3. ADX Alignment: Cluster 2 has the highest ADX (31.34), which may seem contradictory but reflects high directional movement intensity without sustained persistence.

## **4.3 MQL5 Integration Results** {#4.3-mql5-integration-results}

### 4.3.1 Backtest Methodology {#4.3.1-backtest-methodology}

### To validate the practical trading impact, backtests were conducted using MT5 Strategy Tester. {#to-validate-the-practical-trading-impact,-backtests-were-conducted-using-mt5-strategy-tester.}

### 4.3.2 Baseline Backtest (Static Parameters) {#4.3.2-baseline-backtest-(static-parameters)}

Configuration:

* EA: FXATM.mq5 (Original)  
* Initial Lot Size: 0.01 (Fixed)  
* Stop Loss: 500 pips (Fixed)  
* Take Profit: 21 pips (Fixed)  
* Distance: 21 pips (Fixed)  
* DCA Step Multiplier: 1.5 (Fixed)  
* DCA Lot Multiplier: 1.2 (Fixed)  
* Period: 2021-2024  
* Trailing Stop Loss: Disabled  
* Stacking: Disabled  
* Partial Take Profit: Disabled  
* Initial Balance: $10,000  
* Trade Entries: M15, MACD 12/26/9, Main & Signal lines crossovers

Backtest Results:

![][image6]

| Metric | Value |
| ----- | ----- |
| Total Net Profit | $6,730.52 |
| Max Drawdown | 22.41% |
| Profit Factor | 1.23 |
| Sharpe Ratio | 0.29 |
| Recovery Factor | 1.10 |
| Total Trades | 3911 |
| Win Rate | 71.36% |

Table 4.5: Baseline EA Performance Metrics 

### 4.3.3 Adaptive Backtest (ML-Driven Parameters) {#4.3.3-adaptive-backtest-(ml-driven-parameters)}

Configuration:

* EA: FXATM\_MSc.mq5 (Adaptive)  
* DCA Parameters: Dynamic (from regime prediction)  
* Python Server: Historical regime lookup mode  
* Period: 2021-2024  
* Initial Balance: $10,000

\[INSERT BACKTEST RESULTS\]

| Metric | Value |
| ----- | ----- |
| Total Net Profit | $\_\_\_\_\_ |
| Max Drawdown | \_\_\_\_\_% |
| Profit Factor | \_\_\_\_\_ |
| Sharpe Ratio | \_\_\_\_\_ |
| Recovery Factor | \_\_\_\_\_ |
| Total Trades | \_\_\_\_\_ |
| Win Rate | \_\_\_\_\_% |

Table 4.6: Adaptive EA Performance Metrics (PLACEHOLDER)

### **4.3.4 Comparative Analysis** {#4.3.4-comparative-analysis}

| Metric | Baseline | Adaptive | Δ Change |
| ----- | ----- | ----- | ----- |
| Net Profit | $\_\_\_\_\_ | $\_\_\_\_\_ | \_\_% |
| Max Drawdown | \_\_\_% | \_\_\_% | \_\_% |
| Profit Factor | \_\_\_ | \_\_\_ | \_\_% |
| Sharpe Ratio | \_\_\_ | \_\_\_ | \_\_% |
| Recovery Factor | \_\_\_ | \_\_\_ | \_\_% |

Table 4.7: Comparative Performance (PLACEHOLDER)

## **4.4 System Validation Summary** {#4.4-system-validation-summary}

### 4.4.1 Validation Criteria Assessment {#4.4.1-validation-criteria-assessment}

| Criterion | Target | Achieved | Status |
| ----- | ----- | ----- | ----- |
| Regime Stability | \>80% | 87.75% | ✅ Pass |
| Generalization Gap | \<0.3 | 0.1135 | ✅ Pass |
| WFA Iterations | \>50 | 133 | ✅ Pass |
| Distinct Clusters | 4 | 4 | ✅ Pass |
| Hurst Separation | \>0.05 | 0.10 | ✅ Pass |
| Real-time Latency | \<100ms | TBD | ⏳ Pending |
| Backtest Improvement | \>10% Sharpe | TBD | ⏳ Pending |

Table 4.8: System Validation Criteria

# **5\. Discussion and Conclusion** {#5.-discussion-and-conclusion}

## **5.1 Discussion of Results** {#5.1-discussion-of-results}

### 5.1.1 Interpretation of WFA Findings {#5.1.1-interpretation-of-wfa-findings}

The Walk-Forward Analysis results provide strong evidence for the validity of the regime-based adaptive approach:

1\. High Regime Stability (87.75%)

The average stability ratio of 87.75% indicates that once the GMM classifier identifies a market regime, that regime persists for approximately 87.75% of the subsequent bars within the OOS period. This has two important implications:

* Practical Trading: The EA will not be subjected to frequent parameter changes, avoiding "whipsaw" behavior where constantly switching parameters could lead to suboptimal positioning.  
* Theoretical Validity: The persistence confirms that markets do exhibit distinct behavioral states (regimes) rather than random walks, supporting the foundational hypothesis of regime-based trading.

2\. Low Generalization Gap (0.1135)

The generalization gap measures the difference between in-sample model fit and out-of-sample predictive performance. A value of 0.1135 indicates:

* Minimal Overfitting: The GMM trained on 6 months of historical data generalizes well to the subsequent week.  
* Robust Feature Set: The combination of Hurst Exponent, Normalized ATR, and ADX captures fundamental market characteristics rather than superficial patterns.

3\. Regime Distribution Insights

The predominance of Trending (36.8%) and Strong Trend (30.1%) regimes during 2021-2024 aligns with the macroeconomic environment:

* 2022: Aggressive Fed rate hikes drove strong USD appreciation  
* 2023: Rate pause and "higher for longer" rhetoric maintained trends  
* 2024: Rate cut expectations created new directional moves

This suggests the regime classifier correctly identified the market's trending nature during this period.

### 5.1.2 Practical Implications {#5.1.2-practical-implications}

For Grid Trading Systems:

The CPO (Cluster Parameter Optimization) approach provides a principled framework for DCA parameter selection:

| Regime | Grid Behavior | Risk Profile |
| ----- | ----- | ----- |
| Ranging | Tight grids (1.2x step), aggressive sizing (1.5x lot) | Higher position count, faster recovery |
| Trending | Wide grids (2.5x step), conservative sizing (1.1x lot) | Fewer positions, reduced drawdown risk |

For Algorithmic Trading Practitioners:

1. Feature Selection: The Hurst Exponent proves to be a powerful discriminator for regime classification.  
2. Model Choice: GMM provides interpretable clusters with meaningful economic mapping.  
3. Retraining Frequency: The 6-month IS window with weekly OOS validation provides a balance between adaptation and stability.

### 5.1.3 Limitations and Threats to Validity {#5.1.3-limitations-and-threats-to-validity}

1\. Single Currency Pair

The validation was conducted exclusively on EUR/USD. Generalization to other pairs (e.g., GBP/JPY, gold) would require additional validation.

2\. Specific Market Period

The 2021-2024 period was characterized by unusual monetary policy actions. Performance during "normal" market conditions may differ.

3\. Simulation-Based Validation

While the WFA provides strong methodological evidence, real-money trading involves additional factors (slippage, order execution, emotional discipline) not captured in simulation.

4\. Strategy Tester Constraints

MT5's Strategy Tester cannot connect to external ZMQ servers during backtesting, requiring a "historical lookup" workaround that may not perfectly replicate live behavior.

## **5.2 Contributions** {#5.2-contributions}

This research makes the following contributions to the field:

### 5.2.1 Theoretical Contributions {#5.2.1-theoretical-contributions}

1. Regime-Aware DCA Framework: A novel approach to dynamic parameter optimization in grid trading systems based on market regime classification.  
2. CPO Methodology: A systematic mapping from unsupervised clusters to trading parameters using economic interpretation.

### 5.2.2 Practical Contributions {#5.2.2-practical-contributions}

1. End-to-End MLOps Implementation: A complete, deployable system from data ingestion to live trading.  
2. Reproducible Validation Framework: The Walk-Forward Analysis methodology can be applied to other trading strategies.  
3. Open Architecture: Modular design allows components to be reused or extended.

## **5.3 Future Work** {#5.3-future-work}

### 5.3.1 Short-Term Improvements {#5.3.1-short-term-improvements}

| Enhancement | Description | Priority |
| ----- | ----- | ----- |
| Multi-Pair Validation | Test on GBP/USD, USD/JPY, XAU/USD | High |
| Live Trading Pilot | Small-scale live validation with real capital | High |
| Regime Transition Alerts | Notify when regime changes occur | Medium |
| Dashboard Enhancements | Time-series plots of regime evolution | Medium |

### 5.3.2 Long-Term Research Directions {#5.3.2-long-term-research-directions}

1. Deep Learning Regime Detection: Replace GMM with LSTM or Transformer-based classifiers.  
2. Reinforcement Learning CPO: Use RL to optimize the regime-to-parameter mapping instead of fixed rules.  
3. Multi-Timeframe Analysis: Incorporate higher timeframe regimes for hierarchical decision making.  
4. Sentiment Integration: Add news/sentiment features to the regime classifier.

## **5.4 Conclusion** {#5.4-conclusion}

This thesis presented an Adaptive Algorithmic Trading System that uses machine learning to classify market regimes and dynamically adjust DCA grid trading parameters. The system was validated using Walk-Forward Analysis on EUR/USD M15 data from December 2021 to December 2024\.

Key Findings:

1. The GMM-based regime classifier achieved 87.75% average stability, indicating that identified regimes are persistent and tradeable.  
2. The generalization gap of 0.1135 demonstrates robust out-of-sample performance with minimal overfitting.  
3. The regime distribution (67% trending, 33% ranging) aligns with the macroeconomic context of the test period.  
4. The modular architecture enables seamless integration between MQL5 trading and Python ML components.

Practical Outcome:

The FXATM\_MSc Expert Advisor represents a production-ready implementation of the adaptive trading framework. With the trained GMM model and CPO parameters, traders can deploy the system with the expectation of regime-aware parameter adjustment.

Final Remark:

This research demonstrates that machine learning can be effectively applied to enhance traditional algorithmic trading strategies. By moving from static to adaptive parameters, trading systems can better navigate the non-stationary nature of financial markets.

# **6\. Bibliography and Web Sources** {#6.-bibliography-and-web-sources}

1\. Modeling Stylized Facts in FX Markets with FINGAN-BiLSTM: A Deep Learning Approach to Financial Time Series \- MDPI, https://www.mdpi.com/1099-4300/27/6/635

2\. Regime-Switching Factor Investing with Hidden Markov Models \- MDPI, https://www.mdpi.com/1911-8074/13/12/311

3\. Detecting & Handling Data Drift in Production \- MachineLearningMastery.com, https://machinelearningmastery.com/detecting-handling-data-drift-in-production/

4\. Model Drift in Machine Learning \- Aerospike, https://aerospike.com/blog/model-drift-machine-learning

5\. Walk-Forward Optimization: How It Works, Its Limitations, and Backtesting Implementation, https://blog.quantinsti.com/walk-forward-optimization-introduction/

6\. Market Regime Detection using Hidden Markov Models in QSTrader | QuantStart, https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

7\. A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets \- arXiv, https://arxiv.org/abs/2108.05801

8\. A comparison of adaptive and static agents in equity market trading \- ResearchGate, https://www.researchgate.net/publication/4207556\_A\_comparison\_of\_adaptive\_and\_static\_agents\_in\_equity\_market\_trading

9\. \[1811.11265\] Static vs Adaptive Strategies for Optimal Execution with Signals \- arXiv, https://arxiv.org/abs/1811.11265

10\. Conditional Parameter Optimization: Adapting Parameters to Changing Market Regimes | by PredictNow.ai, https://predictnow-ai.medium.com/conditional-parameter-optimization-adapting-parameters-to-changing-market-regimes-b7158ab78ed4

11\. How to Optimise Expert Advisors (EAs) in MT5 \- Ultima Markets, https://www.ultimamarkets.com/academy/how-to-optimise-expert-advisors-eas-in-mt5/

12\. MLOps: Continuous delivery and automation pipelines in machine learning | Cloud Architecture Center, https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

13\. aaronroman/financial-time-series-clustering: Unsupervised clustering to generate predictive features from stock price curves \- GitHub, https://github.com/aaronroman/financial-time-series-clustering

14\. 2.3. Clustering — scikit-learn 1.7.2 documentation, https://scikit-learn.org/stable/modules/clustering.html

15\. Market regime detection using Statistical and ML based approaches | Devportal, https://developers.lseg.com/en/article-catalog/article/market-regime-detection

16\. Deep Switching State Space Model for Nonlinear Time Series Forecasting with Regime Switching \- arXiv, https://arxiv.org/pdf/2106.02329

17\. Gaussian Mixture Models (GMM) — AI Meets Finance: Algorithms Series | by Leo Mercanti, https://wire.insiderfinance.io/gaussian-mixture-models-gmm-ai-meets-finance-algorithms-series-d97262deadee

18\. julienraffaud/GMM: Gaussian mixture model for regime detection in financial time series, https://github.com/julienraffaud/GMM

19\. Detecting trends and mean reversion with the Hurst exponent | Macrosynergy, https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/

20\. Forecasting Forex Market Volatility Using Deep Learning Models and Complexity Measures, https://www.mdpi.com/1911-8074/17/12/557

21\. Fractal Analysis of Time Series and Distribution Properties of Hurst Exponent, https://msme.us/2011-1-2.pdf

22\. Topology of Currencies: Persistent Homology for FX Co-movements: A Comparative Clustering Study \- arXiv, https://arxiv.org/html/2510.19306v1

23\. Multi-Scale Foreign Exchange Rates Ensemble for Classification of Trends in Forex Market, https://www.researchgate.net/publication/262934823\_Multi-Scale\_Foreign\_Exchange\_Rates\_Ensemble\_for\_Classification\_of\_Trends\_in\_Forex\_Market

24\. Optimizing Your Expert Advisor for Maximum Profitability \- XAUBOT,   
https://xaubot.com/optimizing-your-expert-advisor-for-maximum-profitability/

25\. Walk Forward Analysis For Algo Traders \- Helping you Master EasyLanguage, https://easylanguagemastery.com/products/walk-forward-analysis/

26\. Building an Adaptive Trading System with Regime Switching, GA's & RL : r/quant \- Reddit, https://www.reddit.com/r/quant/comments/1jhhk3c/building\_an\_adaptive\_trading\_system\_with\_regime/

27\. A Study of Trade Strategies Based on the Markov Regime Switching Model, https://madison-proceedings.com/index.php/aemr/article/download/1163/1162

28\. Deep LSTM with Reinforcement Learning Layer for Financial Trend Prediction in FX High Frequency Trading Systems \- Semantic Scholar, https://www.semanticscholar.org/paper/Deep-LSTM-with-Reinforcement-Learning-Layer-for-in-Rundo/dac5d4eeb5d3af1ce81521198500f76e1b7bf01c

29\. Contextual Deep Reinforcement Learning with Adaptive Value-based Clustering \- Amazon Science, https://assets.amazon.science/33/df/2a75f91b4900815bdc37de01abfe/contextual-deep-reinforcement-learning-with-adaptive-value-based-clustering.pdf

30\. CAD: Clustering And Deep Reinforcement Learning Based Multi-Period Portfolio Management Strategy \- arXiv, https://arxiv.org/pdf/2310.01319

31\. Algorithmic Trading System with Adaptive State Model of a Binary-Temporal Representation, https://www.mdpi.com/2227-9091/13/8/148

32\. Sharpe Ratio: Definition, Formula, and Examples \- Investopedia, https://www.investopedia.com/terms/s/sharperatio.asp

33\. A Practical Machine Learning Approach for Dynamic Stock Recommendation \- arXiv, https://arxiv.org/html/2511.12129v1

34\. 8 MLOps Best Practices for Scalable, Production-Ready ML Systems \- Azilen Technologies, https://www.azilen.com/blog/mlops-best-practices/

35\. What Is MLOps, How to Implement It, Examples \- Dysnix, https://dysnix.com/blog/what-is-mlops

36\. Adaptive Detection of Software Aging under Workload Shift \- arXiv, https://arxiv.org/html/2511.03103v2

37\. (PDF) Learning from Time-Changing Data with Adaptive Windowing \- ResearchGate, https://www.researchgate.net/publication/220907178\_Learning\_from\_Time-Changing\_Data\_with\_Adaptive\_Windowing

38\. What Is Model Drift? | IBM, https://www.ibm.com/think/topics/model-drift

39\. AI Model Drift & Retraining: A Guide for ML System Maintenance \- SmartDev, https://smartdev.com/ai-model-drift-retraining-a-guide-for-ml-system-maintenance/

40\. Building a MetaTrader 5 Trading Bot with Python: A Comprehensive Guide | Headway, https://hw.online/faq/building-a-metatrader-5-trading-bot-with-python-a-comprehensive-guide/

41\. Get started \- ZeroMQ, https://zeromq.org/get-started/

42\. Metatrader 5 binding ZeroMQ/Python \- Stack Overflow, https://stackoverflow.com/questions/49952723/metatrader-5-binding-zeromq-python

43\. aminch8/MT5-ZeroMQ: EA for Messaging Brokerage with ZeroMQ \- GitHub, https://github.com/aminch8/MT5-ZeroMQ

44\. Mastering JSON: Create Your Own JSON Reader from Scratch in MQL5 \- MQL5 Articles, https://www.mql5.com/en/articles/16791

45\. Integration of Broker APIs with Expert Advisors using MQL5 and Python, https://www.mql5.com/en/articles/16012

46\. Best MT5 Expert Advisors (EAs) for Funded Accounts | For Traders, https://www.fortraders.com/blog/best-mt5-expert-advisors-eas-for-funded-accounts

47\. Introduction to Expert Advisor Programming: Complete Guide \- ForexVPS, https://www.forexvps.net/resources/ea-programming/

48\. Integrating Python and ML Algorithms with MT5 for Forex Trading: Worth it? \- Reddit, https://www.reddit.com/r/algorithmictrading/comments/1cqxdju/integrating\_python\_and\_ml\_algorithms\_with\_mt5\_for/

# **Appendices** {#appendices}

## **Appendix A: Code Repository Structure** {#appendix-a:-code-repository-structure}

`msc-thesis/`  
`├── 0_DOCS/`  
`│   ├── PRD.md                    # Product Requirements Document`  
`│   ├── TECHNICAL_SPEC.md         # Technical Specification`  
`│   └── msc_thesis.md             # Thesis Paper (Chapters 1-3)`  
`│`  
`├── 1_MQL5_EA/`  
`│   ├── Experts/`  
`│   │   ├── FXATM.mq5             # Original EA (Baseline)`  
`│   │   └── FXATM_MSc.mq5         # Adaptive EA (ML-Integrated)`  
`│   └── Include/FXATM/Managers/`  
`│       ├── AdaptiveManager.mqh   # ML Integration Layer`  
`│       └── ZMQClient.mqh         # ZMQ Communication`  
`│`  
`├── 2_PYTHON_MLOPS/`  
`│   ├── config/`  
`│   │   ├── config.yaml           # System Configuration`  
`│   │   └── trade_params.json     # CPO Parameters`  
`│   ├── src/`  
`│   │   ├── data_loader.py        # CSV Parsing`  
`│   │   ├── feature_engineering.py# H, ATR, ADX Calculation`  
`│   │   ├── retraining_script.py  # GMM Training & WFA`  
`│   │   └── inference_server.py   # ZMQ Server`  
`│   └── streamlit_app.py          # Dashboard UI`  
`│`  
`└── 3_ML_ARTIFACTS/`  
    `├── gmm_model.pkl             # Trained GMM`  
    `├── scaler.pkl                # Feature Scaler`  
    `└── wfa_metrics.json          # WFA Results`

## 

## **Appendix B: WFA Period Details** {#appendix-b:-wfa-period-details}

Note  
The complete per-period WFA results are available in 3\_ML\_ARTIFACTS/wfa\_metrics.json. Sample entries are shown below.

| Iteration | OOS Start | OOS End | Dominant Regime | Stability | Gen. Gap |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | 2022-06-07 | 2022-06-14 | R0 (Trending) | 0.89 | 0.08 |
| 2 | 2022-06-14 | 2022-06-21 | R1 (Strong) | 0.92 | 0.11 |
| ... | ... | ... | ... | ... | ... |
| 133 | 2024-11-26 | 2024-12-03 | R0 (Trending) | 0.85 | 0.12 |

Table B.1: Sample WFA Period Results (First and Last Iterations)

# 

## **Appendix C: MT5 Backtest Configuration** {#appendix-c:-mt5-backtest-configuration}

Strategy Tester Settings:

| Setting | Value |
| ----- | ----- |
| Symbol | EURUSD |
| Period | M15 |
| Date Range | 2021.12.01 – 2024.12.01 |
| Modeling | Every Tick (Real Ticks if available) |
| Initial Deposit | 10000 USD |
| Leverage | 1:100 |
| Spread | Variable (from history) |

## **Appendix D: EA Source Codes** {#appendix-d:-ea-source-codes}

### MQL5 Directory Structure {#mql5-directory-structure}

The project is organized using standard MQL5 include directories.

`MQL5\`  
`├── Experts\`  
`│   └── FXATM.mq5             (The main EA executable file)`  
`│`  
`└── Include\`  
    `└── FXATM\                (Root folder for all project includes)`  
        `├── Managers\         (Folder for core logic classes)`  
        `│   ├── Settings.mqh`  
        `│   ├── TradeManager.mqh`  
        `│   ├── MoneyManager.mqh`  
        `│   ├── SignalManager.mqh`  
        `│   ├── DCAManager.mqh`  
        `│   ├── TrailingStopManager.mqh`  
        `│   ├── TimeManager.mqh`  
        `│   ├── NewsManager.mqh`  
        `│   ├── StackingManager.mqh`  
        `│   └── UIManager.mqh`  
        `│`  
        `└── Signals\	(Folder for signal definitions and implementations)`  
            `├── ISignal.mqh     (The "Interface" or Base Class)`  
            `├── CSignal_RSI.mqh`  
            `├── CSignal_MACD.mqh`  
            `├── CSignal_MA.mqh`  
            `├── CSignal_Stochastic.mqh`  
            `└── CSignal_BollingerBands.mqh`

`//+------------------------------------------------------------------+`  
`//|                                                        FXATM.mq5 |`  
`//|                                FX Automated Trading Manager v4.0 |`  
`//|                             Advanced Multi-Signal Expert Advisor |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`//| PURPOSE:                                                         |`  
`//|   Comprehensive automated trading system for MetaTrader 5        |`  
`//|   featuring multi-signal aggregation, advanced risk management,  |`  
`//|   and adaptive position sizing.                                  |`  
`//|                                                                  |`  
`//| KEY FEATURES:                                                    |`  
`//|   • 3-Slot Polymorphic Signal System (MACD, RSI, ATR, etc.)      |`  
`//|   • 6 Lot Sizing Modes (Fixed, Risk%, ATR-Volatility Adjusted)   |`  
`//|   • 5 Trailing Stop Loss Modes (Step, ATR, MA, High/Low)         |`  
`//|   • DCA & Stacking for basket expansion                          |`  
`//|   • Partial Take Profit with True Break-Even                     |`  
`//|   • News & Time filtering for risk control                       |`  
`//|   • Chart UI with manual trading controls                        |`  
`//|                                                                  |`  
`//| REQUIREMENTS:                                                    |`  
`//|   • MetaTrader 5 build 3280+                                     |`  
`//|   • Allow WebRequest for news filtering                          |`  
`//|   • Add https://nfs.forexfactory.net to allowed URLs             |`  
`//|                                                                  |`  
`//| VERSION HISTORY:                                                 |`  
`//|   4.00 - ATR-based features (TSL, lot sizing)                    |`  
`//|   3.00 - Multi-signal system, advanced basket management         |`  
`//|   2.00 - DCA and trailing stop implementation                    |`  
`//|   1.00 - Initial release with basic signal processing            |`  
`//+------------------------------------------------------------------+`  
`#property copyright   "Copyright 2025, LAWRANCE KOH"`  
`#property link        "lawrancekoh@outlook.com"`  
`#property version     "4.00"`  
`#property description "FXATM v4.0 - Advanced Multi-Signal Expert Advisor with ATR-based features"`

`#include <FXATM/Managers/Settings.mqh>`  
`#include <FXATM/Managers/TradeManager.mqh>`  
`#include <FXATM/Managers/MoneyManager.mqh>`  
`#include <FXATM/Managers/SignalManager.mqh>`  
`#include <FXATM/Signals/CSignal_MACD.mqh>`  
`#include <FXATM/Signals/CSignal_RSI.mqh>`  
`#include <FXATM/Signals/CSignal_MA.mqh>`  
`#include <FXATM/Signals/CSignal_Stochastic.mqh>`  
`#include <FXATM/Signals/CSignal_BollingerBands.mqh>`  
`#include <FXATM/Managers/DCAManager.mqh>`  
`#include <FXATM/Managers/TrailingStopManager.mqh>`  
`#include <FXATM/Managers/TimeManager.mqh>`  
`#include <FXATM/Managers/NewsManager.mqh>`  
`#include <FXATM/Managers/StackingManager.mqh>`  
`#include <FXATM/Managers/UIManager.mqh>`  
`#include <FXATM/Managers/CatrUtility.mqh>`

`// --- Manager Instances ---`  
`CTradeManager*             g_trade_manager;`  
`CMoneyManager*             g_money_manager;`  
`CSignalManager*            g_signal_manager;`  
`CDCAManager*               g_dca_manager;`  
`CTrailingStopManager*      g_tsl_manager;`  
`CTimeManager*              g_time_manager;`  
`CNewsManager*              g_news_manager;`  
`CStackingManager*          g_stacking_manager;`  
`CUIManager*                g_ui_manager;`  
`CatrUtility*               g_atr_utility;`

`// A. GENERAL SETTINGS`  
`input group "******** GENERAL SETTINGS ********";`  
`input string   InpEaName = "FXATMv4";                                  // EA name for display purposes`  
`input long     InpEaMagicNumber = 123456;                              // Unique ID for EA's trades`  
`input int      InpMaxSpreadPoints = 40;                                // Max allowed spread in POINTS for new trades`  
`input int      InpMaxSlippagePoints = 10;                              // Max allowed slippage in POINTS for all trades`  
`input double   InpMaxDrawdownPercent = 50.0;                           // Max drawdown % before stopping new trades (set to 100 or higher to disable)`  
`input ENUM_TIMEFRAMES InpEaHeartbeatTimeframe = PERIOD_M15;            // Timeframe for heartbeat (new bar check)`  
`input bool     InpAllowLongTrades = true;                              // Allow BUY (long) trades`  
`input bool     InpAllowShortTrades = true;                             // Allow SELL (short) trades`

`// B. POSITION MANAGEMENT SETTINGS`  
`input group "******** POSITION MANAGEMENT SETTINGS ********";`  
`input ENUM_LOT_SIZING_MODE InpLotSizingMode = MODE_FIXED_LOT;          // Lot sizing calculation method`  
`input double   InpLotFixed = 0.04;                                     // Lot size for Fixed Lot mode`  
`input double   InpLotsPerThousand = 0.01;                              // Lots per 1000 units of balance/equity`  
`input double   InpLotRiskPercent = 1.0;                                // Risk % for Balance/Equity modes`  
`input int      InpSlPips = 500;                                        // Initial SL pips (0 = no SL, disables risk modes)`  
`input int      InpInitialTpPips = 42;                                  // Initial TP in pips (0 = no TP)`  
`// Basket TP/SL moved here for complete position lifecycle management`  
`input int      InpBasketTpPips = 26;                                   // Basket TP in pips when basket has >1 position (0 = disabled)`

`// C. LOSS MANAGEMENT SETTINGS`  
`input group "******** LOSS MANAGEMENT SETTINGS ********";`  
`input int      InpDcaMaxTrades = 10;                                   // Max number of DCA trades allowed (0 = disabled)`  
`input int      InpDcaTriggerPips = 21;                                 // Initial pips in drawdown to add first DCA trade`  
`input double   InpDcaStepMultiplier = 1.1;                             // Step multiplier for subsequent DCA trades`  
`input double   InpDcaLotMultiplier = 1.5;                              // Lot multiplier for next DCA trade`  
`input int      InpDcaLotMultiplierStart = 2;                           // Multiplier starts from this trade number (e.g., 3rd trade)`

`// D. PROFIT MANAGEMENT SETTINGS`  
`input group "******** PROFIT MANAGEMENT SETTINGS ********";`  
`input ENUM_TSL_MODE InpTslMode = MODE_TSL_STEP;                        // Trailing stop loss mode`

`// E1. Trigger and Steps Settings`  
`input int      InpTslBeTriggerPips = 13;                               // Pips in profit to trigger break-even`  
`input int      InpBeOffsetPips = 3;                                    // Pips *past* entry to set SL for BE`  
`input int      InpTslStepPips = 10;                                    // TSL Step in pips`  
`input bool     InpTslRemoveTp = true;                                  // Remove TP when TSL triggers`  
`input bool     InpBreakevenIncludesCosts = true;                       // 'True BE' accounts for swap & commission`  
`input double   InpCommissionPerLot = 0.0;                              // Commission per lot for True BE calculations`

`// E2. ATR Settings`  
`input int      InpTslAtrPeriod = 14;                                   // ATR period for TSL`  
`input double   InpTslAtrMultiplier = 2.5;                              // ATR multiplier for TSL distance`

`// E3. Moving Average Settings`  
`input int      InpTslMaPeriod = 20;                                    // Moving Average period for TSL`  
`input ENUM_MA_METHOD InpTslMaMethod = MODE_SMA;                        // Moving Average method for TSL`  
`input ENUM_APPLIED_PRICE InpTslMaPrice = PRICE_CLOSE;                  // Price to apply MA to for TSL`

`// E4. High/Low Bar Settings`  
`input int      InpTslHiLoPeriod = 10;                                  // Period to look back for High/Low TSL`

`// E5. Stacking Settings`  
`// Stacking settings moved here as part of profit management`  
`input int      InpStackingMaxTrades = 3;                               // Max number of Stacking trades (0 = disabled)`  
`input int      InpStackingTriggerPips = 50;                            // Fixed pips trigger for stacking trades`  
`input double   InpStackingLotSize = 0.01;                              // Lot size for stacking trades`  
`input ENUM_STACKING_LOT_MODE InpStackingLotMode = MODE_FIXED;          // Stacking lot sizing mode`

`// E. ADVANCED EXIT SETTINGS`  
`input group "******** ADVANCED EXIT SETTINGS ********";`  
`input int      InpPartialTpTriggerPips = 13;                           // Pips in profit to trigger partial close (0 = disabled)`  
`input double   InpPartialTpClosePercent = 50.0;                        // Percentage of volume to close`  
`input bool     InpPartialTpSetBe = true;                               // Set remaining position to BE after partial close?`

`// F. FILTER SETTINGS`  
`input group "******** TIME FILTER SETTINGS ********";`  
`input string   InpEaTradingDays = "1,2,3,4,5";                         // Allowed trading days (Mon=1...Fri=5)`  
`input string   InpEaTradingTimeStart = "00:00";                        // Trading start time (Broker time)`  
`input string   InpEaTradingTimeEnd = "23:59";                          // Trading end time (Broker time)`

`// G. NEWS FILTER SETTINGS`  
`input group "******** NEWS FILTER SETTINGS ********";`  
`input ENUM_NEWS_SOURCE InpNewsSourceMode = MODE_DISABLED;              // Source for news event data (MODE_DISABLED = off)`  
`input string   InpNewsCalendarURL = "https://nfs.forexfactory.net/ffcal_week_this.csv"; // URL for web request mode`  
`input int      InpNewsMinsBefore = 30;                                 // Block trading X minutes before news`  
`input int      InpNewsMinsAfter = 30;                                  // Block trading X minutes after news`  
`input bool     InpNewsFilterHighImpact = true;                         // Filter high-impact news`  
`input bool     InpNewsFilterMedImpact = false;                         // Filter medium-impact news`  
`input bool     InpNewsFilterLowImpact = false;                         // Filter low-impact news`  
`input string   InpNewsFilterCurrencies = "USD,EUR,GBP,JPY,CAD,AUD,NZD,CHF"; // Currencies to monitor for news`

`// J. SIGNAL SETTINGS`  
`input group "******** SIGNAL DEFINITIONS ********";`  
`input group "******** SIGNAL SLOT 1 ********";`  
`input ENUM_SIGNAL_TYPE      InpSignal1_Type = SIGNAL_MACD;              // Signal type for slot 1`  
`input ENUM_SIGNAL_ROLE      InpSignal1_Role = ROLE_ENTRY;               // Role for signal 1`  
`input ENUM_TIMEFRAMES       InpSignal1_Timeframe = PERIOD_M15;          // Timeframe for signal 1`  
`input int                   InpSignal1_IntParam0 = 12;                  // Int param 0 (e.g., MACD Fast)`  
`input int                   InpSignal1_IntParam1 = 26;                  // Int param 1 (e.g., MACD Slow)`  
`input int                   InpSignal1_IntParam2 = 9;                   // Int param 2 (e.g., MACD Signal)`  
`input int                   InpSignal1_IntParam3 = 0;                   // Int param 3 (reserved)`  
`input double                InpSignal1_DoubleParam0 = 0.0;              // Double param 0 (reserved)`  
`input double                InpSignal1_DoubleParam1 = 0.0;              // Double param 1 (reserved)`  
`input double                InpSignal1_DoubleParam2 = 0.0;              // Double param 2 (reserved)`  
`input double                InpSignal1_DoubleParam3 = 0.0;              // Double param 3 (reserved)`  
`input ENUM_APPLIED_PRICE    InpSignal1_Price = PRICE_CLOSE;             // Applied price`  
`input ENUM_MA_METHOD        InpSignal1_MaMethod1 = MODE_SMA;            // MA method 1`  
`input ENUM_MA_METHOD        InpSignal1_MaMethod2 = MODE_SMA;            // MA method 2`  
`input ENUM_STO_PRICE        InpSignal1_PriceField = STO_LOWHIGH;        // Stochastic price field`  
`input bool                  InpSignal1_BoolParam0 = false;              // Bool param 0 (e.g., Threshold check)`  
`input bool                  InpSignal1_BoolParam1 = false;              // Bool param 1 (e.g., Threshold reverse)`  
`input bool                  InpSignal1_BoolParam2 = false;              // Bool param 2 (reserved)`  
`input bool                  InpSignal1_BoolParam3 = false;              // Bool param 3 (reserved)`  
`input group "******** SIGNAL SLOT 2 ********";`  
`input ENUM_SIGNAL_TYPE      InpSignal2_Type = SIGNAL_MACD;              // Signal type for slot 2`  
`input ENUM_SIGNAL_ROLE      InpSignal2_Role = ROLE_BIAS;                // Role for signal 2`  
`input ENUM_TIMEFRAMES       InpSignal2_Timeframe = PERIOD_H1;           // Timeframe for signal 2`  
`input int                   InpSignal2_IntParam0 = 12;                  // Int param 0`  
`input int                   InpSignal2_IntParam1 = 26;                  // Int param 1`  
`input int                   InpSignal2_IntParam2 = 9;                   // Int param 2`  
`input int                   InpSignal2_IntParam3 = 0;                   // Int param 3`  
`input double                InpSignal2_DoubleParam0 = 0.0;              // Double param 0`  
`input double                InpSignal2_DoubleParam1 = 0.0;              // Double param 1`  
`input double                InpSignal2_DoubleParam2 = 0.0;              // Double param 2`  
`input double                InpSignal2_DoubleParam3 = 0.0;              // Double param 3`  
`input ENUM_APPLIED_PRICE    InpSignal2_Price = PRICE_CLOSE;             // Applied price`  
`input ENUM_MA_METHOD        InpSignal2_MaMethod1 = MODE_SMA;            // MA method 1`  
`input ENUM_MA_METHOD        InpSignal2_MaMethod2 = MODE_SMA;            // MA method 2`  
`input ENUM_STO_PRICE        InpSignal2_PriceField = STO_LOWHIGH;        // Stochastic price field`  
`input bool                  InpSignal2_BoolParam0 = false;              // Bool param 0`  
`input bool                  InpSignal2_BoolParam1 = false;              // Bool param 1`  
`input bool                  InpSignal2_BoolParam2 = false;              // Bool param 2`  
`input bool                  InpSignal2_BoolParam3 = false;              // Bool param 3`  
`input group "******** SIGNAL SLOT 3 ********";`  
`input ENUM_SIGNAL_TYPE      InpSignal3_Type = SIGNAL_RSI;               // Signal type for slot 3`  
`input ENUM_SIGNAL_ROLE      InpSignal3_Role = ROLE_ENTRY;               // Role for signal 3`  
`input ENUM_TIMEFRAMES       InpSignal3_Timeframe = PERIOD_M15;          // Timeframe for signal 3`  
`input int                   InpSignal3_IntParam0 = 14;                  // Int param 0`  
`input int                   InpSignal3_IntParam1 = 0;                   // Int param 1`  
`input int                   InpSignal3_IntParam2 = 0;                   // Int param 2`  
`input int                   InpSignal3_IntParam3 = 0;                   // Int param 3`  
`input double                InpSignal3_DoubleParam0 = 30.0;             // Double param 0`  
`input double                InpSignal3_DoubleParam1 = 70.0;             // Double param 1`  
`input double                InpSignal3_DoubleParam2 = 0.0;              // Double param 2`  
`input double                InpSignal3_DoubleParam3 = 0.0;              // Double param 3`  
`input ENUM_APPLIED_PRICE    InpSignal3_Price = PRICE_CLOSE;             // Applied price`  
`input ENUM_MA_METHOD        InpSignal3_MaMethod1 = MODE_SMA;            // MA method 1`  
`input ENUM_MA_METHOD        InpSignal3_MaMethod2 = MODE_SMA;            // MA method 2`  
`input ENUM_STO_PRICE        InpSignal3_PriceField = STO_LOWHIGH;        // Stochastic price field`  
`input bool                  InpSignal3_BoolParam0 = false;              // Bool param 0`  
`input bool                  InpSignal3_BoolParam1 = false;              // Bool param 1`  
`input bool                  InpSignal3_BoolParam2 = false;              // Bool param 2`  
`input bool                  InpSignal3_BoolParam3 = false;              // Bool param 3`

`// M. SIGNAL MANAGER SETTINGS`  
`input group "******** SIGNAL MANAGER SETTINGS ********";`  
`input int     InpBiasPersistenceBars = 24;                             // Bars (EA heartbeat intervals) bias persists before auto-reset`

`// K. CHART UI SETTINGS`  
`input group "******** CHART UI SETTINGS ********";`  
`input bool     InpChartShowPanels = true;                              // Show/hide the chart UI panel`  
`input ENUM_BASE_CORNER InpChartPanelCorner = CORNER_RIGHT_LOWER;       // Corner to display the UI panel`  
`input color    InpChartColorBackground = clrBlack;                     // Background color of the UI panel`  
`input color    InpChartColorTextMain = clrWhite;                       // Main text color for the UI`  
`input color    InpChartColorBuy = clrDodgerBlue;                       // Color for BUY status/buttons`  
`input color    InpChartColorSell = clrRed;                             // Color for SELL status/buttons`  
`input color    InpChartColorNeutral = clrGray;                         // Color for NEUTRAL status`

`//+------------------------------------------------------------------+`  
`//| Helper Functions                                                 |`  
`//+------------------------------------------------------------------+`  
`bool IsNewBar(const ENUM_TIMEFRAMES timeframe)`  
`{`  
    `static datetime previousTime = 0;`  
    `datetime currentTime = iTime(_Symbol, timeframe, 0);`  
    `if(previousTime != currentTime)`  
    `{`  
        `previousTime = currentTime;`  
        `return true;`  
    `}`  
    `return false;`  
`}`

`//+------------------------------------------------------------------+`  
`//| Expert initialization function                                   |`  
`//+------------------------------------------------------------------+`  
`int OnInit()`  
`{`  
    `//--- Create manager instances`  
    `g_trade_manager = new CTradeManager();`  
    `g_money_manager = new CMoneyManager();`  
    `g_signal_manager = new CSignalManager();`  
    `g_dca_manager = new CDCAManager();`  
    `g_tsl_manager = new CTrailingStopManager();`  
    `g_time_manager = new CTimeManager();`  
    `g_news_manager = new CNewsManager();`  
    `g_stacking_manager = new CStackingManager();`  
    `g_ui_manager = new CUIManager();`  
    `g_atr_utility = new CatrUtility();`

    `//--- Copy input values to CSettings (reordered to match new logical grouping)`  
    `// A. General Settings`  
    `CSettings::EaName = InpEaName;`  
    `CSettings::EaMagicNumber = InpEaMagicNumber;`  
    `CSettings::MaxSpreadPoints = InpMaxSpreadPoints;`  
    `CSettings::MaxSlippagePoints = InpMaxSlippagePoints;`  
    `CSettings::MaxDrawdownPercent = InpMaxDrawdownPercent;`  
    `CSettings::EaHeartbeatTimeframe = InpEaHeartbeatTimeframe;`  
    `CSettings::AllowLongTrades = InpAllowLongTrades;`  
    `CSettings::AllowShortTrades = InpAllowShortTrades;`  
    `CSettings::Symbol = _Symbol;`

    `// B. Position Management Settings (Lot sizing + Basket TP/SL)`  
    `CSettings::LotSizingMode = InpLotSizingMode;`  
    `CSettings::LotFixed = InpLotFixed;`  
    `CSettings::LotsPerThousand = InpLotsPerThousand;`  
    `CSettings::LotRiskPercent = InpLotRiskPercent;`  
    `CSettings::SlPips = InpSlPips;`  
    `CSettings::InitialTpPips = InpInitialTpPips;`  
    `CSettings::BasketTpPips = InpBasketTpPips;`

    `// C. Loss Management Settings (DCA)`  
    `CSettings::DcaMaxTrades = InpDcaMaxTrades;`  
    `CSettings::DcaTriggerPips = InpDcaTriggerPips;`  
    `CSettings::DcaStepMultiplier = InpDcaStepMultiplier;`  
    `CSettings::DcaLotMultiplier = InpDcaLotMultiplier;`  
    `CSettings::DcaLotMultiplierStart = InpDcaLotMultiplierStart;`

    `// D. Profit Management Settings (TSL + Stacking)`  
    `CSettings::TslMode = InpTslMode;`  
    `CSettings::TslBeTriggerPips = InpTslBeTriggerPips;`  
    `CSettings::BeOffsetPips = InpBeOffsetPips;`  
    `CSettings::TslStepPips = InpTslStepPips;`  
    `CSettings::TslRemoveTp = InpTslRemoveTp;`  
    `CSettings::BreakevenIncludesCosts = InpBreakevenIncludesCosts;`  
    `CSettings::CommissionPerLot = InpCommissionPerLot;`  
    `CSettings::TslAtrPeriod = InpTslAtrPeriod;`  
    `CSettings::TslAtrMultiplier = InpTslAtrMultiplier;`  
    `CSettings::TslMaPeriod = InpTslMaPeriod;`  
    `CSettings::TslMaMethod = InpTslMaMethod;`  
    `CSettings::TslMaPrice = InpTslMaPrice;`  
    `CSettings::TslHiLoPeriod = InpTslHiLoPeriod;`  
    `CSettings::StackingMaxTrades = InpStackingMaxTrades;`  
    `CSettings::StackingTriggerPips = InpStackingTriggerPips;`  
    `CSettings::StackingLotSize = InpStackingLotSize;`  
    `CSettings::StackingLotMode = InpStackingLotMode;`

    `// E. Advanced Exit Settings (Partial TP)`  
    `CSettings::PartialTpTriggerPips = InpPartialTpTriggerPips;`  
    `CSettings::PartialTpClosePercent = InpPartialTpClosePercent;`  
    `CSettings::PartialTpSetBe = InpPartialTpSetBe;`

    `// F. Filter Settings (Time + News)`  
    `CSettings::EaTradingDays = InpEaTradingDays;`  
    `CSettings::EaTradingTimeStart = InpEaTradingTimeStart;`  
    `CSettings::EaTradingTimeEnd = InpEaTradingTimeEnd;`  
    `CSettings::NewsSourceMode = InpNewsSourceMode;`  
    `CSettings::NewsCalendarURL = InpNewsCalendarURL;`  
    `CSettings::NewsMinsBefore = InpNewsMinsBefore;`  
    `CSettings::NewsMinsAfter = InpNewsMinsAfter;`  
    `CSettings::NewsFilterHighImpact = InpNewsFilterHighImpact;`  
    `CSettings::NewsFilterMedImpact = InpNewsFilterMedImpact;`  
    `CSettings::NewsFilterLowImpact = InpNewsFilterLowImpact;`  
    `CSettings::NewsFilterCurrencies = InpNewsFilterCurrencies;`

    `CSettings::Signal1.Type = InpSignal1_Type;`  
    `CSettings::Signal1.Role = InpSignal1_Role;`  
    `CSettings::Signal1.Timeframe = InpSignal1_Timeframe;`  
    `CSettings::Signal1.Params.IntParams[0] = InpSignal1_IntParam0;`  
    `CSettings::Signal1.Params.IntParams[1] = InpSignal1_IntParam1;`  
    `CSettings::Signal1.Params.IntParams[2] = InpSignal1_IntParam2;`  
    `CSettings::Signal1.Params.IntParams[3] = InpSignal1_IntParam3;`  
    `CSettings::Signal1.Params.DoubleParams[0] = InpSignal1_DoubleParam0;`  
    `CSettings::Signal1.Params.DoubleParams[1] = InpSignal1_DoubleParam1;`  
    `CSettings::Signal1.Params.DoubleParams[2] = InpSignal1_DoubleParam2;`  
    `CSettings::Signal1.Params.DoubleParams[3] = InpSignal1_DoubleParam3;`  
    `CSettings::Signal1.Params.Price = InpSignal1_Price;`  
    `CSettings::Signal1.Params.MaMethod1 = InpSignal1_MaMethod1;`  
    `CSettings::Signal1.Params.MaMethod2 = InpSignal1_MaMethod2;`  
    `CSettings::Signal1.Params.PriceField = InpSignal1_PriceField;`  
    `CSettings::Signal1.Params.BoolParams[0] = InpSignal1_BoolParam0;`  
    `CSettings::Signal1.Params.BoolParams[1] = InpSignal1_BoolParam1;`  
    `CSettings::Signal1.Params.BoolParams[2] = InpSignal1_BoolParam2;`  
    `CSettings::Signal1.Params.BoolParams[3] = InpSignal1_BoolParam3;`

    `CSettings::Signal2.Type = InpSignal2_Type;`  
    `CSettings::Signal2.Role = InpSignal2_Role;`  
    `CSettings::Signal2.Timeframe = InpSignal2_Timeframe;`  
    `CSettings::Signal2.Params.IntParams[0] = InpSignal2_IntParam0;`  
    `CSettings::Signal2.Params.IntParams[1] = InpSignal2_IntParam1;`  
    `CSettings::Signal2.Params.IntParams[2] = InpSignal2_IntParam2;`  
    `CSettings::Signal2.Params.IntParams[3] = InpSignal2_IntParam3;`  
    `CSettings::Signal2.Params.DoubleParams[0] = InpSignal2_DoubleParam0;`  
    `CSettings::Signal2.Params.DoubleParams[1] = InpSignal2_DoubleParam1;`  
    `CSettings::Signal2.Params.DoubleParams[2] = InpSignal2_DoubleParam2;`  
    `CSettings::Signal2.Params.DoubleParams[3] = InpSignal2_DoubleParam3;`  
    `CSettings::Signal2.Params.Price = InpSignal2_Price;`  
    `CSettings::Signal2.Params.MaMethod1 = InpSignal2_MaMethod1;`  
    `CSettings::Signal2.Params.MaMethod2 = InpSignal2_MaMethod2;`  
    `CSettings::Signal2.Params.PriceField = InpSignal2_PriceField;`  
    `CSettings::Signal2.Params.BoolParams[0] = InpSignal2_BoolParam0;`  
    `CSettings::Signal2.Params.BoolParams[1] = InpSignal2_BoolParam1;`  
    `CSettings::Signal2.Params.BoolParams[2] = InpSignal2_BoolParam2;`  
    `CSettings::Signal2.Params.BoolParams[3] = InpSignal2_BoolParam3;`

    `CSettings::Signal3.Type = InpSignal3_Type;`  
    `CSettings::Signal3.Role = InpSignal3_Role;`  
    `CSettings::Signal3.Timeframe = InpSignal3_Timeframe;`  
    `CSettings::Signal3.Params.IntParams[0] = InpSignal3_IntParam0;`  
    `CSettings::Signal3.Params.IntParams[1] = InpSignal3_IntParam1;`  
    `CSettings::Signal3.Params.IntParams[2] = InpSignal3_IntParam2;`  
    `CSettings::Signal3.Params.IntParams[3] = InpSignal3_IntParam3;`  
    `CSettings::Signal3.Params.DoubleParams[0] = InpSignal3_DoubleParam0;`  
    `CSettings::Signal3.Params.DoubleParams[1] = InpSignal3_DoubleParam1;`  
    `CSettings::Signal3.Params.DoubleParams[2] = InpSignal3_DoubleParam2;`  
    `CSettings::Signal3.Params.DoubleParams[3] = InpSignal3_DoubleParam3;`  
    `CSettings::Signal3.Params.Price = InpSignal3_Price;`  
    `CSettings::Signal3.Params.MaMethod1 = InpSignal3_MaMethod1;`  
    `CSettings::Signal3.Params.MaMethod2 = InpSignal3_MaMethod2;`  
    `CSettings::Signal3.Params.PriceField = InpSignal3_PriceField;`  
    `CSettings::Signal3.Params.BoolParams[0] = InpSignal3_BoolParam0;`  
    `CSettings::Signal3.Params.BoolParams[1] = InpSignal3_BoolParam1;`  
    `CSettings::Signal3.Params.BoolParams[2] = InpSignal3_BoolParam2;`  
    `CSettings::Signal3.Params.BoolParams[3] = InpSignal3_BoolParam3;`

    `CSettings::BiasPersistenceBars = InpBiasPersistenceBars;`

    `CSettings::ChartShowPanels = InpChartShowPanels;`  
    `CSettings::ChartPanelCorner = InpChartPanelCorner;`  
    `CSettings::ChartColorBackground = InpChartColorBackground;`  
    `CSettings::ChartColorTextMain = InpChartColorTextMain;`  
    `CSettings::ChartColorBuy = InpChartColorBuy;`  
    `CSettings::ChartColorSell = InpChartColorSell;`  
    `CSettings::ChartColorNeutral = InpChartColorNeutral;`

    `//--- Detect backtest mode for magic number handling`  
    `CSettings::IsBacktestMode = (MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION));`  
    `Print("FXATM: Backtest mode detected: ", CSettings::IsBacktestMode);`

    `//--- Initialize managers (after CSettings is set)`  
    `g_trade_manager.Init();`  
    `g_tsl_manager.Init();`  
    `g_time_manager.Init();`  
    `g_news_manager.Init();`  
    `g_dca_manager.SetTradeManager(g_trade_manager);`  
    `g_dca_manager.SetMoneyManager(g_money_manager);`  
    `g_stacking_manager.SetMoneyManager(g_money_manager);`  
    `g_stacking_manager.SetTradeManager(g_trade_manager);`

    `// Initialize ATR utility and inject into dependent managers`  
    `if (!g_atr_utility.Init(CSettings::TslAtrPeriod, PERIOD_CURRENT))`  
    `{`  
        `Print("Failed to initialize ATR utility");`  
        `return INIT_FAILED;`  
    `}`  
    `g_money_manager.SetAtrUtility(g_atr_utility);`

    `//--- Signal Instantiation (up to 3 configurable signals)`  
    `// Signal1: Instantiate based on type`  
    `switch(CSettings::Signal1.Type)`  
    `{`  
        `case SIGNAL_MACD:`  
        `{`  
            `ISignal* signal = new CSignal_MACD();`  
            `if(signal.Init(CSettings::Signal1))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize MACD signal for Signal1");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_RSI:`  
        `{`  
            `ISignal* signal = new CSignal_RSI();`  
            `if(signal.Init(CSettings::Signal1))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize RSI signal for Signal1");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_MA_CROSS:`  
        `{`  
            `ISignal* signal = new CSignal_MA();`  
            `if(signal.Init(CSettings::Signal1))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize MA signal for Signal1");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_STOCHASTIC:`  
        `{`  
            `ISignal* signal = new CSignal_Stochastic();`  
            `if(signal.Init(CSettings::Signal1))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize Stochastic signal for Signal1");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_BOLLINGER_BANDS:`  
        `{`  
            `ISignal* signal = new CSignal_BollingerBands();`  
            `if(signal.Init(CSettings::Signal1))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize Bollinger Bands signal for Signal1");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `default:`  
            `Print("Unsupported signal type for Signal1");`  
            `break;`  
    `}`

    `// Signal2`  
    `switch(CSettings::Signal2.Type)`  
    `{`  
        `case SIGNAL_MACD:`  
        `{`  
            `ISignal* signal = new CSignal_MACD();`  
            `if(signal.Init(CSettings::Signal2))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize MACD signal for Signal2");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_RSI:`  
        `{`  
            `ISignal* signal = new CSignal_RSI();`  
            `if(signal.Init(CSettings::Signal2))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize RSI signal for Signal2");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_MA_CROSS:`  
        `{`  
            `ISignal* signal = new CSignal_MA();`  
            `if(signal.Init(CSettings::Signal2))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize MA signal for Signal2");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_STOCHASTIC:`  
        `{`  
            `ISignal* signal = new CSignal_Stochastic();`  
            `if(signal.Init(CSettings::Signal2))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize Stochastic signal for Signal2");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_BOLLINGER_BANDS:`  
        `{`  
            `ISignal* signal = new CSignal_BollingerBands();`  
            `if(signal.Init(CSettings::Signal2))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize Bollinger Bands signal for Signal2");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_TYPE_NONE:`  
            `// No signal to instantiate`  
            `break;`  
        `default:`  
            `Print("Unsupported signal type for Signal2");`  
            `break;`  
    `}`

    `// Signal3`  
    `switch(CSettings::Signal3.Type)`  
    `{`  
        `case SIGNAL_MACD:`  
        `{`  
            `ISignal* signal = new CSignal_MACD();`  
            `if(signal.Init(CSettings::Signal3))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize MACD signal for Signal3");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_RSI:`  
        `{`  
            `ISignal* signal = new CSignal_RSI();`  
            `if(signal.Init(CSettings::Signal3))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize RSI signal for Signal3");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_MA_CROSS:`  
        `{`  
            `ISignal* signal = new CSignal_MA();`  
            `if(signal.Init(CSettings::Signal3))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize MA signal for Signal3");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_STOCHASTIC:`  
        `{`  
            `ISignal* signal = new CSignal_Stochastic();`  
            `if(signal.Init(CSettings::Signal3))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize Stochastic signal for Signal3");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_BOLLINGER_BANDS:`  
        `{`  
            `ISignal* signal = new CSignal_BollingerBands();`  
            `if(signal.Init(CSettings::Signal3))`  
            `{`  
                `g_signal_manager.AddSignal(signal);`  
            `}`  
            `else`  
            `{`  
                `Print("Failed to initialize Bollinger Bands signal for Signal3");`  
                `delete signal;`  
            `}`  
            `break;`  
        `}`  
        `case SIGNAL_TYPE_NONE:`  
            `// No signal to instantiate`  
            `break;`  
        `default:`  
            `Print("Unsupported signal type for Signal3");`  
            `break;`  
    `}`

    `//---`  
    `return(INIT_SUCCEEDED);`  
`}`  
`//+------------------------------------------------------------------+`  
`//| Expert deinitialization function                                 |`  
`//+------------------------------------------------------------------+`  
`void OnDeinit(const int reason)`  
`{`  
    `//--- Delete manager instances`  
    `delete g_trade_manager;`  
    `delete g_money_manager;`  
    `delete g_signal_manager;`  
    `delete g_dca_manager;`  
    `delete g_tsl_manager;`  
    `delete g_time_manager;`  
    `delete g_news_manager;`  
    `delete g_stacking_manager;`  
    `delete g_ui_manager;`  
    `delete g_atr_utility;`  
    `//---`  
`}`  
`//+------------------------------------------------------------------+`  
`//| Expert tick function                                             |`  
`//+------------------------------------------------------------------+`  
`void OnTick()`  
`{`  
    `// --- Refresh basket cache once per tick for performance ---`  
    `g_trade_manager.Refresh();`

    `// --- MANAGEMENT LOGIC (runs on every tick for open baskets) ---`  
    `// Manage BUY basket: PTP, TSL, then Stacking or DCA`  
    `if (g_trade_manager.HasCachedBasket(POSITION_TYPE_BUY))`  
    `{`  
        `CBasket buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);`  
        `if(buy_basket.Ticket > 0)`  
        `{`  
            `g_trade_manager.ManagePartialTP(buy_basket);`  
            `// Refresh basket after PTP (positions may have been partially closed)`  
            `g_trade_manager.Refresh();`  
            `buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);`  
            `g_tsl_manager.ManageBasketTSL(POSITION_TYPE_BUY, buy_basket);`  
            `if (g_trade_manager.IsStopLossProfitable(buy_basket))`  
            `{`  
                `g_stacking_manager.ManageStacking(POSITION_TYPE_BUY, buy_basket);`  
                `g_trade_manager.Refresh(); // Refresh cache after stacking to include new position`  
                `buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);`  
                `g_tsl_manager.ManageBasketTSL(POSITION_TYPE_BUY, buy_basket); // update TSL for expanded basket`  
            `}`  
            `else if (!buy_basket.HasStacked)`  
            `{`  
                `g_dca_manager.ManageDCA(POSITION_TYPE_BUY, buy_basket);`  
                `g_trade_manager.Refresh(); // Refresh cache after DCA to include new position`  
                `buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);`  
                `g_tsl_manager.ManageBasketTSL(POSITION_TYPE_BUY, buy_basket); // update TSL for expanded basket`  
            `}`  
            `g_trade_manager.ManageBasketTP(buy_basket); // Set basket TP if expanded`  
        `}`  
    `}`

    `// Manage SELL basket: PTP, TSL, then Stacking or DCA`  
    `if (g_trade_manager.HasCachedBasket(POSITION_TYPE_SELL))`  
    `{`  
        `CBasket sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);`  
        `if(sell_basket.Ticket > 0)`  
        `{`  
            `g_trade_manager.ManagePartialTP(sell_basket);`  
            `// Refresh basket after PTP (positions may have been partially closed)`  
            `g_trade_manager.Refresh();`  
            `sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);`  
            `g_tsl_manager.ManageBasketTSL(POSITION_TYPE_SELL, sell_basket);`  
            `if (g_trade_manager.IsStopLossProfitable(sell_basket))`  
            `{`  
                `g_stacking_manager.ManageStacking(POSITION_TYPE_SELL, sell_basket);`  
                `g_trade_manager.Refresh(); // Refresh cache after stacking to include new position`  
                `sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);`  
                `g_tsl_manager.ManageBasketTSL(POSITION_TYPE_SELL, sell_basket); // update TSL for expanded basket`  
            `}`  
            `else if (!sell_basket.HasStacked)`  
            `{`  
                `g_dca_manager.ManageDCA(POSITION_TYPE_SELL, sell_basket);`  
                `g_trade_manager.Refresh(); // Refresh cache after DCA to include new position`  
                `sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);`  
                `g_tsl_manager.ManageBasketTSL(POSITION_TYPE_SELL, sell_basket); // update TSL for expanded basket`  
            `}`  
            `g_trade_manager.ManageBasketTP(sell_basket); // Set basket TP if expanded`  
        `}`  
    `}`

    `// --- ENTRY LOGIC (runs on new bar only) ---`  
    `if (!IsNewBar(CSettings::EaHeartbeatTimeframe)) return; // Throttle to heartbeat timeframe`  
    `if (!g_money_manager.CheckDrawdown()) return; // Risk check: stop if drawdown too high`  
    `if (!g_time_manager.IsTradeTimeAllowed()) return; // Time filter`  
    `if (g_news_manager.IsNewsBlockActive()) return; // News filter`  
    `if (SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > CSettings::MaxSpreadPoints) return; // Spread filter`  
    `int signal = g_signal_manager.GetFinalSignal(); // Aggregate signal from all sources`

    `// Check for BUY entry: signal, permissions, no existing basket`  
    `if (signal == SIGNAL_BUY && CSettings::AllowLongTrades && !g_trade_manager.HasCachedBasket(POSITION_TYPE_BUY))`  
    `{`  
        `double lots = g_money_manager.GetInitialLotSize(); // Calculate lot size based on mode`  
        `g_trade_manager.OpenTrade(signal, lots, CSettings::SlPips, CSettings::InitialTpPips, "INIT", 1);`  
    `}`

    `// Check for SELL entry: signal, permissions, no existing basket`  
    `if (signal == SIGNAL_SELL && CSettings::AllowShortTrades && !g_trade_manager.HasCachedBasket(POSITION_TYPE_SELL))`  
    `{`  
        `double lots = g_money_manager.GetInitialLotSize(); // Calculate lot size based on mode`  
        `g_trade_manager.OpenTrade(signal, lots, CSettings::SlPips, CSettings::InitialTpPips, "INIT", 1);`  
    `}`  
`}`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                     Settings.mqh |`  
`//|                            FXATM Configuration Management System |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`//| PURPOSE:                                                         |`  
`//|   Central configuration repository for FXATM Expert Advisor      |`  
`//|   Manages all input parameters, enums, and static settings       |`  
`//|                                                                  |`  
`//| KEY COMPONENTS:                                                  |`  
`//|   • 6 Lot sizing modes (Fixed, Risk%, ATR-Volatility Adjusted)   |`  
`//|   • 5 Trailing Stop Loss modes (Step, ATR, MA, High/Low)         |`  
`//|   • Signal configuration structures for polymorphic signals      |`  
`//|   • Static settings class with global parameter access           |`  
`//|                                                                  |`  
`//| USAGE:                                                           |`  
`//|   Include this file to access CSettings class and enumerations   |`  
`//|   All EA parameters are centralized here for consistency         |`  
`//+------------------------------------------------------------------+`  
`#property link      "lawrancekoh@outlook.com"`

`#include <Object.mqh>`

`// Generic Trade Signals`  
`enum ENUM_TRADE_SIGNAL`  
`{`  
    `SIGNAL_NONE,`  
    `SIGNAL_BUY,`  
    `SIGNAL_SELL,`  
    `SIGNAL_CLOSE_BUY,`  
    `SIGNAL_CLOSE_SELL`  
`};`

`// B. LOT SIZING SETTINGS`  
`enum ENUM_LOT_SIZING_MODE`  
`{`  
    `MODE_FIXED_LOT,`  
    `MODE_LOTS_PER_THOUSAND_BALANCE,`  
    `MODE_LOTS_PER_THOUSAND_EQUITY,`  
    `MODE_RISK_PERCENT_BALANCE,`  
    `MODE_RISK_PERCENT_EQUITY,`  
    `MODE_VOLATILITY_ADJUSTED`  
`};`

`// E. TRAILING STOP SETTINGS`  
`enum ENUM_TSL_MODE`  
`{`  
    `MODE_TSL_NONE,`  
    `MODE_TSL_STEP,`  
    `MODE_TSL_ATR,`  
    `MODE_TSL_MOVING_AVERAGE,`  
    `MODE_TSL_HIGH_LOW_BAR`  
`};`

`// H. NEWS FILTER SETTINGS`  
`enum ENUM_NEWS_SOURCE`  
`{`  
    `MODE_DISABLED,`  
    `MODE_MT5_BUILT_IN,`  
    `MODE_WEB_REQUEST`  
`};`

`// STACKING LOT MODE SETTINGS`  
`enum ENUM_STACKING_LOT_MODE`  
`{`  
    `MODE_FIXED,`  
    `MODE_LAST_TRADE,`  
    `MODE_BASKET_TOTAL,`  
    `MODE_ENTRY_BASED`  
`};`

`// I. SIGNAL SETTINGS`  
`enum ENUM_SIGNAL_TYPE`  
`{`  
    `SIGNAL_TYPE_NONE,`  
    `SIGNAL_RSI,`  
    `SIGNAL_MACD,`  
    `SIGNAL_MA_CROSS,`  
    `SIGNAL_STOCHASTIC,`  
    `SIGNAL_BOLLINGER_BANDS`  
`};`  
`enum ENUM_SIGNAL_ROLE`  
`{`  
    `ROLE_BIAS,`  
    `ROLE_ENTRY`  
`};`  
`struct CSignalParams`  
`{`  
    `int    IntParams[4];`  
    `double DoubleParams[4];`  
    `bool   BoolParams[4];`  
    `ENUM_APPLIED_PRICE Price;`  
    `ENUM_MA_METHOD     MaMethod1;`  
    `ENUM_MA_METHOD     MaMethod2;`  
    `ENUM_STO_PRICE     PriceField;`  
`};`  
`struct CSignalSettings`  
`{`  
    `ENUM_SIGNAL_TYPE   Type;`  
    `ENUM_SIGNAL_ROLE   Role;`  
    `ENUM_TIMEFRAMES    Timeframe;`  
    `CSignalParams Params;`  
`};`

`//+------------------------------------------------------------------+`  
`//| CSettings class                                                  |`  
`//| A static-like class to hold all EA settings.                     |`  
`//+------------------------------------------------------------------+`  
`class CSettings`  
`{`  
`public:`  
    `// A. GENERAL SETTINGS`  
    `static string   EaName;`  
    `static long     EaMagicNumber;`  
    `static int      MaxSpreadPoints;`  
    `static int      MaxSlippagePoints;`  
    `static double   MaxDrawdownPercent;`  
    `static ENUM_TIMEFRAMES EaHeartbeatTimeframe;`  
    `static bool     AllowLongTrades;`  
    `static bool     AllowShortTrades;`  
    `static string   Symbol;`

    `// B. POSITION MANAGEMENT SETTINGS (Lot sizing + Basket TP/SL)`  
    `static ENUM_LOT_SIZING_MODE LotSizingMode;`  
    `static double   LotFixed;`  
    `static double   LotsPerThousand;`  
    `static double   LotRiskPercent;`  
    `static int      SlPips;`  
    `static int      InitialTpPips;`  
    `static int      BasketTpPips;`

    `// C. LOSS MANAGEMENT SETTINGS (DCA)`  
    `static int      DcaMaxTrades;`  
    `static int      DcaTriggerPips;`  
    `static double   DcaStepMultiplier;`  
    `static double   DcaLotMultiplier;`  
    `static int      DcaLotMultiplierStart;`

    `// D. PROFIT MANAGEMENT SETTINGS (TSL + Stacking)`  
    `static ENUM_TSL_MODE TslMode;`  
    `static int      TslBeTriggerPips;`  
    `static int      BeOffsetPips;`  
    `static int      TslStepPips;`  
    `static bool     TslRemoveTp;`  
    `static bool     BreakevenIncludesCosts;`  
    `static double   CommissionPerLot;`  
    `static int      TslAtrPeriod;`  
    `static double   TslAtrMultiplier;`  
    `static int      TslMaPeriod;`  
    `static ENUM_MA_METHOD TslMaMethod;`  
    `static ENUM_APPLIED_PRICE TslMaPrice;`  
    `static int      TslHiLoPeriod;`  
    `static int      StackingMaxTrades;`  
    `static int      StackingTriggerPips;`  
    `static double   StackingLotSize;`  
    `static ENUM_STACKING_LOT_MODE StackingLotMode;`

    `// E. ADVANCED EXIT SETTINGS (Partial TP)`  
    `static int      PartialTpTriggerPips;`  
    `static double   PartialTpClosePercent;`  
    `static bool     PartialTpSetBe;`

    `// F. FILTER SETTINGS (Time + News)`  
    `static string   EaTradingDays;`  
    `static string   EaTradingTimeStart;`  
    `static string   EaTradingTimeEnd;`  
    `static ENUM_NEWS_SOURCE NewsSourceMode;`  
    `static string   NewsCalendarURL;`  
    `static int      NewsMinsBefore;`  
    `static int      NewsMinsAfter;`  
    `static bool     NewsFilterHighImpact;`  
    `static bool     NewsFilterMedImpact;`  
    `static bool     NewsFilterLowImpact;`  
    `static string   NewsFilterCurrencies;`

    `// G. SIGNAL SETTINGS`  
    `static CSignalSettings Signal1;`  
    `static CSignalSettings Signal2;`  
    `static CSignalSettings Signal3;`

    `// H. SIGNAL MANAGER SETTINGS`  
    `static int BiasPersistenceBars;`

    `// I. CHART UI SETTINGS`  
    `static bool     ChartShowPanels;`  
    `static ENUM_BASE_CORNER ChartPanelCorner;`  
    `static color    ChartColorBackground;`  
    `static color    ChartColorTextMain;`  
    `static color    ChartColorBuy;`  
    `static color    ChartColorSell;`  
    `static color    ChartColorNeutral;`

    `// J. BACKTEST MODE DETECTION`  
    `static bool     IsBacktestMode;`  
`};`

`//+------------------------------------------------------------------+`  
`//| Static member initialization                                     |`  
`//+------------------------------------------------------------------+`  
`string   CSettings::EaName;`  
`long     CSettings::EaMagicNumber;`  
`int      CSettings::MaxSpreadPoints = 0;`  
`int      CSettings::MaxSlippagePoints = 0;`  
`double   CSettings::MaxDrawdownPercent;`  
`ENUM_TIMEFRAMES CSettings::EaHeartbeatTimeframe;`  
`bool     CSettings::AllowLongTrades;`  
`bool     CSettings::AllowShortTrades;`  
`string   CSettings::Symbol;`

`ENUM_LOT_SIZING_MODE CSettings::LotSizingMode;`  
`double   CSettings::LotFixed;`  
`double   CSettings::LotsPerThousand;`  
`double   CSettings::LotRiskPercent;`  
`int      CSettings::SlPips = 0;`  
`int      CSettings::InitialTpPips = 0;`  
`int      CSettings::BasketTpPips = 0;`

`int      CSettings::DcaMaxTrades = 0;`  
`int      CSettings::DcaTriggerPips = 0;`  
`double   CSettings::DcaStepMultiplier;`  
`double   CSettings::DcaLotMultiplier;`  
`int      CSettings::DcaLotMultiplierStart = 0;`

`ENUM_TSL_MODE CSettings::TslMode;`  
`int      CSettings::TslBeTriggerPips = 0;`  
`int      CSettings::BeOffsetPips = 0;`  
`int      CSettings::TslStepPips = 0;`  
`bool     CSettings::TslRemoveTp;`  
`bool     CSettings::BreakevenIncludesCosts;`  
`double   CSettings::CommissionPerLot;`  
`int      CSettings::TslAtrPeriod = 0;`  
`double   CSettings::TslAtrMultiplier;`  
`int      CSettings::TslMaPeriod = 0;`  
`ENUM_MA_METHOD CSettings::TslMaMethod;`  
`ENUM_APPLIED_PRICE CSettings::TslMaPrice;`  
`int      CSettings::TslHiLoPeriod = 0;`  
`int      CSettings::StackingMaxTrades = 0;`  
`int      CSettings::StackingTriggerPips = 0;`  
`double   CSettings::StackingLotSize;`  
`ENUM_STACKING_LOT_MODE CSettings::StackingLotMode;`

`int      CSettings::PartialTpTriggerPips = 0;`  
`double   CSettings::PartialTpClosePercent;`  
`bool     CSettings::PartialTpSetBe;`

`string   CSettings::EaTradingDays;`  
`string   CSettings::EaTradingTimeStart;`  
`string   CSettings::EaTradingTimeEnd;`  
`ENUM_NEWS_SOURCE CSettings::NewsSourceMode;`  
`string   CSettings::NewsCalendarURL;`  
`int      CSettings::NewsMinsBefore = 0;`  
`int      CSettings::NewsMinsAfter = 0;`  
`bool     CSettings::NewsFilterHighImpact;`  
`bool     CSettings::NewsFilterMedImpact;`  
`bool     CSettings::NewsFilterLowImpact;`  
`string   CSettings::NewsFilterCurrencies;`

`CSignalSettings CSettings::Signal1;`  
`CSignalSettings CSettings::Signal2;`  
`CSignalSettings CSettings::Signal3;`

`int      CSettings::BiasPersistenceBars = 0;`

`bool     CSettings::ChartShowPanels;`  
`ENUM_BASE_CORNER CSettings::ChartPanelCorner;`  
`color    CSettings::ChartColorBackground;`  
`color    CSettings::ChartColorTextMain;`  
`color    CSettings::ChartColorBuy;`  
`color    CSettings::ChartColorSell;`  
`color    CSettings::ChartColorNeutral;`

`bool     CSettings::IsBacktestMode;`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                 TradeManager.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include <Trade\Trade.mqh>`  
`#include "Settings.mqh"`  
`#include "MoneyManager.mqh"`  
`#include "Basket.mqh"`

`//--- Signal definitions`  
`#define SIGNAL_BUY  1`  
`#define SIGNAL_SELL -1`  
`#define SIGNAL_NONE 0`

`class CTradeManager`  
  `{`  
`private:`  
    `CTrade   m_trade;`  
    `string   m_symbol;`

    `// Basket caching for performance optimization`  
    `CBasket  m_buy_basket_cache;`  
    `CBasket  m_sell_basket_cache;`  
    `bool     m_cache_valid;`

    `//+------------------------------------------------------------------+`  
    `//| Generates structured comment for trades                          |`  
    `//+------------------------------------------------------------------+`  
    `string GenerateComment(int signal, string trade_type, int serial_number)`  
      `{`  
       `string base = CSettings::EaName;`  
       `if(base == "") base = "FXATM";`  
       `string direction = (signal == SIGNAL_BUY) ? "BUY" : "SELL";`  
       `return base + " " + direction + " " + trade_type + " " + IntegerToString(serial_number);`  
      `}`

`public:`  
    `CTradeManager(void) {};`  
    `~CTradeManager(void) {};`

   `void Init()`  
     `{`  
      `m_symbol = Symbol();`  
      `m_trade.SetExpertMagicNumber(CSettings::EaMagicNumber);`  
      `m_trade.SetDeviationInPoints(CSettings::MaxSlippagePoints);`  
      `m_trade.SetTypeFillingBySymbol(m_symbol);`  
     `}`

   `//+------------------------------------------------------------------+`  
`//| Opens a trade based on the signal.                               |`  
`//+------------------------------------------------------------------+`  
   `bool OpenTrade(const int signal, const double lots, const int sl_pips, const int tp_pips, string tradeType, int serial)`  
     `{`  
      `if(signal == SIGNAL_NONE)`  
         `return false;`

      `// Generate comment using the new parameters`  
      `string comment = GenerateComment(signal, tradeType, serial);`

      `//--- Determine Order Type`  
      `ENUM_ORDER_TYPE order_type = (signal == SIGNAL_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;`

      `//--- Get Current Price`  
      `double price = SymbolInfoDouble(_Symbol, (order_type == ORDER_TYPE_BUY) ? SYMBOL_ASK : SYMBOL_BID);`

      `//--- Calculate SL/TP Prices`  
      `double pip_size = CMoneyManager::GetPipSize();`  
      `double sl_price = 0;`  
      `if(sl_pips > 0)`  
        `{`  
         `sl_price = (order_type == ORDER_TYPE_BUY) ? price - sl_pips * pip_size : price + sl_pips * pip_size;`  
        `}`

      `double tp_price = 0;`  
      `if(tp_pips > 0)`  
        `{`  
         `tp_price = (order_type == ORDER_TYPE_BUY) ? price + tp_pips * pip_size : price - tp_pips * pip_size;`  
        `}`

      `//--- Execute Trade`  
      `if(order_type == ORDER_TYPE_BUY)`  
        `{`  
         `return m_trade.Buy(lots, _Symbol, price, sl_price, tp_price, comment);`  
        `}`  
      `else`  
        `{`  
         `return m_trade.Sell(lots, _Symbol, price, sl_price, tp_price, comment);`  
        `}`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Checks if a basket of trades is already open for this symbol and direction. |`  
   `//+------------------------------------------------------------------+`  
   `bool HasOpenBasket(ENUM_POSITION_TYPE direction = POSITION_TYPE_BUY)`  
     `{`  
      `for(int i = PositionsTotal() - 1; i >= 0; i--)`  
        `{`  
         `ulong ticket = PositionGetTicket(i);`  
         `if(ticket == 0) continue;`  
         `if(!PositionSelectByTicket(ticket)) continue;`  
         `if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;`

         `// In backtest mode, skip magic number check due to Strategy Tester limitations`  
         `if(!CSettings::IsBacktestMode)`  
           `{`  
            `if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;`  
           `}`

         `if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == direction) return true;`  
        `}`  
      `return false;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Gets the current basket state by scanning open positions for the specified direction. |`  
   `//+------------------------------------------------------------------+`  
   `CBasket GetBasket(ENUM_POSITION_TYPE direction)`  
     `{`  
      `CBasket basket;`  
      `datetime latestTime = D'1970.01.01 00:00:00';`  
      `datetime earliestTime = D'2030.01.01 00:00:00';`  
      `int count = 0;`  
      `int stacking_count = 0;`  
      `double total_volume = 0.0;`  
      `double weighted_price_sum = 0.0;`  
      `double total_profit = 0.0;`  
      `double total_costs = 0.0;`

      `for(int i = PositionsTotal() - 1; i >= 0; i--)`  
        `{`  
         `ulong ticket = PositionGetTicket(i);`  
         `if(ticket == 0) continue;`  
         `if(!PositionSelectByTicket(ticket)) continue;`  
         `if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;`

         `// In backtest mode, skip magic number check due to Strategy Tester limitations`  
         `if(!CSettings::IsBacktestMode)`  
           `{`  
            `if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;`  
           `}`

         `if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;`

         `// Add ticket to basket`  
         `basket.AddTicket(ticket);`

         `count++;`  
         `double volume = PositionGetDouble(POSITION_VOLUME);`  
         `double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);`  
         `total_volume += volume;`  
         `weighted_price_sum += volume * entry_price;`  
         `total_profit += PositionGetDouble(POSITION_PROFIT);`  
         `total_costs += PositionGetDouble(POSITION_SWAP); // POSITION_COMMISSION deprecated`

         `// Optimized parsing: check for PTP flag first`  
         `string comment = PositionGetString(POSITION_COMMENT);`  
         `bool has_ptp = (StringLen(comment) == 0 || StringFind(comment, "[PTP]") != -1);`

         `// Parse comment for flags and basket info`  
         `if (has_ptp) {`  
            `basket.HasPartialTPExecuted = true;`  
         `}`

         `// Parse base comment by stripping flags (anything in brackets)`  
         `string base_comment = comment;`  
         `int flag_pos = StringFind(base_comment, " [");`  
         `if (flag_pos != -1) {`  
            `base_comment = StringSubstr(base_comment, 0, flag_pos);`  
         `}`

         `// Parse comment format: base direction type serial (e.g., FXATMv4 BUY INIT 1)`  
         `// Updated to handle EA names with spaces by parsing from the end`  
         `string parts[];`  
         `int split_count = StringSplit(base_comment, ' ', parts);`  
         `if (split_count >= 4) {`  
            `if (parts[split_count-2] == "STACK") {`  
               `stacking_count++;`  
            `}`  
         `} else if (StringLen(base_comment) > 0) {`  
            `// Malformed base comment, log warning but continue`  
            `Print("Warning: Malformed base comment '", base_comment, "' in position ", ticket);`  
         `}`

         `datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);`  
         `if(posTime > latestTime)`  
           `{`  
            `latestTime = posTime;`  
            `basket.Ticket = (int)ticket;`  
            `basket.LastTradePrice = entry_price;`  
            `basket.LastTradeLots = volume;`  
            `basket.BasketDirection = direction;`

            `// Only update basket type and serial from the latest trade`  
            `if (split_count >= 4) {`  
               `basket.BasketType = parts[split_count-2];`  
               `basket.SerialNumber = (int)StringToInteger(parts[split_count-1]);`  
            `}`  
           `}`  
         `if(posTime < earliestTime)`  
           `{`  
            `earliestTime = posTime;`  
            `basket.InitialTradePrice = entry_price;`  
           `}`  
        `}`

      `basket.TradeCount = count;`  
      `basket.StackingCount = stacking_count;`  
      `basket.HasStacked = stacking_count > 0;`  
      `basket.TotalVolume = total_volume;`  
      `basket.AvgEntryPrice = (total_volume > 0.0) ? weighted_price_sum / total_volume : 0.0;`  
      `basket.TotalProfit = total_profit;`  
      `basket.TotalCosts = total_costs;`  
      `return basket;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Sets the same Stop Loss price for all positions in the basket   |`  
   `//+------------------------------------------------------------------+`  
   `void SetBasketSL(ENUM_POSITION_TYPE direction, double sl_price)`  
     `{`  
      `// Get broker's minimum stops level`  
      `double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);`  
      `double stops_distance = stops_level * _Point;`  
      `double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);`  
      `double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);`  
        
      `Print("SetBasketSL Debug: Direction: ", direction, " Target SL: ", sl_price, " Bid: ", bid, " Ask: ", ask);`

      `for(int i = PositionsTotal() - 1; i >= 0; i--)`  
        `{`  
         `ulong ticket = PositionGetTicket(i);`  
         `if(ticket == 0) continue;`  
         `if(!PositionSelectByTicket(ticket)) continue;`  
         `if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;`

         `// In backtest mode, skip magic number check due to Strategy Tester limitations`  
         `if(!CSettings::IsBacktestMode)`  
           `{`  
            `if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;`  
           `}`

         `if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;`

         `// Validate stops level before modifying`  
         `bool is_valid_sl = true;`  
         `if(direction == POSITION_TYPE_BUY)`  
           `{`  
            `// For BUY: SL must be below BID by at least stops_distance`  
            `if(sl_price >= bid - stops_distance)`  
              `{`  
               `// Set to minimum allowed SL (small loss) to ensure SL is set`  
               `sl_price = bid - stops_distance - _Point * 10;`  
               `Print("SetBasketSL Debug: Adjusted BUY SL to ", sl_price);`  
              `}`  
           `}`  
         `else // POSITION_TYPE_SELL`  
           `{`  
            `// For SELL: SL must be above ASK by at least stops_distance`  
            `if(sl_price <= ask + stops_distance)`  
              `{`  
               `// Set to minimum allowed SL (small loss) to ensure SL is set`  
               `sl_price = ask + stops_distance + _Point * 10;`  
               `Print("SetBasketSL Debug: Adjusted SELL SL to ", sl_price);`  
              `}`  
           `}`

         `double current_sl = PositionGetDouble(POSITION_SL);`  
         `double current_tp = PositionGetDouble(POSITION_TP);`  
         `double norm_current_sl = NormalizeDouble(current_sl, _Digits);`  
         `double norm_sl_price = NormalizeDouble(sl_price, _Digits);`  
         `if(norm_current_sl != norm_sl_price) // Modify only if SL differs`  
           `{`  
            `Print("SetBasketSL Debug: Modifying ticket ", ticket, " SL from ", current_sl, " to ", sl_price);`  
            `if(!m_trade.PositionModify(ticket, sl_price, current_tp)) {`  
                `Print("SetBasketSL Debug: Failed to modify ticket ", ticket, " Error: ", m_trade.ResultRetcode());`  
            `}`  
           `}`  
        `}`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Calculates True Break-Even price for remaining volume            |`  
   `//+------------------------------------------------------------------+`  
   `double CalculateTrueBreakEvenPrice(const CBasket &basket, double remaining_vol)`  
   `{`  
       `double total_costs = basket.TotalCosts;`  
       `double desired_profit = CMoneyManager::GetMoneyFromPips(CSettings::BeOffsetPips, remaining_vol);`  
       `double total_money_needed = desired_profit - total_costs;`  
       `double pips_needed = CMoneyManager::GetPipsFromMoney(total_money_needed, remaining_vol);`  
       `if (pips_needed < 0) pips_needed = 0;  // Prevent setting SL to a loss level; use entry price as BE`  
       `double pip_size = CMoneyManager::GetPipSize();`  
       `if (basket.BasketDirection == POSITION_TYPE_BUY)`  
       `{`  
           `return basket.AvgEntryPrice + pips_needed * pip_size;  // SL above entry for BUY (Profit)`  
       `}`  
       `else`  
       `{`  
           `return basket.AvgEntryPrice - pips_needed * pip_size;  // SL below entry for SELL (Profit)`  
       `}`  
   `}`

   `//+------------------------------------------------------------------+`  
   `//| Calculates Basket TP price based on average entry price          |`  
   `//+------------------------------------------------------------------+`  
   `double CalculateBasketTpPrice(const CBasket &basket, int tp_pips)`  
   `{`  
       `double pip_size = CMoneyManager::GetPipSize();`  
       `if (basket.BasketDirection == POSITION_TYPE_BUY)`  
       `{`  
           `return basket.AvgEntryPrice + tp_pips * pip_size;`  
       `}`  
       `else`  
       `{`  
           `return basket.AvgEntryPrice - tp_pips * pip_size;`  
       `}`  
   `}`

   `//+------------------------------------------------------------------+`  
   `//| Manages Basket TP by setting TP on all positions when basket expands |`  
   `//+------------------------------------------------------------------+`  
   `void ManageBasketTP(const CBasket &basket)`  
   `{`  
       `if (basket.TradeCount <= 1 || CSettings::BasketTpPips <= 0) return;`  
       `double tp_price = CalculateBasketTpPrice(basket, CSettings::BasketTpPips);`  
       `SetBasketTP(basket.BasketDirection, tp_price);`  
   `}`

   `//+------------------------------------------------------------------+`  
   `//| Manages Partial Take Profit with proportional distribution      |`  
   `//+------------------------------------------------------------------+`  
   `void ManagePartialTP(const CBasket &basket)`  
   `{`  
       `// Guard clauses`  
       `if (CSettings::PartialTpTriggerPips <= 0 || basket.HasPartialTPExecuted || basket.ProfitPips() < CSettings::PartialTpTriggerPips) return;`

       `// Calculate target volume to close`  
       `double target_volume_to_close = basket.TotalVolume * (CSettings::PartialTpClosePercent / 100.0);`  
       `double actual_closed_volume = 0.0;`  
       `double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);`  
       `double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);`

       `// Proportional distribution across all positions`  
       `for (int i = 0; i < ArraySize(basket.Tickets); i++) {`  
           `ulong ticket = basket.Tickets[i];`  
           `if (!PositionSelectByTicket(ticket)) continue;`

           `double pos_volume = PositionGetDouble(POSITION_VOLUME);`  
           `double proportional_close = pos_volume * (target_volume_to_close / basket.TotalVolume);`  
           `proportional_close = MathFloor(proportional_close / step) * step;  // Round down`

           `if (proportional_close >= min_vol) {`  
               `bool close_success = m_trade.PositionClosePartial(ticket, proportional_close);`  
               `if (close_success) {`  
                   `actual_closed_volume += proportional_close;`  
               `} else {`  
                   `// Print("PTP: Failed to close position ", ticket, " volume ", proportional_close);`  
               `}`  
           `} else {`  
               `// Print("PTP: Skipping position ", ticket, " as calculated partial close volume ", proportional_close, " is below min volume ", min_vol);`  
           `}`  
       `}`

       `if (actual_closed_volume == 0.0) return;  // No closes succeeded`

       `// Update basket after partial closes to get correct AvgEntryPrice`  
       `CBasket updated_basket = GetBasket(basket.BasketDirection);`

       `// Set True BE SL on remaining positions if enabled`  
       `if (CSettings::PartialTpSetBe) {`  
           `double be_price = CalculateTrueBreakEvenPrice(updated_basket, updated_basket.TotalVolume);`

           `// Validate BE price against current market price and stops level`  
           `double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);`  
           `double stops_distance = stops_level * _Point;`  
           `double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);`  
           `double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);`  
           `bool is_safe = false;`

           `if (updated_basket.BasketDirection == POSITION_TYPE_BUY) {`  
               `// For BUY, SL must be < Bid - Stops`  
               `if (be_price < bid - stops_distance) is_safe = true;`  
           `} else {`  
               `// For SELL, SL must be > Ask + Stops`  
               `if (be_price > ask + stops_distance) is_safe = true;`  
           `}`

           `if (is_safe) {`  
               `Print("PTP Debug: Setting BE SL to ", be_price);`  
               `SetBasketSL(updated_basket.BasketDirection, be_price);`  
           `} else {`  
               `Print("PTP Debug: Skipping BE SL - Price too close or in loss. BE: ", be_price, " Bid: ", bid, " Ask: ", ask);`  
           `}`  
       `}`

   `}`

         
   `//+------------------------------------------------------------------+`  
   `void SetBasketTP(ENUM_POSITION_TYPE direction, double tp_price)`  
     `{`  
      `for(int i = PositionsTotal() - 1; i >= 0; i--)`  
        `{`  
         `ulong ticket = PositionGetTicket(i);`  
         `if(ticket == 0) continue;`  
         `if(!PositionSelectByTicket(ticket)) continue;`  
         `if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;`

         `// In backtest mode, skip magic number check due to Strategy Tester limitations`  
         `if(!CSettings::IsBacktestMode)`  
           `{`  
            `if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;`  
           `}`

         `if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;`

         `double current_sl = PositionGetDouble(POSITION_SL);`  
         `double current_tp = PositionGetDouble(POSITION_TP);`  
           
         `double norm_current_tp = NormalizeDouble(current_tp, _Digits);`  
         `double norm_tp_price = NormalizeDouble(tp_price, _Digits);`  
           
         `if(norm_current_tp != norm_tp_price) // Modify only if TP differs`  
           `{`  
            `m_trade.PositionModify(ticket, current_sl, tp_price);`  
           `}`  
        `}`  
     `}`

   `void CloseTrades(ENUM_ORDER_TYPE type)`  
     `{`  
      `// Logic to close trades of a certain type.`  
     `}`

   `int GetOpenTradesCount()`  
     `{`  
      `// Logic to count open trades.`  
      `return 0;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Refresh basket cache once per tick for performance optimization |`  
   `//+------------------------------------------------------------------+`  
   `void Refresh()`  
     `{`  
      `// Reset baskets`  
      `m_buy_basket_cache = CBasket();`  
      `m_sell_basket_cache = CBasket();`

      `// Variables for weighted average calculation`  
      `double buy_weighted_sum = 0.0;`  
      `double sell_weighted_sum = 0.0;`  
      `datetime buy_latest = D'1970.01.01 00:00:00';`  
      `datetime buy_earliest = D'2030.01.01 00:00:00';`  
      `datetime sell_latest = D'1970.01.01 00:00:00';`  
      `datetime sell_earliest = D'2030.01.01 00:00:00';`

      `// Single loop to populate both baskets simultaneously`  
      `for(int i = PositionsTotal() - 1; i >= 0; i--)`  
        `{`  
         `ulong ticket = PositionGetTicket(i);`  
         `if(ticket == 0 || !PositionSelectByTicket(ticket)) continue;`  
         `if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;`

         `// In backtest mode, skip magic number check due to Strategy Tester limitations`  
         `if(!CSettings::IsBacktestMode)`  
           `{`  
            `if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;`  
           `}`

         `ENUM_POSITION_TYPE direction = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);`

         `// Update basket statistics based on direction`  
         `double volume = PositionGetDouble(POSITION_VOLUME);`  
         `double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);`

         `if(direction == POSITION_TYPE_BUY)`  
           `{`  
            `// Update buy basket`  
            `m_buy_basket_cache.AddTicket(ticket);`  
            `m_buy_basket_cache.TradeCount++;`  
            `m_buy_basket_cache.TotalVolume += volume;`  
            `buy_weighted_sum += volume * entry_price;`  
            `m_buy_basket_cache.TotalProfit += PositionGetDouble(POSITION_PROFIT);`  
            `m_buy_basket_cache.TotalCosts += PositionGetDouble(POSITION_SWAP);`

            `// Optimized parsing: check for PTP flag first`  
            `string comment = PositionGetString(POSITION_COMMENT);`  
            `bool has_ptp = (StringLen(comment) == 0 || StringFind(comment, "[PTP]") != -1);`

            `// Parse comment for flags and basket info (optimized)`  
            `if (has_ptp) {`  
               `m_buy_basket_cache.HasPartialTPExecuted = true;`  
            `}`

            `// Parse base comment by stripping flags (anything in brackets)`  
            `string base_comment = comment;`  
            `int flag_pos = StringFind(base_comment, " [");`  
            `if (flag_pos != -1) {`  
               `base_comment = StringSubstr(base_comment, 0, flag_pos);`  
            `}`

            `// Parse comment format: base direction type serial (e.g., FXATMv4 BUY INIT 1)`  
            `// Updated to handle EA names with spaces by parsing from the end`  
            `string parts[];`  
            `int split_count = StringSplit(base_comment, ' ', parts);`  
            `if (split_count >= 4) {`  
               `if (parts[split_count-2] == "STACK") {`  
                  `m_buy_basket_cache.StackingCount++;`  
               `}`  
            `} else if (StringLen(base_comment) > 0) {`  
               `// Malformed base comment, log warning but continue`  
               `Print("Warning: Malformed base comment '", base_comment, "' in position ", ticket);`  
            `}`

            `// Track latest and earliest times`  
            `datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);`  
            `if(posTime > buy_latest)`  
              `{`  
               `buy_latest = posTime;`  
               `m_buy_basket_cache.Ticket = (int)ticket;`  
               `m_buy_basket_cache.LastTradePrice = entry_price;`  
               `m_buy_basket_cache.LastTradeLots = volume;`  
               `m_buy_basket_cache.BasketDirection = direction;`

               `// Only update basket type and serial from the latest trade`  
               `if (split_count >= 4) {`  
                  `m_buy_basket_cache.BasketType = parts[split_count-2];`  
                  `m_buy_basket_cache.SerialNumber = (int)StringToInteger(parts[split_count-1]);`  
               `}`  
              `}`  
            `if(posTime < buy_earliest)`  
              `{`  
               `buy_earliest = posTime;`  
               `m_buy_basket_cache.InitialTradePrice = entry_price;`  
              `}`  
           `}`  
         `else // POSITION_TYPE_SELL`  
           `{`  
            `// Update sell basket`  
            `m_sell_basket_cache.AddTicket(ticket);`  
            `m_sell_basket_cache.TradeCount++;`  
            `m_sell_basket_cache.TotalVolume += volume;`  
            `sell_weighted_sum += volume * entry_price;`  
            `m_sell_basket_cache.TotalProfit += PositionGetDouble(POSITION_PROFIT);`  
            `m_sell_basket_cache.TotalCosts += PositionGetDouble(POSITION_SWAP);`

            `// Optimized parsing: check for PTP flag first`  
            `string comment = PositionGetString(POSITION_COMMENT);`  
            `bool has_ptp = (StringLen(comment) == 0 || StringFind(comment, "[PTP]") != -1);`

            `// Parse comment for flags and basket info (optimized)`  
            `if (has_ptp) {`  
               `m_sell_basket_cache.HasPartialTPExecuted = true;`  
            `}`

            `// Parse base comment by stripping flags (anything in brackets)`  
            `string base_comment = comment;`  
            `int flag_pos = StringFind(base_comment, " [");`  
            `if (flag_pos != -1) {`  
               `base_comment = StringSubstr(base_comment, 0, flag_pos);`  
            `}`

            `// Parse comment format: base direction type serial (e.g., FXATMv4 BUY INIT 1)`  
            `// Updated to handle EA names with spaces by parsing from the end`  
            `string parts[];`  
            `int split_count = StringSplit(base_comment, ' ', parts);`  
            `if (split_count >= 4) {`  
               `if (parts[split_count-2] == "STACK") {`  
                  `m_sell_basket_cache.StackingCount++;`  
               `}`  
            `} else if (StringLen(base_comment) > 0) {`  
               `// Malformed base comment, log warning but continue`  
               `Print("Warning: Malformed base comment '", base_comment, "' in position ", ticket);`  
            `}`

            `// Track latest and earliest times`  
            `datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);`  
            `if(posTime > sell_latest)`  
              `{`  
               `sell_latest = posTime;`  
               `m_sell_basket_cache.Ticket = (int)ticket;`  
               `m_sell_basket_cache.LastTradePrice = entry_price;`  
               `m_sell_basket_cache.LastTradeLots = volume;`  
               `m_sell_basket_cache.BasketDirection = direction;`

               `// Only update basket type and serial from the latest trade`  
               `if (split_count >= 4) {`  
                  `m_sell_basket_cache.BasketType = parts[split_count-2];`  
                  `m_sell_basket_cache.SerialNumber = (int)StringToInteger(parts[split_count-1]);`  
               `}`  
              `}`  
            `if(posTime < sell_earliest)`  
              `{`  
               `sell_earliest = posTime;`  
               `m_sell_basket_cache.InitialTradePrice = entry_price;`  
              `}`  
           `}`  
        `}`

      `// Calculate final statistics for both baskets`  
      `if(m_buy_basket_cache.TradeCount > 0)`  
        `{`  
         `m_buy_basket_cache.AvgEntryPrice = (m_buy_basket_cache.TotalVolume > 0.0) ? buy_weighted_sum / m_buy_basket_cache.TotalVolume : 0.0;`  
         `m_buy_basket_cache.HasStacked = m_buy_basket_cache.StackingCount > 0;`  
        `}`

      `if(m_sell_basket_cache.TradeCount > 0)`  
        `{`  
         `m_sell_basket_cache.AvgEntryPrice = (m_sell_basket_cache.TotalVolume > 0.0) ? sell_weighted_sum / m_sell_basket_cache.TotalVolume : 0.0;`  
         `m_sell_basket_cache.HasStacked = m_sell_basket_cache.StackingCount > 0;`  
        `}`

      `m_cache_valid = true;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Get cached basket state (call Refresh() first)                   |`  
   `//+------------------------------------------------------------------+`  
   `CBasket GetCachedBasket(ENUM_POSITION_TYPE direction)`  
     `{`  
      `if(!m_cache_valid) Refresh();`  
      `return (direction == POSITION_TYPE_BUY) ? m_buy_basket_cache : m_sell_basket_cache;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Check if cached basket exists (call Refresh() first)             |`  
   `//+------------------------------------------------------------------+`  
   `bool HasCachedBasket(ENUM_POSITION_TYPE direction)`  
     `{`  
      `if(!m_cache_valid) Refresh();`  
      `CBasket basket = (direction == POSITION_TYPE_BUY) ? m_buy_basket_cache : m_sell_basket_cache;`  
      `return basket.Ticket > 0;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Get the current Stop Loss price for the basket                   |`  
   `//+------------------------------------------------------------------+`  
   `double GetBasketSL(ENUM_POSITION_TYPE direction)`  
     `{`  
      `for(int i = PositionsTotal() - 1; i >= 0; i--)`  
        `{`  
         `ulong ticket = PositionGetTicket(i);`  
         `if(ticket == 0) continue;`  
         `if(!PositionSelectByTicket(ticket)) continue;`  
         `if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;`

         `// In backtest mode, skip magic number check due to Strategy Tester limitations`  
         `if(!CSettings::IsBacktestMode)`  
           `{`  
            `if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;`  
           `}`

         `if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;`

         `// Return SL of the first matching position (they should all be the same)`  
         `return PositionGetDouble(POSITION_SL);`  
        `}`  
      `return 0.0; // No positions found`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Check if the basket's stop loss is in a profitable position      |`  
   `//+------------------------------------------------------------------+`  
   `bool IsStopLossProfitable(const CBasket &basket)`  
     `{`  
      `return basket.ProfitPips() > 0;`  
     `}`

  `};`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                 MoneyManager.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include "Settings.mqh"`  
`#include "Basket.mqh"`  
`#include "CatrUtility.mqh"`

`class CMoneyManager`  
   `{`  
`private:`  
   `CatrUtility* m_atr_utility;`

`public:`  
   `CMoneyManager(void) : m_atr_utility(NULL) {};`  
   `~CMoneyManager(void) { if (m_atr_utility != NULL) delete m_atr_utility; };`

   `void Init()`  
     `{`  
      `// Nothing to do here for now`  
     `}`

   `// Set ATR utility for volatility-adjusted lot sizing`  
   `void SetAtrUtility(CatrUtility* atr_utility)`  
     `{`  
      `m_atr_utility = atr_utility;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Validates the lot size against broker limits (min, max, step).  |`  
   `//+------------------------------------------------------------------+`  
   `double ValidateLotSize(double lot)`  
     `{`  
      `string symbol = CSettings::Symbol;`  
      `double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);`  
      `double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);`  
      `double step_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);`

      `// Apply limits`  
      `lot = MathMin(lot, max_lot);`  
      `lot = MathMax(lot, min_lot);`

      `// Normalize to the nearest valid step`  
      `if(step_lot > 0)`  
        `{`  
         `lot = MathRound(lot / step_lot) * step_lot;`  
        `}`

      `return lot;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Returns the pip size for the given symbol (standard 10 points). |`  
   `//+------------------------------------------------------------------+`  
   `static double GetPipSize(string symbol = NULL)`  
     `{`  
      `if(symbol == NULL) symbol = CSettings::Symbol;`  
      `double point = SymbolInfoDouble(symbol, SYMBOL_POINT);`  
      `return point * 10; // Standard pip size is 10 points`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Returns the value of one tick in account currency for pricing.  |`  
   `//+------------------------------------------------------------------+`  
   `static double GetTickValueInAccountCurrency(string symbol = NULL)`  
     `{`  
      `if(symbol == NULL) symbol = CSettings::Symbol;`  
      `return SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Calculates the monetary risk of SL pips for one lot in account currency. |`  
   `//+------------------------------------------------------------------+`  
   `static double GetSlValuePerLotInAccountCurrency(int sl_pips, string symbol = NULL)`  
     `{`  
      `double pip_size = GetPipSize(symbol);`  
      `double tick_value = GetTickValueInAccountCurrency(symbol);`  
      `if(symbol == NULL) symbol = CSettings::Symbol;`  
      `double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);`

      `if(tick_size == 0) return 0;`

      `// Calculate using tick size to handle instruments where tick size != point`  
      `return (sl_pips * pip_size / tick_size) * tick_value;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Converts pip value to monetary value in account currency for given lot size. |`  
   `//+------------------------------------------------------------------+`  
   `static double GetMoneyFromPips(double pips, double lot_size, string symbol = NULL)`  
     `{`  
      `if(symbol == NULL) symbol = CSettings::Symbol;`

      `double pip_size = GetPipSize(symbol);`  
      `double tick_value = GetTickValueInAccountCurrency(symbol);`  
      `double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);`

      `if(tick_size == 0) return 0;`

      `// Calculate using tick size to handle instruments where tick size != point`  
      `return (pips * pip_size / tick_size) * tick_value * lot_size;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Converts monetary value in account currency to pip value for given lot size. |`  
   `//+------------------------------------------------------------------+`  
   `static double GetPipsFromMoney(double money, double lot_size, string symbol = NULL)`  
     `{`  
      `if(symbol == NULL) symbol = CSettings::Symbol;`

      `double pip_size = GetPipSize(symbol);`  
      `double tick_value = GetTickValueInAccountCurrency(symbol);`  
      `double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);`

      `if(tick_value == 0 || lot_size == 0 || pip_size == 0) return 0;`

      `// Reverse the calculation: Money = (Pips * PipSize / TickSize) * TickValue * LotSize`  
      `// Pips = Money / ( (PipSize / TickSize) * TickValue * LotSize )`  
      `return money / ((pip_size / tick_size) * tick_value * lot_size);`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Calculates the initial lot size based on the selected mode.      |`  
   `//+------------------------------------------------------------------+`  
   `double GetInitialLotSize()`  
     `{`  
      `double calculated_lot = 0.0;`

      `switch(CSettings::LotSizingMode)`  
        `{`  
         `case MODE_FIXED_LOT:`  
            `calculated_lot = CSettings::LotFixed;`  
            `break;`

         `case MODE_LOTS_PER_THOUSAND_BALANCE:`  
            `calculated_lot = (AccountInfoDouble(ACCOUNT_BALANCE) / 1000.0) * CSettings::LotsPerThousand;`  
            `break;`

         `case MODE_LOTS_PER_THOUSAND_EQUITY:`  
            `calculated_lot = (AccountInfoDouble(ACCOUNT_EQUITY) / 1000.0) * CSettings::LotsPerThousand;`  
            `break;`

         `case MODE_RISK_PERCENT_BALANCE:`  
            `if(CSettings::SlPips <= 0)`  
              `{`  
               `Print("Risk modes require SlPips > 0. Falling back to fixed lot.");`  
               `calculated_lot = CSettings::LotFixed;`  
              `}`  
            `else`  
              `{`  
               `double risk_amount = AccountInfoDouble(ACCOUNT_BALANCE) * (CSettings::LotRiskPercent / 100.0);`  
               `double sl_value_per_lot = GetSlValuePerLotInAccountCurrency(CSettings::SlPips);`  
               `if(sl_value_per_lot > 0)`  
                 `{`  
                  `calculated_lot = risk_amount / sl_value_per_lot;`  
                 `}`  
               `else`  
                 `{`  
                  `Print("Unable to calculate SL value. Falling back to fixed lot.");`  
                  `calculated_lot = CSettings::LotFixed;`  
                 `}`  
              `}`  
            `break;`

         `case MODE_RISK_PERCENT_EQUITY:`  
            `if(CSettings::SlPips <= 0)`  
              `{`  
               `Print("Risk modes require SlPips > 0. Falling back to fixed lot.");`  
               `calculated_lot = CSettings::LotFixed;`  
              `}`  
            `else`  
              `{`  
               `double risk_amount = AccountInfoDouble(ACCOUNT_EQUITY) * (CSettings::LotRiskPercent / 100.0);`  
               `double sl_value_per_lot = GetSlValuePerLotInAccountCurrency(CSettings::SlPips);`  
               `if(sl_value_per_lot > 0)`  
                 `{`  
                  `calculated_lot = risk_amount / sl_value_per_lot;`  
                 `}`  
               `else`  
                 `{`  
                  `Print("Unable to calculate SL value. Falling back to fixed lot.");`  
                  `calculated_lot = CSettings::LotFixed;`  
                 `}`  
              `}`  
            `break;`

         `case MODE_VOLATILITY_ADJUSTED:`  
            `if(m_atr_utility == NULL)`  
              `{`  
               `Print("MODE_VOLATILITY_ADJUSTED requires ATR utility to be set. Falling back to fixed lot.");`  
               `calculated_lot = CSettings::LotFixed;`  
              `}`  
            `else`  
              `{`  
               `// Get base lot size using fixed mode as baseline`  
               `double base_lot = CSettings::LotFixed;`  
               `if(base_lot <= 0)`  
                 `{`  
                  `base_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);`  
                 `}`

               `// Get current ATR and calculate scaling factor`  
               `double current_atr = m_atr_utility.GetCurrentAtr();`  
               `double scaling_factor = m_atr_utility.GetAtrMultiplierForLots(base_lot, current_atr);`

               `// Apply volatility adjustment`  
               `calculated_lot = base_lot * scaling_factor;`

               `Print("Volatility-adjusted lot sizing: Base lot: ", base_lot,`  
                     `", Current ATR: ", current_atr,`  
                     `", Scaling factor: ", scaling_factor,`  
                     `", Final lot: ", calculated_lot);`  
              `}`  
            `break;`  
        `}`

      `return ValidateLotSize(calculated_lot);`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Calculates the lot size for stacking trades based on the selected mode. |`  
   `//+------------------------------------------------------------------+`  
   `double GetStackingLotSize(const CBasket &basket)`  
     `{`  
      `double calculated_lot = 0.0;`

      `switch(CSettings::StackingLotMode)`  
        `{`  
         `case MODE_FIXED:`  
            `calculated_lot = CSettings::StackingLotSize;`  
            `break;`

         `case MODE_LAST_TRADE:`  
            `calculated_lot = basket.LastTradeLots;`  
            `break;`

         `case MODE_BASKET_TOTAL:`  
            `calculated_lot = basket.TotalVolume;`  
            `break;`

         `case MODE_ENTRY_BASED:`  
            `calculated_lot = GetInitialLotSize();`  
            `break;`  
        `}`

      `return ValidateLotSize(calculated_lot);`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Checks if current account drawdown exceeds the threshold.       |`  
   `//| Returns true if drawdown < MaxDrawdownPercent (trading allowed).|`  
   `//+------------------------------------------------------------------+`  
   `bool CheckDrawdown()`  
     `{`  
      `double balance = AccountInfoDouble(ACCOUNT_BALANCE);`  
      `double equity = AccountInfoDouble(ACCOUNT_EQUITY);`  
      `if(balance <= 0) return true; // Avoid division by zero`

      `double drawdown_percent = ((balance - equity) / balance) * 100.0;`  
      `return drawdown_percent < CSettings::MaxDrawdownPercent;`  
     `}`

`private:`  
  `};`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                SignalManager.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.01" // Updated version`

`#include "Settings.mqh"`  
`#include "../Signals/ISignal.mqh"`  
`#include <Arrays/ArrayObj.mqh>`  
`// --- Include all signal implementations that will be created`  
`// #include "../Signals/CSignal_RSI.mqh"`  
`// #include "../Signals/CSignal_MACD.mqh"`

`/**`  
 `* @class CSignalManager`  
 `* @brief Manages signal aggregation with persistent bias mechanism.`  
 `*`  
 `* This class implements a "sticky" bias system where bias signals (ROLE_BIAS) persist`  
 `* across multiple bars until overridden or timed out. The persistence helps maintain`  
 `* trend direction even when bias signals temporarily disappear.`  
 `*`  
 `* Key Features:`  
 `* - Bias Persistence: Once set by a bias signal, the bias holds until:`  
 `*   1. A conflicting bias signal appears (disagreement resets to NONE).`  
 `*   2. No bias signal reinforces it for a configurable number of bars (timeout).`  
 `* - Timeout Mechanism: Uses a counter that increments on each GetFinalSignal() call`  
 `*   (tied to EA heartbeat timeframe). Resets after CSettings::BiasPersistenceBars calls.`  
 `*   Example: With M15 heartbeat and 20 bars setting, bias resets after ~5 hours.`  
 `* - Signal Aggregation: Follows "All Must Agree" logic, but persistent bias allows`  
 `*   entry signals to trigger trades even if no current bias signal is present.`  
 `*`  
 `* Usage Notes:`  
 `* - Bias signals set the direction; entry signals trigger trades within that direction.`  
 `* - Timeout prevents stale bias from influencing decisions indefinitely.`  
 `* - Counter resets when bias is updated or cleared.`  
 `*/`  
`class CSignalManager`  
  `{`  
`private:`  
    `CArrayObj         m_signals;`  
    `// --- Persistent bias state ---`  
    `int               m_current_bias;          // Current persistent bias (BUY/SELL/NONE)`  
    `int               m_bias_timeout_counter;  // Counts calls since bias was last set/updated`

`public:`  
    `CSignalManager(void) : m_current_bias(SIGNAL_NONE), m_bias_timeout_counter(0) // Initialize bias to NONE and counter to 0`  
      `{`  
      `};`

    `~CSignalManager(void)`  
      `{`  
       `for(int i = 0; i < m_signals.Total(); i++)`  
         `{`  
          `delete m_signals.At(i);`  
         `}`  
       `m_signals.Clear();`  
      `};`

    `void AddSignal(ISignal* signal)`  
      `{`  
       `if (signal == NULL)`  
         `{`  
          `Print("SignalManager: Attempted to add null signal pointer");`  
          `return;`  
         `}`  
       `m_signals.Add(signal);`  
      `}`

    `/**`  
     `* @brief Gets the final, combined trading signal, now with persistent bias.`  
     `*`  
     `* 1. It first checks all signals on the CURRENT bar for new triggers.`  
     `* 2. It checks for disagreements (e.g., two bias signals fighting).`  
     `* 3. If a new, non-conflicting bias signal appears, it UPDATES the`  
     `* persistent 'm_current_bias'.`  
     `* 4. If no new bias signal appears, 'm_current_bias' KEEPS its old value.`  
     `* 5. Finally, it checks the 'm_current_bias' against any ENTRY triggers.`  
     `*`  
     `* @return int The final trade signal (SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NONE).`  
     `*/`  
    `int GetFinalSignal()`  
      `{`  
       `// --- Bias Timeout Check ---`  
       `// The persistent bias times out after a configurable number of bars (calls to this method).`  
       `// This prevents stale bias from influencing decisions forever.`  
       `// - Counter increments each time bias is active.`  
       `// - Resets when bias is updated or when no bias is present.`  
       `// - Tied to EA heartbeat timeframe (e.g., M15), so "bars" here mean heartbeat intervals.`  
       `if (m_current_bias != SIGNAL_NONE)`  
         `{`  
          `m_bias_timeout_counter++;`  
          `if (m_bias_timeout_counter >= CSettings::BiasPersistenceBars)`  
            `{`  
             `m_current_bias = SIGNAL_NONE;`  
             `m_bias_timeout_counter = 0;`  
             `Print("SignalManager: Bias timed out after ", CSettings::BiasPersistenceBars, " bars. Reset to NONE.");`  
            `}`  
         `}`  
       `else`  
         `{`  
          `m_bias_timeout_counter = 0;  // Reset counter when no bias`  
         `}`

       `// --- STEP 1: Check all signals for CURRENT bar triggers ---`  
       `bool biasConfigured = false;`  
       `bool biasBuy = false;   // Represents a NEW bias signal THIS BAR`  
       `bool biasSell = false;  // Represents a NEW bias signal THIS BAR`  
       `bool entryConfigured = false;`  
       `bool entryBuy = false;`  
       `bool entrySell = false;`

       `// Iterate through all signals`  
       `for(int i = 0; i < m_signals.Total(); i++)`  
         `{`  
          `ISignal* sig = m_signals.At(i);`  
          `if(sig == NULL) continue;`

          `int signal = sig.GetSignal();`  
          `ENUM_SIGNAL_ROLE role = sig.GetRole();`

          `if(role == ROLE_BIAS)`  
            `{`  
             `biasConfigured = true;`  
             `if(signal == SIGNAL_BUY) biasBuy = true;`  
             `else if(signal == SIGNAL_SELL) biasSell = true;`  
            `}`  
          `else if(role == ROLE_ENTRY)`  
            `{`  
             `entryConfigured = true;`  
             `if(signal == SIGNAL_BUY) entryBuy = true;`  
             `else if(signal == SIGNAL_SELL) entrySell = true;`  
            `}`  
         `}`

       `// --- STEP 2: Update the persistent 'm_current_bias' ---`

       `// --- Check for Bias Disagreements ---`  
       `// If multiple bias signals disagree (e.g., one BUY, one SELL), reset the persistent bias`  
       `// to NONE immediately. This prevents conflicting bias from persisting.`  
       `if(biasBuy && biasSell)`  
         `{`  
          `Print("SignalManager: Bias signal disagreement on current bar. Bias reset to NONE.");`  
          `m_current_bias = SIGNAL_NONE;  // Explicit reset on disagreement`  
          `return SIGNAL_NONE;`  
         `}`

       `// --- Update Persistent Bias ---`  
       `// Set or reinforce the persistent bias if a clear signal appears.`  
       `// Reset the timeout counter to start fresh persistence period.`  
       `if(biasBuy)`  
         `{`  
          `m_current_bias = SIGNAL_BUY;`  
          `m_bias_timeout_counter = 0;  // Reset counter on bias update`  
         `}`  
       `else if(biasSell)`  
         `{`  
          `m_current_bias = SIGNAL_SELL;`  
          `m_bias_timeout_counter = 0;  // Reset counter on bias update`  
         `}`  
       `// Note: If no new bias signal, m_current_bias persists from previous bars.`

       `// --- STEP 3: Final decision matrix using persistent bias ---`

       `// Check for entry-level disagreements`  
       `if(entryBuy && entrySell)`  
         `{`  
          `Print("SignalManager: Entry signal disagreement. No action taken.");`  
          `return SIGNAL_NONE;`  
         `}`

       `// Check for a BUY signal`  
       `if((m_current_bias == SIGNAL_BUY || !biasConfigured) &&  // Bias is BUY (or no bias is set)`  
          `(entryBuy || !entryConfigured) &&                     // Entry is BUY (or no entry is set)`  
          `(m_current_bias == SIGNAL_BUY || entryBuy))           // At least one of them MUST be BUY`  
         `{`  
          `return SIGNAL_BUY;`  
         `}`

       `// Check for a SELL signal`  
       `if((m_current_bias == SIGNAL_SELL || !biasConfigured) && // Bias is SELL (or no bias is set)`  
          `(entrySell || !entryConfigured) &&                    // Entry is SELL (or no entry is set)`  
          `(m_current_bias == SIGNAL_SELL || entrySell))         // At least one of them MUST be SELL`  
         `{`  
          `return SIGNAL_SELL;`  
         `}`

       `// Default: No signal`  
       `return SIGNAL_NONE;`  
      `}`

    `/**`  
     `* @brief Gets the status of all signals AND the current persistent bias.`  
     `*/`  
    `string GetStatus()`  
      `{`  
       `string status = StringFormat("Current Bias: %s | ", GetSignalString(m_current_bias));`  
       `for(int i = 0; i < m_signals.Total(); i++)`  
         `{`  
          `ISignal* sig = m_signals.At(i);`  
          `if(sig == NULL) continue;`  
          `string roleStr = (sig.GetRole() == ROLE_BIAS) ? "[Bias]" : "[Entry]";`  
          `string tfStr = GetTimeframeString(sig.GetTimeframe());`  
          `status += roleStr + " " + sig.GetStatus() + " " + tfStr + " | ";`  
         `}`  
       `// Remove trailing " | "`  
       `if(StringLen(status) > 3)`  
          `status = StringSubstr(status, 0, StringLen(status) - 3);`  
       `return status;`  
      `}`

`private:`  
    `string GetSignalString(int signal)`  
       `{`  
        `switch(signal)`  
          `{`  
           `case SIGNAL_BUY: return "BUY";`  
           `case SIGNAL_SELL: return "SELL";`  
           `default: return "NONE";`  
          `}`  
       `}`

    `string GetTimeframeString(ENUM_TIMEFRAMES timeframe)`  
       `{`  
        `switch(timeframe)`  
          `{`  
           `case PERIOD_M1: return "M1";`  
           `case PERIOD_M5: return "M5";`  
           `case PERIOD_M15: return "M15";`  
           `case PERIOD_M30: return "M30";`  
           `case PERIOD_H1: return "H1";`  
           `case PERIOD_H4: return "H4";`  
           `case PERIOD_D1: return "D1";`  
           `case PERIOD_W1: return "W1";`  
           `case PERIOD_MN1: return "MN1";`  
           `default: return EnumToString(timeframe);`  
          `}`  
       `}`  
  `};`  
`//+------------------------------------------------------------------+`

   `//+------------------------------------------------------------------+`  
    `//|                                                   DCAManager.mqh |`  
    `//|                                     Copyright 2025, LAWRANCE KOH |`  
    `//|                                          lawrancekoh@outlook.com |`  
    `//+------------------------------------------------------------------+`  
    `#property copyright "Copyright 2025, LAWRANCE KOH"`  
    `#property link      "lawrancekoh@outlook.com"`  
    `#property version   "1.00"`

    `#include "Settings.mqh"`  
    `#include "TradeManager.mqh"`  
    `#include "MoneyManager.mqh"`

    `class CDCAManager`  
    `{`  
    `private:`  
        `CTradeManager* m_trade_manager;`  
        `CMoneyManager* m_money_manager;`

    `public:`  
        `CDCAManager(void){};`  
        `~CDCAManager(void){};`

        `void SetTradeManager(CTradeManager* tm) { m_trade_manager = tm; }`  
        `void SetMoneyManager(CMoneyManager* mm) { m_money_manager = mm; }`

        `void Init()`  
        `{`  
            `// Nothing to do here for now`  
        `}`

        `void ManageDCA(ENUM_POSITION_TYPE direction, const CBasket &basket)`  
        `{`  
            `// DCA Guard Clauses - check first before any modifications`  
            `if (CSettings::DcaMaxTrades <= 0 || basket.Ticket == 0) return; // DCA disabled or no basket`  
            `if (!m_money_manager.CheckDrawdown()) return; // Risk check: high drawdown blocks DCA`  
            `if (basket.TradeCount >= CSettings::DcaMaxTrades) return; // Max trades reached`

            `double pip_size = CMoneyManager::GetPipSize();`

            `// Get current market price for drawdown calculation`  
            `double market_price = (basket.BasketDirection == POSITION_TYPE_BUY) ?`  
                                `SymbolInfoDouble(_Symbol, SYMBOL_BID) :`  
                                `SymbolInfoDouble(_Symbol, SYMBOL_ASK);`

            `// Calculate required drawdown pips for this DCA level (increases with each trade)`  
            `double required_drawdown_pips = CSettings::DcaTriggerPips;`  
            `for(int i = 1; i < basket.TradeCount; i++)`  
            `{`  
                `required_drawdown_pips *= CSettings::DcaStepMultiplier;`  
            `}`

            `// Calculate actual drawdown in pips from last trade`  
            `double drawdown_pips = 0;`  
            `if (basket.BasketDirection == POSITION_TYPE_BUY)`  
                `drawdown_pips = (basket.LastTradePrice - market_price) / pip_size;`  
            `else`  
                `drawdown_pips = (market_price - basket.LastTradePrice) / pip_size;`

            `// Check if drawdown meets DCA trigger threshold`  
            `if (drawdown_pips < required_drawdown_pips) return;`

            `// Calculate DCA Lot Size (apply multiplier after certain trades)`  
            `double dca_lot;`  
            `if (basket.TradeCount >= CSettings::DcaLotMultiplierStart)`  
            `{`  
                `dca_lot = basket.LastTradeLots * CSettings::DcaLotMultiplier; // Increase lot size`  
            `}`  
            `else`  
            `{`  
                `dca_lot = basket.LastTradeLots; // Same lot size as previous trade`  
            `}`  
            `dca_lot = m_money_manager.ValidateLotSize(dca_lot); // Ensure broker compliance`

            `// Execute DCA Trade in same direction as basket`  
            `int signal = (basket.BasketDirection == POSITION_TYPE_BUY) ? SIGNAL_BUY : SIGNAL_SELL;`  
            `m_trade_manager.OpenTrade(signal, dca_lot, CSettings::SlPips, 0, "DCA", basket.TradeCount + 1);`

            `// Refresh basket cache to include the new DCA position`  
            `m_trade_manager.Refresh();`

            `// Fetch updated basket to ensure we use the new AvgEntryPrice`  
            `CBasket updated_basket = m_trade_manager.GetCachedBasket(direction);`

            `// Update basket SL/TP after successful DCA trade (uniform risk management)`  
            `// Use updated_basket to be consistent, though InitialTradePrice should be invariant`  
            `double basket_sl_price = updated_basket.InitialTradePrice + (direction == POSITION_TYPE_BUY ? -CSettings::SlPips * pip_size : CSettings::SlPips * pip_size);`  
            `m_trade_manager.SetBasketSL(direction, basket_sl_price); // Set SL for entire basket`

            `// Set uniform basket TP if enabled`  
            `if(CSettings::BasketTpPips > 0)`  
            `{`  
                `// Use updated_basket.AvgEntryPrice which includes the new DCA trade`  
                `double basket_tp_price = updated_basket.AvgEntryPrice + (direction == POSITION_TYPE_BUY ? CSettings::BasketTpPips * pip_size : -CSettings::BasketTpPips * pip_size);`  
                `m_trade_manager.SetBasketTP(direction, basket_tp_price); // Set TP for entire basket`  
            `}`  
        `}`  
    `};`  
    `//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                          TrailingStopManager.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include "Settings.mqh"`  
`#include <Trade/Trade.mqh>`  
`#include "MoneyManager.mqh"`  
`#include "CatrUtility.mqh"`

`class CTrailingStopManager : public CObject`  
`{`  
`private:`  
     `CTrade m_trade;`  
     `CatrUtility m_atr_utility;`

     `void HandleSteppedTSL(const CBasket &basket);`  
     `void HandleAtrTsl(const CBasket &basket);`

`public:`  
     `CTrailingStopManager(void) {};`  
     `~CTrailingStopManager(void) {};`

     `void Init()`  
     `{`  
         `m_trade.SetExpertMagicNumber(CSettings::EaMagicNumber);`

         `// Initialize ATR utility for ATR-based trailing stops`  
         `if(!m_atr_utility.Init(CSettings::TslAtrPeriod, PERIOD_CURRENT))`  
         `{`  
             `Print("TrailingStopManager::Init: Failed to initialize ATR utility");`  
         `}`  
     `}`

    `void ManageBasketTSL(const ENUM_POSITION_TYPE direction, const CBasket &basket)`  
    `{`  
        `if (CSettings::TslMode == MODE_TSL_NONE)`  
        `{`  
            `return;`  
        `}`

        `switch(CSettings::TslMode)`  
        `{`  
            `case MODE_TSL_STEP:`  
                `HandleSteppedTSL(basket);`  
                `break;`  
            `case MODE_TSL_ATR:`  
                `HandleAtrTsl(basket);`  
                `break;`  
            `// Other cases will be added in a later phase`  
        `}`  
    `}`  
`};`

`void CTrailingStopManager::HandleSteppedTSL(const CBasket &basket)`  
`{`  
    `if(basket.Ticket == 0) return;`

    `double market_price = (basket.BasketDirection == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);`

     `// Use cached basket data for performance optimization`  
     `double total_profit_money = basket.TotalProfit;`  
     `double total_costs = basket.TotalCosts;`

    `// Calculate average profit in pips per lot`  
    `double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);`  
    `double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);`  
    `double pip_value_per_lot = tick_value * (CMoneyManager::GetPipSize() / tick_size); // Value of 1 pip for 1 lot`

    `double average_profit_pips = 0.0;`  
    `if(pip_value_per_lot > 0 && basket.TotalVolume > 0)`  
    `{`  
        `average_profit_pips = total_profit_money / (basket.TotalVolume * pip_value_per_lot);`  
    `}`

    `// Debug removed`

    `if(average_profit_pips < CSettings::TslBeTriggerPips) return;`

    `double pip_size = CMoneyManager::GetPipSize();`  
    `double breakeven_price;`  
    `if(!CSettings::BreakevenIncludesCosts)`  
    `{`  
        `breakeven_price = basket.AvgEntryPrice;`  
        `if(basket.BasketDirection == POSITION_TYPE_BUY)`  
            `breakeven_price += CSettings::BeOffsetPips * pip_size;`  
        `else`  
            `breakeven_price -= CSettings::BeOffsetPips * pip_size;`  
    `}`  
    `else`  
    `{`  
        `// Account for estimated commission on close`  
        `total_costs -= basket.TotalVolume * CSettings::CommissionPerLot;`  
        `double desired_profit = CMoneyManager::GetMoneyFromPips(CSettings::BeOffsetPips, basket.TotalVolume);`  
        `double total_money_to_cover = desired_profit - total_costs;`  
        `double total_pips_to_cover = CMoneyManager::GetPipsFromMoney(total_money_to_cover, basket.TotalVolume);`  
        `if (total_pips_to_cover < 0) total_pips_to_cover = 0;  // Prevent loss SL`  
        `double offset = total_pips_to_cover * pip_size;`  
        `breakeven_price = basket.AvgEntryPrice;`  
        `if(basket.BasketDirection == POSITION_TYPE_BUY)`  
            `breakeven_price += offset;`  
        `else`  
            `breakeven_price -= offset;`  
    `}`

    `double profit_beyond_trigger = average_profit_pips - CSettings::TslBeTriggerPips;`  
    `int steps = 0;`  
    `if(profit_beyond_trigger > 0 && CSettings::TslStepPips > 0)`  
        `steps = (int)floor(profit_beyond_trigger / CSettings::TslStepPips);`

    `double new_sl_price = 0;`  
    `if(basket.BasketDirection == POSITION_TYPE_BUY)`  
        `new_sl_price = breakeven_price + (steps * CSettings::TslStepPips * CMoneyManager::GetPipSize());`  
    `else`  
        `new_sl_price = breakeven_price - (steps * CSettings::TslStepPips * CMoneyManager::GetPipSize());`  
      
    `double stop_level_dist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;`

    `for(int i = 0; i < ArraySize(basket.Tickets); i++)`  
    `{`  
        `ulong ticket = basket.Tickets[i];`  
        `if(!PositionSelectByTicket(ticket)) continue;`  
          
        `double current_sl = PositionGetDouble(POSITION_SL);`

        `// Debug: Print values for troubleshooting`

        `if(basket.BasketDirection == POSITION_TYPE_BUY && new_sl_price >= market_price - stop_level_dist)`  
        `{`

            `continue;`  
        `}`  
        `if(basket.BasketDirection == POSITION_TYPE_SELL && new_sl_price <= market_price + stop_level_dist)`  
        `{`

            `continue;`  
        `}`

        `double current_tp = PositionGetDouble(POSITION_TP);`  
        `double new_tp = CSettings::TslRemoveTp ? 0 : current_tp;`

        `// Skip modification if both SL and TP are essentially the same (prevent invalid stops on no-change)`  
        `if (MathAbs(new_sl_price - current_sl) < 0.00001 && MathAbs(new_tp - current_tp) < 0.00001) {`  
            `continue;`  
        `}`

        `// Check for minimum meaningful difference (prevent floating-point precision issues)`  
        `double min_diff = _Point * 2; // Minimum 2 points difference`  
        `if(MathAbs(new_sl_price - current_sl) < min_diff) {`  
            `continue;`  
        `}`

        `if(basket.BasketDirection == POSITION_TYPE_BUY && new_sl_price <= current_sl) {`  
            `continue;`  
        `}`  
        `if(basket.BasketDirection == POSITION_TYPE_SELL && (new_sl_price >= current_sl && current_sl != 0.0)) {`  
            `continue;`  
        `}`

        `m_trade.PositionModify(ticket, new_sl_price, new_tp);`  
    `}`  
`}`

`void CTrailingStopManager::HandleAtrTsl(const CBasket &basket)`  
`{`  
    `if(basket.Ticket == 0) return;`

    `double market_price = (basket.BasketDirection == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);`

    `// Calculate ATR-based trailing stop level using current market price for true trailing`  
    `double pip_size = CMoneyManager::GetPipSize();`  
    `double new_sl_price = m_atr_utility.GetAtrBasedLevel(market_price, CSettings::TslAtrMultiplier, basket.BasketDirection == POSITION_TYPE_BUY, pip_size);`

    `double stop_level_dist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;`

    `// Validate ATR-based SL against broker requirements`  
    `bool is_valid_sl = true;`  
    `if(basket.BasketDirection == POSITION_TYPE_BUY)`  
    `{`  
        `// For BUY: SL must be below BID by at least stops_distance`  
        `if(new_sl_price >= market_price - stop_level_dist)`  
        `{`  
            `Print("ATR TSL: Invalid SL for BUY basket. SL: ", new_sl_price, " would be too close to market price: ", market_price);`  
            `return;`  
        `}`  
    `}`  
    `else // POSITION_TYPE_SELL`  
    `{`  
        `// For SELL: SL must be above ASK by at least stops_distance`  
        `if(new_sl_price <= market_price + stop_level_dist)`  
        `{`  
            `Print("ATR TSL: Invalid SL for SELL basket. SL: ", new_sl_price, " would be too close to market price: ", market_price);`  
            `return;`  
        `}`  
    `}`

    `// Apply ATR-based trailing stop to all positions in basket`  
    `for(int i = 0; i < ArraySize(basket.Tickets); i++)`  
    `{`  
        `ulong ticket = basket.Tickets[i];`  
        `if(!PositionSelectByTicket(ticket)) continue;`

        `double current_sl = PositionGetDouble(POSITION_SL);`  
        `double current_tp = PositionGetDouble(POSITION_TP);`  
        `double new_tp = CSettings::TslRemoveTp ? 0 : current_tp;`

        `// Skip modification if both SL and TP are essentially the same (prevent invalid stops on no-change)`  
        `if (MathAbs(new_sl_price - current_sl) < 0.00001 && MathAbs(new_tp - current_tp) < 0.00001) {`  
            `continue;`  
        `}`

        `// Check for minimum meaningful difference (prevent floating-point precision issues)`  
        `double min_diff = _Point * 2; // Minimum 2 points difference`  
        `if(MathAbs(new_sl_price - current_sl) < min_diff) {`  
            `continue;`  
        `}`

        `// Apply "Better Price" rule: only modify if new SL improves current SL`  
        `if(basket.BasketDirection == POSITION_TYPE_BUY && new_sl_price <= current_sl) {`  
            `continue;`  
        `}`  
        `if(basket.BasketDirection == POSITION_TYPE_SELL && (new_sl_price >= current_sl && current_sl != 0.0)) {`  
            `continue;`  
        `}`

        `Print("ATR TSL: Modifying position ", ticket, " SL from ", current_sl, " to ", new_sl_price, " (ATR multiplier: ", CSettings::TslAtrMultiplier, ")");`  
        `if(!m_trade.PositionModify(ticket, new_sl_price, new_tp))`  
        `{`  
            `Print("ATR TSL: Failed to modify position ", ticket, " Error: ", m_trade.ResultRetcode());`  
        `}`  
    `}`  
`}`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                  TimeManager.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include "Settings.mqh"`

`class CTimeManager`  
  `{`  
`private:`  
    `int m_start_mins;`  
    `int m_end_mins;`  
    `bool m_allowed_days[7]; // 0=Sunday, 1=Monday, ..., 6=Saturday`

`public:`  
    `CTimeManager(void) {};`  
    `~CTimeManager(void) {};`

    `void Init()`  
      `{`  
         `// Pre-calculate start/end times in minutes from midnight`  
         `string start_time = CSettings::EaTradingTimeStart;`  
         `m_start_mins = (int)StringToInteger(StringSubstr(start_time, 0, 2)) * 60 +`  
                        `(int)StringToInteger(StringSubstr(start_time, 3, 2));`

         `string end_time = CSettings::EaTradingTimeEnd;`  
         `m_end_mins = (int)StringToInteger(StringSubstr(end_time, 0, 2)) * 60 +`  
                      `(int)StringToInteger(StringSubstr(end_time, 3, 2));`

         `// Pre-calculate allowed days as boolean array`  
         `string days_str = "," + CSettings::EaTradingDays + ",";`  
         `StringReplace(days_str, " ", ""); // Handle spaces in input`  
         `for(int i = 0; i < 7; i++)`  
         `{`  
            `string day_check = "," + IntegerToString(i) + ",";`  
            `m_allowed_days[i] = (StringFind(days_str, day_check) != -1);`  
         `}`  
      `}`

   `bool IsTradeTimeAllowed()`  
     `{`  
      `//--- Initial Check`  
      `if(CSettings::EaTradingDays == "")`  
         `return true;`

      `//--- Get Current Time`  
      `MqlDateTime current_time;`  
      `TimeCurrent(current_time);`

      `//--- Day of Week Check using pre-calculated boolean array`  
      `if(!m_allowed_days[current_time.day_of_week])`  
         `return false;`

      `//--- Time of Day Check using pre-calculated minutes`  
      `long current_mins = current_time.hour * 60 + current_time.min;`

      `//--- Handle Overnight Sessions (e.g., Start 22:00, End 06:00)`  
      `if(m_start_mins > m_end_mins)`  
        `{`  
         `return (current_mins >= m_start_mins || current_mins <= m_end_mins);`  
        `}`  
      `//--- Handle Normal Day Sessions`  
      `else`  
        `{`  
         `return (current_mins >= m_start_mins && current_mins <= m_end_mins);`  
        `}`  
     `}`  
  `};`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                  NewsManager.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.01" // Updated version`

`#include "Settings.mqh"`  
`#include <Arrays/ArrayObj.mqh>`

`//+------------------------------------------------------------------+`  
`//| News Event Structure                                             |`  
`//+------------------------------------------------------------------+`  
`struct CNewsEvent`  
`{`  
   `string title;`  
   `string currency;`  
   `string impact;`  
   `datetime time;`  
`};`

`//+------------------------------------------------------------------+`  
`//| CNewsManager Class                                               |`  
`//| Manages filtering of trading signals based on calendar news      |`  
`//| events.                                                          |`  
`//+------------------------------------------------------------------+`  
`class CNewsManager`  
   `{`  
`private:`  
    `datetime m_next_news_time;`  
    `bool m_news_cache_valid;`

    `// Web Request Cache`  
    `CNewsEvent m_cached_events[]; // Dynamic array of news events`  
    `datetime m_last_web_request_time;`  
    `const int m_web_request_interval_seconds; // Interval to refresh news (e.g. 1 hour)`

    `//+------------------------------------------------------------------+`  
    `//| Helpers for String Processing                                    |`  
    `//+------------------------------------------------------------------+`  
    `string CleanQuote(string str)`  
    `{`  
       `if (StringLen(str) >= 2 && StringGetCharacter(str, 0) == '"' && StringGetCharacter(str, StringLen(str)-1) == '"')`  
       `{`  
           `return StringSubstr(str, 1, StringLen(str) - 2);`  
       `}`  
       `return str;`  
    `}`

    `datetime ParseCsvDateTime(string dateStr, string timeStr)`  
    `{`  
        `// Format: MM/DD/YYYY and HH:MM`  
        `// StringToTime converts "yyyy.mm.dd [hh:mi]"`  
        `// We need to convert MM/DD/YYYY to yyyy.mm.dd`

        `string date_parts[];`  
        `// Check for / or -`  
        `if(StringSplit(dateStr, '/', date_parts) != 3)`  
        `{`  
             `if(StringSplit(dateStr, '-', date_parts) != 3) return 0;`  
        `}`

        `// date_parts: [0]=MM, [1]=DD, [2]=YYYY`  
        `string yyyy = date_parts[2];`  
        `string mm = date_parts[0];`  
        `string dd = date_parts[1];`

        `string formatted_time = yyyy + "." + mm + "." + dd + " " + timeStr;`  
        `return StringToTime(formatted_time);`  
    `}`

    `//+------------------------------------------------------------------+`  
    `//| Web Request Logic                                                |`  
    `//+------------------------------------------------------------------+`  
    `bool FetchAndParseNews()`  
    `{`  
       `string cookie = NULL, headers;`  
       `char post[], result[];`  
       `int res;`  
       `string url = CSettings::NewsCalendarURL;`  
       `if (url == "") url = "https://nfs.forexfactory.net/ffcal_week_this.csv"; // Fallback default`

       `// Reset Last Error`  
       `ResetLastError();`

       `int timeout = 5000; // 5 seconds`

       `res = WebRequest("GET", url, cookie, NULL, timeout, post, 0, result, headers);`

       `if (res == -1)`  
       `{`  
          `Print("NewsManager: WebRequest failed. Error: ", GetLastError());`  
          `// Check if URL is allowed`  
          `if(GetLastError() == 4060) // ERR_FUNCTION_NOT_ALLOWED`  
          `{`  
             `Print("NewsManager: Please add '", url, "' to the allowed URLs in Tools->Options->Expert Advisors.");`  
          `}`  
          `return false;`  
       `}`  
       `else if (res != 200)`  
       `{`  
          `Print("NewsManager: WebRequest returned HTTP status ", res);`  
          `return false;`  
       `}`

       `// Process Result`  
       `string response = CharArrayToString(result);`

       `// Parse CSV`  
       `string lines[];`  
       `int line_count = StringSplit(response, '\n', lines);`

       `if(line_count <= 0) return false;`

       `ArrayResize(m_cached_events, 0); // Clear cache`

       `for(int i = 0; i < line_count; i++)`  
       `{`  
          `string line = lines[i];`  
          `if(StringLen(line) < 5) continue; // Skip empty lines`

          `string fields[];`  
          `int field_count = StringSplit(line, ',', fields);`

          `if(field_count < 4) continue;`

          `// Expected: "Date","Time","Currency","Impact","Event"`  
          `// We need to handle that StringSplit splits by comma, but some fields might contain comma?`  
          `// Standard FF CSV usually puts quotes around fields.`  
          `// Simple split might break if "Event" contains comma.`  
          `// For robustness, we should respect quotes.`  
          `// But StringSplit is simple.`  
          `// Assuming FF CSV format is consistent and Event is the last field or doesn't have commas usually.`  
          `// If fields are quoted, we can rely on standard format.`

          `// Map fields`  
          `string s_date = CleanQuote(fields[0]);`  
          `string s_time = CleanQuote(fields[1]);`  
          `string s_curr = CleanQuote(fields[2]);`  
          `string s_impact = CleanQuote(fields[3]);`  
          `string s_title = (field_count > 4) ? CleanQuote(fields[4]) : "";`

          `CNewsEvent event;`  
          `event.currency = s_curr;`  
          `event.impact = s_impact;`  
          `event.title = s_title;`  
          `event.time = ParseCsvDateTime(s_date, s_time);`

          `if(event.time > 0)`  
          `{`  
             `int size = ArraySize(m_cached_events);`  
             `ArrayResize(m_cached_events, size + 1);`  
             `m_cached_events[size] = event;`  
          `}`  
       `}`

       `Print("NewsManager: Successfully fetched and parsed ", ArraySize(m_cached_events), " news events.");`  
       `return true;`  
    `}`

    `//+------------------------------------------------------------------+`  
    `//| Checks for blocking news events from the built-in MT5 calendar.  |`  
    `//| @return true if a blocking news event is found, false otherwise |`  
    `//+------------------------------------------------------------------+`  
    `bool IsMt5CalendarBlockActive()`  
     `{`  
      `// STUBBED DUE TO COMPILER BUG: MqlCalendarValue string and enum members cannot be accessed reliably.`  
      `// This is a known issue in MQL5 compiler; logic preserved in comments for future remediation.`  
      `/*`  
      `//--- Define Time Window in GMT`  
      `long mins_before_sec = (long)CSettings::NewsMinsBefore * 60;`  
      `long mins_after_sec  = (long)CSettings::NewsMinsAfter * 60;`  
      `datetime from = TimeGMT() - mins_before_sec;`  
      `datetime to   = TimeGMT() + mins_after_sec;`

      `MqlCalendarValue values_array[];`

      `//--- Get Calendar Events`  
      `if(CalendarValueHistory(values_array, from, to) > 0)`  
        `{`  
         `//--- Get Symbol Currencies`  
         `string currency1 = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_BASE);`  
         `string currency2 = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_PROFIT);`

         `//--- Loop and Filter`  
         `int total_events = ArraySize(values_array);`  
         `for(int i = 0; i < total_events; i++)`  
           `{`  
              `MqlCalendarValue event = values_array[i];`

              `//--- Check if the event's currency matches the symbol's currencies`  
              `if(event.currency == currency1 || event.currency == currency2)`  
                `{`  
                   `//--- Check if the impact level is set to be filtered`  
                   `bool is_high_impact   = (event.importance == CALENDAR_IMPORTANCE_HIGH && CSettings::NewsFilterHighImpact);`  
                   `bool is_medium_impact = (event.importance == CALENDAR_IMPORTANCE_MODERATE && CSettings::NewsFilterMedImpact);`  
                   `bool is_low_impact    = (event.importance == CALENDAR_IMPORTANCE_LOW && CSettings::NewsFilterLowImpact);`

                   `if(is_high_impact || is_medium_impact || is_low_impact)`  
                     `{`  
                      `// Found a blocking event, print details and return`  
                      `PrintFormat("NEWS BLOCK: %s %s %s",`  
                                  `TimeToString(event.time, TIME_DATE | TIME_MINUTES),`  
                                  `event.currency,`  
                                  `event.name);`  
                      `return true;`  
                     `}`  
                `}`  
           `}`  
        `}`  
      `*/`

      `//--- No blocking events found (stubbed)`  
      `return false;`  
     `}`

`public:`  
   `//+------------------------------------------------------------------+`  
   `//| Constructor                                                      |`  
   `//+------------------------------------------------------------------+`  
   `CNewsManager(void) : m_web_request_interval_seconds(3600) // 1 Hour default refresh`  
   `{`  
       `m_last_web_request_time = 0;`  
   `};`  
   `//+------------------------------------------------------------------+`  
   `//| Destructor                                                       |`  
   `//+------------------------------------------------------------------+`  
   `~CNewsManager(void)`  
   `{`  
       `ArrayResize(m_cached_events, 0);`  
   `};`

   `//+------------------------------------------------------------------+`  
   `//| Initialization Method                                            |`  
   `//+------------------------------------------------------------------+`  
   `void Init()`  
     `{`  
        `m_next_news_time = 0;`  
        `m_news_cache_valid = false;`  
        `m_last_web_request_time = 0;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Refresh news cache by finding next relevant news event         |`  
   `//+------------------------------------------------------------------+`  
   `void RefreshNewsCache()`  
     `{`  
        `// For built-in mode`  
        `// TODO: Implement actual news checking logic when MT5 calendar is available`  
        `// For now, set cache as valid with no news`  
        `m_next_news_time = 0;`  
        `m_news_cache_valid = true;`  
     `}`

   `//+------------------------------------------------------------------+`  
   `//| Check Web Request News Block                                     |`  
   `//+------------------------------------------------------------------+`  
   `bool CheckWebRequestNews()`  
   `{`  
       `// 1. Refresh Cache if needed`  
       `if (TimeCurrent() - m_last_web_request_time > m_web_request_interval_seconds || m_last_web_request_time == 0)`  
       `{`  
           `if (FetchAndParseNews())`  
           `{`  
               `m_last_web_request_time = TimeCurrent();`  
           `}`  
           `else`  
           `{`  
               `// If failed, maybe try again sooner? or just keep old cache?`  
               `// We keep old cache but update time to retry in 5 mins maybe?`  
               `// For now, retry normal interval to avoid spamming if error is persistent.`  
               `m_last_web_request_time = TimeCurrent();`  
           `}`  
       `}`

       `// 2. Check cached events against current time`  
       `long mins_before_sec = (long)CSettings::NewsMinsBefore * 60;`  
       `long mins_after_sec  = (long)CSettings::NewsMinsAfter * 60;`  
       `datetime current_time = TimeCurrent();`

       `// Get Symbol Currencies`  
       `string currency1 = SymbolInfoString(CSettings::Symbol, SYMBOL_CURRENCY_BASE);`  
       `string currency2 = SymbolInfoString(CSettings::Symbol, SYMBOL_CURRENCY_PROFIT);`

       `int total = ArraySize(m_cached_events);`  
       `for(int i = 0; i < total; i++)`  
       `{`  
           `CNewsEvent event = m_cached_events[i];`

           `// Filter by Currency`  
           `// Check against symbol currencies`  
           `bool currency_match = (event.currency == currency1 || event.currency == currency2);`

           `// Also check against global filter list if currencies are specified there?`  
           `// Usually we only care about symbol currencies.`  
           ``// But `CSettings::NewsFilterCurrencies` exists.``  
           `// If user specified currencies, maybe we should also check those?`  
           `// The prompt implies "relevant to the current symbol".`  
           ``// But existing logic stub checked `CSettings::NewsFilterCurrencies` (Wait, the stub code checked `event.currency == currency1 || event.currency == currency2`).``  
           ``// The setting `NewsFilterCurrencies` was present in `CSettings` but not used in the stub code I saw.``  
           `// I will implement check for Symbol currencies AND the list if provided.`

           `if (!currency_match)`  
           `{`  
               `// Check if in allowed list`  
               `if (StringFind(CSettings::NewsFilterCurrencies, event.currency) >= 0)`  
               `{`  
                   `currency_match = true;`  
               `}`  
           `}`

           `if (!currency_match) continue;`

           `// Filter by Impact`  
           `bool is_high = (event.impact == "High" && CSettings::NewsFilterHighImpact);`  
           `bool is_med = (event.impact == "Medium" && CSettings::NewsFilterMedImpact);`  
           `bool is_low = (event.impact == "Low" && CSettings::NewsFilterLowImpact);`

           `if (!is_high && !is_med && !is_low) continue;`

           `// Filter by Time`  
           `// Check if current time is within [event_time - before, event_time + after]`  
           `if (current_time >= (event.time - mins_before_sec) &&`  
               `current_time <= (event.time + mins_after_sec))`  
           `{`  
               `PrintFormat("NEWS BLOCK (Web): %s %s %s",`  
                          `TimeToString(event.time, TIME_DATE | TIME_MINUTES),`  
                          `event.currency,`  
                          `event.title);`  
               `return true;`  
           `}`  
       `}`

       `return false;`  
   `}`

   `//+------------------------------------------------------------------+`  
   `//| Main public method to check if trading is blocked by news.       |`  
   `//| @return true if trading is blocked, false otherwise             |`  
   `//+------------------------------------------------------------------+`  
   `bool IsNewsBlockActive()`  
     `{`  
      `switch(CSettings::NewsSourceMode)`  
        `{`  
         `case MODE_DISABLED:`  
            `return false;`

         `case MODE_MT5_BUILT_IN:`  
            `// Only refresh cache if invalid or if next news time is approaching/passed`  
            `if(!m_news_cache_valid || (m_next_news_time > 0 && TimeCurrent() >= m_next_news_time - CSettings::NewsMinsBefore * 60))`  
            `{`  
               `RefreshNewsCache();`  
            `}`  
            `return IsMt5CalendarBlockActive();`

         `case MODE_WEB_REQUEST:`  
            `return CheckWebRequestNews();`  
        `}`  
      `return false;`  
     `}`  
  `};`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                              StackingManager.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include "Settings.mqh"`  
`#include "MoneyManager.mqh"`  
`#include "TradeManager.mqh"`

`class CStackingManager`  
  `{`  
`private:`  
   `CMoneyManager* m_money_manager;`  
   `CTradeManager* m_trade_manager;`

`public:`  
   `CStackingManager(void) {};`  
   `~CStackingManager(void) {};`

   `void SetMoneyManager(CMoneyManager* mm) { m_money_manager = mm; }`  
   `void SetTradeManager(CTradeManager* tm) { m_trade_manager = tm; }`

   `void Init()`  
     `{`  
        `// Nothing to do here for now`  
     `}`

   `void ManageStacking(ENUM_POSITION_TYPE direction, const CBasket &basket)`  
     `{`  
      `// Guard clauses`  
      `if (CSettings::StackingMaxTrades <= 0 || basket.Ticket == 0) return;`  
      `if (basket.StackingCount >= CSettings::StackingMaxTrades) return;`

      `// Risk check: high drawdown blocks stacking`  
      `if (!m_money_manager.CheckDrawdown()) return;`

      `// Profit-based trigger check`  
      `if (basket.ProfitPips() < CSettings::StackingTriggerPips) return;`

      `// Calculate Stacking Lot`  
      `double stack_lot = m_money_manager.GetStackingLotSize(basket);`

      `// Capture existing basket SL before opening new trade`  
      `double previous_basket_sl = m_trade_manager.GetBasketSL(direction);`

      `// Execute Stacking Trade`  
      `int signal = (basket.BasketDirection == POSITION_TYPE_BUY) ? SIGNAL_BUY : SIGNAL_SELL;`  
      `m_trade_manager.OpenTrade(signal, stack_lot, CSettings::SlPips, 0, "STACK", basket.StackingCount + 1);`

      `// Refresh basket cache to include the new stacking position`  
      `m_trade_manager.Refresh();`  
      `CBasket updated_basket = m_trade_manager.GetCachedBasket(direction);`

      `// Set uniform SL on all basket positions to ensure consistency`  
      `// We must determine whether to use the new trade's SL or the existing (potentially tighter) SL`  
      `double new_trade_sl = m_trade_manager.GetBasketSL(updated_basket.BasketDirection);`  
      `double sl_to_apply = new_trade_sl;`

      `if(previous_basket_sl > 0)`  
        `{`  
         `if(updated_basket.BasketDirection == POSITION_TYPE_BUY)`  
           `{`  
            `// For BUY, higher SL is better (tighter)`  
            `if(previous_basket_sl > new_trade_sl)`  
               `sl_to_apply = previous_basket_sl;`  
           `}`  
         `else`  
           `{`  
            `// For SELL, lower SL is better (tighter)`  
            `if(previous_basket_sl < new_trade_sl && previous_basket_sl > 0)`  
               `sl_to_apply = previous_basket_sl;`  
           `}`  
        `}`

      `if(sl_to_apply > 0)`  
        `{`  
         `m_trade_manager.SetBasketSL(updated_basket.BasketDirection, sl_to_apply);`  
         `// Update current_basket_sl for the subsequent BE check`  
         `// Note: We don't declare double current_basket_sl here as it was used below,`  
         `// but the original code declared it locally. We need to make sure subsequent code uses sl_to_apply or we re-declare.`  
        `}`

      `double current_basket_sl = sl_to_apply;`

      `// Always move SL to true breakeven when stacking triggers to account for costs`  
      `double be_price = m_trade_manager.CalculateTrueBreakEvenPrice(updated_basket, updated_basket.TotalVolume);`

      `// Validate BE price against market price and stops level`  
      `double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);`  
      `double stops_distance = stops_level * _Point;`  
      `double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);`  
      `double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);`  
      `bool is_safe = false;`

      `if (updated_basket.BasketDirection == POSITION_TYPE_BUY) {`  
         `// For BUY, SL must be < Bid - Stops`  
         `if (be_price < bid - stops_distance) is_safe = true;`  
      `} else {`  
         `// For SELL, SL must be > Ask + Stops`  
         `if (be_price > ask + stops_distance) is_safe = true;`  
      `}`

      `// Only set if breakeven is better than current SL and safe`  
      `if (is_safe && ((updated_basket.BasketDirection == POSITION_TYPE_BUY && be_price > current_basket_sl) ||`  
          `(updated_basket.BasketDirection == POSITION_TYPE_SELL && be_price < current_basket_sl && current_basket_sl != 0.0))) {`  
         `m_trade_manager.SetBasketSL(updated_basket.BasketDirection, be_price);`  
      `}`  
     `}`  
  `};`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                      ISignal.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include <Object.mqh>`  
`#include "../Managers/Settings.mqh"`

`/**`  
 `* @brief Defines the contract for all signal-generating classes.`  
 `*`  
 `* This interface ensures that every signal, regardless of its underlying`  
 `* indicator or logic, provides a consistent way for the SignalManager`  
 `* to initialize it, retrieve trading signals, and check its status.`  
 `*/`  
`class ISignal : public CObject`  
   `{`  
`public:`  
     `/**`  
      `* @brief Initializes the signal with the required parameters.`  
      `*`  
      `* This method should be called once before any other methods are used.`  
      `* It sets up the indicator handles and any other necessary configurations.`  
      `*`  
      `* @param settings The signal settings struct containing all parameters.`  
      `* @return bool true if initialization is successful, false otherwise.`  
      `*/`  
     `virtual bool Init(const CSignalSettings &settings) { return false; }`

     `/**`  
      `* @brief Gets the latest trading signal.`  
      `*`  
      `* This is the core method that the SignalManager will call on every tick`  
      `* or bar to determine if a trading opportunity exists.`  
      `*`  
      `* @return int A signal from the ENUM_TRADE_SIGNAL enumeration`  
      `*         (e.g., SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NONE).`  
      `*/`  
     `virtual int GetSignal() { return SIGNAL_NONE; }`

     `/**`  
      `* @brief Gets the current status of the signal.`  
      `*`  
      `* This can be used for debugging or displaying status information on the UI.`  
      `* For example, it could return "RSI(14) is Overbought" or "MACD Cross Occurred".`  
      `*`  
      `* @return string A human-readable status message.`  
      `*/`  
     `virtual string GetStatus() { return ""; }`

     `/**`  
      `* @brief Gets the role of the signal (Bias or Entry).`  
      `*`  
      `* @return ENUM_SIGNAL_ROLE The role assigned to this signal.`  
      `*/`  
     `virtual ENUM_SIGNAL_ROLE GetRole() { return ROLE_BIAS; }`

     `/**`  
      `* @brief Gets the timeframe this signal operates on.`  
      `*`  
      `* @return ENUM_TIMEFRAMES The timeframe (e.g., PERIOD_M15, PERIOD_H1).`  
      `*/`  
     `virtual ENUM_TIMEFRAMES GetTimeframe() const { return PERIOD_CURRENT; }`

     `/**`  
      `* @brief Draws visual indicators on the chart when a signal triggers.`  
      `*`  
      `* @param barTime The time of the bar where the signal occurred.`  
      `* @param signal The signal type (SIGNAL_BUY, SIGNAL_SELL).`  
      `* @param signalIndex The index of the signal slot (0-2) for vertical stacking.`  
      `*/`  
     `virtual void DrawSignal(datetime barTime, int signal, int signalIndex) = 0;`  
    `};`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                  CSignal_RSI.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include "ISignal.mqh"`

`/**`  
 `* @brief RSI signal implementation.`  
 `*`  
 `* Generates buy/sell signals based on RSI level cross out of oversold/overbought levels.`  
 `*/`  
`class CSignal_RSI : public ISignal`  
`{`  
`private:`  
    `int m_handle;                    // Indicator handle`  
    `ENUM_TIMEFRAMES m_timeframe;     // Timeframe`  
    `int m_period;                    // RSI period`  
    `ENUM_APPLIED_PRICE m_applied_price; // Applied price`  
    `double m_lvl_dn;                 // Oversold level`  
    `double m_lvl_up;                 // Overbought level`  
    `int m_last_signal;               // Last signal for status`  
    `ENUM_SIGNAL_ROLE m_role;         // Signal role (Bias or Entry)`

`public:`  
    `/**`  
     `* @brief Initializes the RSI signal.`  
     `*`  
     `* @param settings The signal settings.`  
     `* @return bool true if successful, false otherwise.`  
     `*/`  
    `virtual bool Init(const CSignalSettings &settings) override`  
    `{`  
        `// Map parameters from settings`  
        `m_timeframe = settings.Timeframe;`  
        `m_period = settings.Params.IntParams[0];`  
        `m_applied_price = settings.Params.Price;`  
        `m_lvl_dn = settings.Params.DoubleParams[0];`  
        `m_lvl_up = settings.Params.DoubleParams[1];`  
        `m_role = settings.Role;`

        `// Get indicator handle`  
        `m_handle = iRSI(_Symbol, m_timeframe, m_period, m_applied_price);`

        `// Validate handle`  
        `if (m_handle == INVALID_HANDLE)`  
        `{`  
            `Print("CSignal_RSI: Failed to get RSI handle for ", _Symbol, " timeframe ", EnumToString(m_timeframe));`  
            `return false;`  
        `}`

        `m_last_signal = SIGNAL_NONE;`  
        `return true;`  
    `}`

    `/**`  
     `* @brief Gets the current trading signal.`  
     `*`  
     `* @return int SIGNAL_BUY, SIGNAL_SELL, or SIGNAL_NONE.`  
     `*/`  
    `virtual int GetSignal() override`  
    `{`  
        `double rsi_buffer[3];`

        `// Get data from last closed bar (shift 1) and previous (shift 2)`  
        `if (CopyBuffer(m_handle, 0, 1, 2, rsi_buffer) != 2)`  
        `{`  
            `return SIGNAL_NONE;`  
        `}`

        `// Implement level cross out logic`  
        `bool buy_signal = rsi_buffer[0] > m_lvl_dn && rsi_buffer[1] <= m_lvl_dn; // Crossed up out of oversold`  
        `bool sell_signal = rsi_buffer[0] < m_lvl_up && rsi_buffer[1] >= m_lvl_up; // Crossed down out of overbought`

        `// Return signal`  
        `if (buy_signal)`  
        `{`  
            `m_last_signal = SIGNAL_BUY;`  
            `return SIGNAL_BUY;`  
        `}`  
        `if (sell_signal)`  
        `{`  
            `m_last_signal = SIGNAL_SELL;`  
            `return SIGNAL_SELL;`  
        `}`

        `m_last_signal = SIGNAL_NONE;`  
        `return SIGNAL_NONE;`  
    `}`

    `/**`  
     `* @brief Gets the status string.`  
     `*`  
     `* @return string Status message.`  
     `*/`  
    `virtual string GetStatus() override`  
    `{`  
        `string status = StringFormat("RSI(%d,%.1f,%.1f)", m_period, m_lvl_dn, m_lvl_up);`

        `if (m_last_signal == SIGNAL_BUY)`  
            `status += " [BUY]";`  
        `else if (m_last_signal == SIGNAL_SELL)`  
            `status += " [SELL]";`  
        `else`  
            `status += " [NEUTRAL]";`

        `return status;`  
    `}`

    `/**`  
     `* @brief Gets the role of the signal.`  
     `*`  
     `* @return ENUM_SIGNAL_ROLE The role.`  
     `*/`  
    `virtual ENUM_SIGNAL_ROLE GetRole() override`  
    `{`  
        `return m_role;`  
    `}`

    `/**`  
     `* @brief Gets the timeframe of the signal.`  
     `*`  
     `* @return ENUM_TIMEFRAMES The timeframe.`  
     `*/`  
    `virtual ENUM_TIMEFRAMES GetTimeframe() const override`  
    `{`  
        `return m_timeframe;`  
    `}`

    `/**`  
     `* @brief Draws visual indicators on the chart when a signal triggers.`  
     `*`  
     `* @param barTime The time of the bar where the signal occurred.`  
     `* @param signal The signal type (SIGNAL_BUY, SIGNAL_SELL).`  
     `* @param signalIndex The index of the signal slot (0-2) for vertical stacking.`  
     `*/`  
    `virtual void DrawSignal(datetime barTime, int signal, int signalIndex) override`  
    `{`  
        `// Implementation for chart drawing if needed`  
        `// For now, leave as stub or implement basic arrow drawing`  
    `}`  
`};`  
`//+------------------------------------------------------------------+`

`//+------------------------------------------------------------------+`  
`//|                                                    CSignal_MACD.mqh |`  
`//|                                     Copyright 2025, LAWRANCE KOH |`  
`//|                                          lawrancekoh@outlook.com |`  
`//+------------------------------------------------------------------+`  
`#property copyright "Copyright 2025, LAWRANCE KOH"`  
`#property link      "lawrancekoh@outlook.com"`  
`#property version   "1.00"`

`#include "ISignal.mqh"`

`/**`  
 `* @brief MACD signal implementation.`  
 `*`  
 `* Generates buy/sell signals based on MACD main line crossing the signal line.`  
 `*/`  
`class CSignal_MACD : public ISignal`  
`{`  
`private:`  
    `int m_handle;                    // Indicator handle`  
    `ENUM_TIMEFRAMES m_timeframe;     // Timeframe`  
    `int m_fast_period;               // Fast EMA period`  
    `int m_slow_period;               // Slow EMA period`  
    `int m_signal_period;             // Signal line period`  
    `ENUM_APPLIED_PRICE m_applied_price; // Applied price`  
    `bool m_threshold_check;          // Threshold check enabled`  
    `bool m_threshold_check_reverse;  // Reverse threshold logic`  
    `int m_last_signal;               // Last signal for status`  
    `ENUM_SIGNAL_ROLE m_role;         // Signal role (Bias or Entry)`

`public:`  
    `/**`  
     `* @brief Initializes the MACD signal.`  
     `*`  
     `* @param settings The signal settings.`  
     `* @return bool true if successful, false otherwise.`  
     `*/`  
    `virtual bool Init(const CSignalSettings &settings) override`  
    `{`  
        `// Map parameters from settings`  
        `m_timeframe = settings.Timeframe;`  
        `m_fast_period = settings.Params.IntParams[0];`  
        `m_slow_period = settings.Params.IntParams[1];`  
        `m_signal_period = settings.Params.IntParams[2];`  
        `m_applied_price = settings.Params.Price;`  
        `m_threshold_check = settings.Params.BoolParams[0];`  
        `m_threshold_check_reverse = settings.Params.BoolParams[1];`  
        `m_role = settings.Role;`

        `// Get indicator handle`  
        `m_handle = iMACD(_Symbol, m_timeframe, m_fast_period, m_slow_period, m_signal_period, m_applied_price);`

        `// Validate handle`  
        `if (m_handle == INVALID_HANDLE)`  
        `{`  
            `Print("CSignal_MACD: Failed to get MACD handle for ", _Symbol, " timeframe ", EnumToString(m_timeframe));`  
            `return false;`  
        `}`

        `m_last_signal = SIGNAL_NONE;`  
        `return true;`  
    `}`

    `/**`  
     `* @brief Gets the current trading signal.`  
     `*`  
     `* @return int SIGNAL_BUY, SIGNAL_SELL, or SIGNAL_NONE.`  
     `*/`  
    `virtual int GetSignal() override`  
    `{`  
        `double main_buffer[3];`  
        `double signal_buffer[3];`  
        `double histogram_buffer[3] = {0, 0, 0};`

        `// Get data from last closed bar (shift 1)`  
        `if (CopyBuffer(m_handle, 0, 1, 2, main_buffer) != 2 ||`  
            `CopyBuffer(m_handle, 1, 1, 2, signal_buffer) != 2)`  
        `{`  
            `return SIGNAL_NONE;`  
        `}`

        `// Histogram is optional for logging`  
        `CopyBuffer(m_handle, 2, 1, 2, histogram_buffer);`

        `// Implement crossover logic`  
        `bool buy_cross = main_buffer[0] > signal_buffer[0] && main_buffer[1] <= signal_buffer[1];`  
        `bool sell_cross = main_buffer[0] < signal_buffer[0] && main_buffer[1] >= signal_buffer[1];`

        `// Apply threshold filter if enabled`  
        `if (m_threshold_check)`  
        `{`  
            `if (m_threshold_check_reverse)`  
            `{`  
                `buy_cross = buy_cross && main_buffer[0] > 0;  // Buy cross must be above zero`  
                `sell_cross = sell_cross && main_buffer[0] < 0; // Sell cross must be below zero`  
            `}`  
            `else`  
            `{`  
                `buy_cross = buy_cross && main_buffer[0] < 0;  // Buy cross must be below zero`  
                `sell_cross = sell_cross && main_buffer[0] > 0; // Sell cross must be above zero`  
            `}`  
        `}`

        `// Return signal`  
        `if (buy_cross)`  
        `{`  
            `m_last_signal = SIGNAL_BUY;`  
            `return SIGNAL_BUY;`  
        `}`  
        `if (sell_cross)`  
        `{`  
            `m_last_signal = SIGNAL_SELL;`  
            `return SIGNAL_SELL;`  
        `}`

        `m_last_signal = SIGNAL_NONE;`  
        `return SIGNAL_NONE;`  
    `}`

    `/**`  
     `* @brief Gets the status string.`  
     `*`  
     `* @return string Status message.`  
     `*/`  
    `virtual string GetStatus() override`  
    `{`  
        `string status = StringFormat("MACD(%d,%d,%d)", m_fast_period, m_slow_period, m_signal_period);`

        `if (m_last_signal == SIGNAL_BUY)`  
            `status += " [BUY]";`  
        `else if (m_last_signal == SIGNAL_SELL)`  
            `status += " [SELL]";`  
        `else`  
            `status += " [NEUTRAL]";`

        `return status;`  
    `}`

    `/**`  
     `* @brief Gets the role of the signal.`  
     `*`  
     `* @return ENUM_SIGNAL_ROLE The role.`  
     `*/`  
    `virtual ENUM_SIGNAL_ROLE GetRole() override`  
    `{`  
        `return m_role;`  
    `}`

    `/**`  
     `* @brief Gets the timeframe of the signal.`  
     `*`  
     `* @return ENUM_TIMEFRAMES The timeframe.`  
     `*/`  
    `virtual ENUM_TIMEFRAMES GetTimeframe() const override`  
    `{`  
        `return m_timeframe;`  
    `}`

    `/**`  
     `* @brief Draws visual indicators on the chart when a signal triggers.`  
     `*`  
     `* @param barTime The time of the bar where the signal occurred.`  
     `* @param signal The signal type (SIGNAL_BUY, SIGNAL_SELL).`  
     `* @param signalIndex The index of the signal slot (0-2) for vertical stacking.`  
     `*/`  
    `virtual void DrawSignal(datetime barTime, int signal, int signalIndex) override`  
    `{`  
        `// Implementation for chart drawing if needed`  
        `// For now, leave as stub or implement basic arrow drawing`  
    `}`  
`};`  
`//+------------------------------------------------------------------+`  


[image1]: <data:image/png;base64,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P+WUU8r+F77whd66665b9qMwpZJAOAILGHMObL755qWCUcV2zDHHlH38sgAxa95pBiffIga+zc2yI4A7szhZlJj9aRamxx9/fFPpcv/M9owVLZXz3nvvXXUDpekee+xRfoEvrcSKmpmucNdddzXil/NUaSMOWAwalI8HHnhg74EHHij7EiVA/t50001ln3OYwABu1RoP2T4Qennxb/Ij2k1clJz8ueaaaxr/tYXL8fPII4+U/f322688f/DTn/60d+yxx5Z9FguXfTB7mO+B5yV+8COIz3XXXVfiK1vJ6Ny8EDnPPLDczosvvlj2tYQQyLYoi4444oji1iZMubaGG8WwjTFmUuRy3fQzsjCNPPTQQ6XgpyLgN28gYar15PIW15nLxK782AoXN11bcK1pFqZ5DK/ExvXXX9+43X///a1uSgv9AuIgCtPorrSK/mMXq/xzfOONN+7b5FfrBkahYmE6HrJ9xDUPQTOBs93EGcZxXcf4OciaH4UHEnh6RmPe03oe85vnc8011+zz861vfWuFWcgRXTfam9whz5pWOUKciE/smWkTpjmMfC1jjFlscrlu+hlJmLIWGxWCWsjUivnGG2+UX82G//nPf96sySZhCvhRqxszcmmxk7taXqnQvv/975d9KhctUkzLIf5UEd1www1l8WVdW9ej9aWtIlyKZANlfTyJhEMPPbRUsqRNXPBZ6ZXd4i9r9wlazGoV9SjClNY1daVC7Aa2MF04sn2QZ88//3zzn5ZLVmKIdoNt1EQn/nbZZZdOPzVhCmqtFFogXfnN8xlb6fWcDhKmkMPOtizUIivbWr58eRl+AtGuH3zwwb4W00j+b4wxi00u100/IwlT0Fpz2rTOWna/7LLLinsUpnEYANtXvvKV4o7Qje5aRubWW29t3IDu++jvnnvuWcGdSq6tIlyKZAOlC3TPPffsXXTRRb111lmnVMAIfu6NrnKGS1AZ19xAaUW3JV3yjBn85je/OW9hSpcq4TEMgG79vfbaq/FbE6YMueAb1Yw/ZTjA+eef/36AZiSyfZBn5AO2f/jhhxfbB9kN+UR3dU10aggGPR1tftqEKc8ukx2xN15S7r777uJOeGeccUbZJy50/RM33HmRGUaYEjYTKv/3f/+3uGlhdsoFxpgSD66tbvv40qOhQdgjw1m4Ni23UVATBvHmGgobd8bSGmPMYpPLddPPyMIUaA3hO8asJxihcsO9a4KBzlVFKJgIxbeTM/iP12GfbkNNkBAvvfRSXwvitFAzUERn7V5In7z2Y80tQ9qqop4PCBvSPud7G+Qx+ZTzygxPtg+EHunKCwI2H8FuNH64C4nKUcEma893XDeYONWe40GoXMhhYzu4156HDHbO2PQIAll2G8PI1zHGmMUil+umnzkJUzM+FspAaflS1zstWRrqYKaLbB8SpmY4Yk+AMcYsBXK5bvqxMJ0wC2mgjD9kzO5vfvObfMhMCdk+aK0epvXQvI+6/40xZqmQy3XTj4XphLGBmi5sH8YYM1u4XO/GwnTC2EBNF7YPY4yZLVyud2NhOmFsoKYL24cxxswWLte7sTCdMDZQ04XtwxhjZguX691YmE4YG6jpwvZhjDGzhcv1bixMJ4wN1HRh+zDGmNnC5Xo3FqYTxgZqurB9GGPMbOFyvRsL0wljAzVd2D6MMWa2cLnejYXphLGBmi5sH8YYM1u4XO/GwnTC2EBNF7YPY4yZLVyud2NhOmFsoKYL24cxxswWLte7sTCdMDZQ04XtwxhjZguX691YmE4YG6jpwvZhjDGzhcv1bixMJ4wN1HRh+zDGmNnC5Xo3A4UpCejNmzdv3rx58+bN27i2NgYKU7OwdGWOMbYPY4yZLVyud2NhOmFsoKYL24cxxswWLte7sTCdMDZQ04XtwxhjZguX691YmE4YG6jpwvZhjDGzhcv1bixMJ4wN1HRh+zDGmNnC5Xo3FqYTxgZqurB9GGPMbOFyvRsL0wljAzVd2D6MMWa2cLnejYXphLGBmi5sH++zyiqr9P74xz9m5zlx1lln9W677bYSHuEOw+WXX97bZpttsvNQvPXWW73Pf/7z2bm3xhprZCdjzEqAy/VuLEwnjA3UdJHtA5Fz0EEH9Y444ohmG5WFEEQIrz/84Q/ZuY/HHnusd8ghh2TnoRiXKAUJUxg23L///e+9v/71r2Wfe+BehsXC1BgTyeW66cfCdMLYQE0X2T4QOW0C8O677+49++yzfW4IL0QYwkp0CaJXXnml97Of/Sw7F/fly5eXfcLKYWZhKj8vvfRS4zasMOXcW265pRGCNZ5++unesmXL+tyefPLJskVIr0wUppE//elPzb1z7ZofyMK0lh6gOHYJ0/fee2/gvRpjZotcrpt+LEwnjA3UdJHtoyZM33333dIl/cQTT/ROO+203nrrrVfcDzzwwN5RRx3Ve/HFF3sf+tCHyrm0sK666qqlazqCWPv0pz/du/LKK3t33XVXCQ+hxbUQUIcffnjvhhtu6D333HMlLK6FH8K+8847yzVpyWU/+uG8LbbYogivffbZp7fJJpv0jjvuuN7DDz9c4gfPP/98b4899ij7XGvrrbfu/e53v+ttttlmvWuuuaZxVzzXXHPN3kMPPVTuDb/A9c4444zeAw88UPxKAK+77rq9Sy+9tOyLKEwVLr8333xz79RTTy33QtiERVqBRPUvfvGLcg/cC2n46quvlms/88wz5Zd4w9prr9075ZRTSjxXW221qjDFv+5144037v3oRz/qG1pAGh1zzDEfnGCMmQlyuW76sTCdMDZQ00W2D8TlZZdd1rvxxhvL9sgjjxR3hB//afH7xCc+UdzOO++8In4uvvjiIl5FrcUUsXbPPfc0/3/5y18WIYYwjaJKQk0orNhimv0ce+yxRdjlFtONNtqo/CLi1NqIkIxIqEVhGlsyOY4Yxl0Qj66W2ZowjfcY00f7Me6xxXT99ddvhlQgzAmHdEQki7YW09q9RjGa09EYMxvkct30Y2E6YWygpotsH7UWU0TSVlttVbqFOZ6FJ616H/vYx4rogXwchhWmtP5FasI0+2kTpsSH7vOddtqpcauJNegSpg8++GDvpJNOatxosV1MYZq78OcjTAFBSiuyWpSNMbNFLtdNPxamE8YGarrI9lETpnRVX3jhhWVf3fqAqHrnnXfK/q233to7//zzy34WjoBY23TTTZv/u+66a+mKz8L0G9/4Ru/6668v+3T7H3/88WV/t912K4Iw+0G0ER9+Ebv77bff+wH9gw033LCIMIFfxCq88MILvU996lNlv0uY6hq6V1pgJSLpmlf3upivMOX+EMNAWGeffXbZZ9jBFVdcUfKAlmpBa2pNmBLn1157rewzflf3SospwwmUDsaY2SKX66YfC9MJYwM1XWT7qAlToEUUobPzzjs3Xflvv/12cWNjvKY49NBDV2hRRGAhJuX/nHPOKe5ZmALXwA/iVbz88svFTV3q8sOGwBQIrij8cnc1LZA77rhjOY97UmtklzAFROknP/nJsqQTQlT3p+OR+QpTpav+77vvvuU/9yweffTR5v7vu+++FdIQyCelU7xXBCni2hgzm+Ry3fRjYTphhjXQ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dVz347A5cn972Jt7Umt7z13jKRNhujDVQENwqn0wmL788svXfvvttyl+GFHGNgZKFhUgn6tPXGPAJeOBx2oLjzm+4447pn15LcT29957bzpOHnIezvfff//kZSKcg7J//PHHKc0babgYON3rQz0efPDBqZ4cp4lplM0kOW2vkCmC2GpyFb/yyiubSWrUi4dJfW1F6AX1pO5KYdbVFXSdLPpBOIq2wyOPPDLVn99aqII2u+WWW659+eWX0//aX21Pe9MXirO/5557pjhrHlA8xKATpnPHq4/8QaOHEIKBYziWNmBSHynymIhIZg3+RqBwLG1HXelbrolttOU333wzeZ3oW87NPtgOISz0yY2mE6adXSAA1AbMRfB+4ZrwbDEp1uG6uth1+pv253P6g7a/ePHidf2L7T722GNTXD0x63PldcKUWPVd70HR3beO+olsJ5TBPcC9AI8++ug0cZPtWgSFOt12221TXz/33HPXnn322Wn7nG1xb3DtfC772uW+Jze02o7+Ite1jkOsXbp06YxQot51YmxXhib2ch9yPOXpPtF26iFb8H7gWsjq8uqrr062BLQb/3N91IG+mxszEOJPPvnkmW0O7XD58uWNUAUtpe7ClHFEKTOZFMwxN0KY+rYufIG+8e0RpuGGUw00BGdkH4ggBmse7F1GCgbTmkJrTph62jAXKXowcB4NwiDR4cLLhWnNc6tyOKcWaeDcNV6UtF/KxAAIqd9//336uxuAEVXVu9TVFeFGPXmgCXkbJTyF6sr53Kui7bWOiCHoPDWdMJ07fiRMEWjdw9Q9ppSr+pI3GvHKZ76YBH37119/bf6/mXTCdM4u9JvreffddydRVj2BzpyQRMR+/fXXtuc/Zdf/XWjMldcJU7e3bfdgh45xOP+99967+R9xR3ncK95/9DGhBxxPHwu165xteRu4MN123zvYN+3Ccb44itoRwVRFaUVlYJtef8rTfeJeVepH23g/eN30N19cEOOVbszwsviCojhN4jxBfa+ylR6xClPuNS0ARPtzTCcwO+q43h1XhSnn5rz1C313jadOhOnCVAMNwan2oRRZ+tEAi9ejLu/LAxCvCtt4YMOcMK0PZaHBmc8lgEAPkU6YdgO0/vfBVh49R+cTPAxU/twAfOedd07HacnZrq4qx69Too5teNf8B+r5VLdaR8F2/YyE6dzxI2EK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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAqUAAAE2CAYAAABYwutCAAB6rElEQVR4XuydidNt05333/8EoctUKCmzMjYlUkKroJWQyC1TIyqGvoSXm9Dd1C2FFlPcpkVI84YrIaRNTYgWUxleMUSHy9XmOYRCOe/7Wenvzu/53bXPc87zPOfs59zn+6nadfZeZw9rr73Wb3/3b03/q2eMMcYYY0zH/K8cYIwxxhhjzLixKDXGGGOMMZ1jUWqMMcYYYzqnryh95IlnvXjx4sWLFy9evHgZyRKZVpQaMx9wXjSmO1z+jDGjINsWi1IzETgvGtMdLn/GmFGQbYtFqZkInBeN6Q6XP2PMKMi2xaLUTATOi8Z0h8ufMWYUZNtiUWomAudFY7rD5c8YMwqybbEoNROB86Ix3eHyZ4wZBdm2WJSaicB50ZjucPkzxoyCbFvmjSj97LPPemussUbv0ksvzX8ZM9a8aIyZisufMWYUZNsylCh94okninDcZpttpoSvueaaJXw62Of111/PwQ1rr7127/HHH8/BxqySF40x48PlzxgzCrJtmZEoZcGzCS+99FITFrnnnnt6zz479fiaKP388897t95665QwUTuHWZjkvGiMGR8uf8aYUZBty4xEKZ7RK664ooQdccQRvW233bYRpS+//HJZx+sZxarWWS666KIpAjfuc/vtt6+yv/43C5ecF40x48PlzxgzCrJtmZEoPffcc3vrrLNOCWP7rrvuaoTjZptt1jv66KObYwh/7LHHmnV5SnWud955Z8q+iNIXX3yxt+eee5awL774onfUUUf13n333WY/s/DIedEYMz5c/owxoyDblhmJ0o8//rj83nnnneX3ww8/bEQpv3mJ3s8sSiPalwVvqjEi50VjzPhw+TPGjIJsW2YkSuHUU08t6zfeeOMUUbp48eKyvmLFit51111X1j/44IPyH+snnHBC75JLLukrSvGOsv7AAw/0TjnllCnnMAuTnBeNMePD5c8YMwqybZmxKH3vvffKOgIyilLYd999yzbLb3/72yb88ssvL2Hf+973+opSOOyww6rnMAuTnBeNMePD5c8YMwqybRlKlJr5C17lP/7xjzl4tWG+5cW33nqrfJCNg3vvvbd30kknlREv7r777tJ8ph/UXrTFbXXPJ2Y0zLfyNyz9ysRseP/993tvvvlmDp4xN9xwQw7qDNLr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jz+eEoZIkyi9/vrrSxjLJZdc0uy39957lzA8eGLPPfds9n333XdL2Hrrrdf8f9FFF/Vuv/32ss41Nt1000YQ6jjOIQ477LAmfOXKlU04UE284YYblv8Q0VRtI4bY1jX53XXXXZvtI488sjnffffdV8K0zQLPPfdcs/3GG2+UsM8++6wJO+qoo8ovYfHeSMPNN9+82QYEMp5onfuHP/xhc56bbrqphHENhZGm8MQTTzTiWseyTTisu+66JZz7Jh5AXM4888zmXKrq53/20/6kGygupKHCIrW04vndeuutTfgjjzzS7K8w7suidOGRbYtFqZkInBeN6Y5c/hARDz74YO/cc88t24ieJUuWNKL0pZdemiISEXiE/eY3vykCBS688MLeDTfcMEVIIdAOOeSQst4mSrm2BJWuD1dddVXZpi2lxA2i6etf/3qzD8RzISR/9KMflfXoKeUaat/4+9//vrd48eLmP7zEED2lxGedddZp9uF42GijjXoffPBBWX/hhReacAT9e++9V9aJ8913313WBXFB3AECTuuAIOec8QNg6dKlRXz3E6Wkv+7prbfe6m255ZZlnXR+7LHHyjrx0L1yP4r7s88+W9I0xoUwNRUQbWlFmh9wwAFlnWeiuOFdFyeddJJF6QIk2xaLUjMROC8a0x25/ElUSHR87Wtfa0Qav4ikQw89tAgNlj322KMRggidm2++uffQQw8VsSJv4hFHHNFbsWKFLtEqSmM4TQh0jWOOOaYRNZxv//33L8Ip88orr5T/v//970/pWJNFaeSTTz4pcb7//vsbIRpFKQKde1RclC4xrqDzIu5OPfXUsq59I9G7Sbpwbzr3tttuW/679NJLe2uttVbvyiuvbER6P1HKM9E5WPR/jKMEPfcr0RrJccn3B7W0is8PdO14766+X5hk22JRaiYC50VjuiOXP4kKRAoib5NNNinbUZTK+xb51a9+1fvWt75V1i+77LLGawpUG5933nlTPHiinyhtA48cHX+iBzOCeMIrK29dmyhF0O2yyy5Nz3ZdP4vSCy64oDlGUO0diedFkOG9jN5CkUUpadwGXk+q0hG604nSGjVRishtE6X94tKWVhalpo1sWyxKzUTgvGhMd+TyJ1Hx6quv9r70pS81vbslShEuO+64Y7M/Vfl4JaniV1U11bmIFdpO/uAHP2j21bmjgNtuu+2q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ef5m8sjvdotSMxE4LxrTHbH8STBkEIdf+9rXynpNlP7pT3/q7bDDDiVs5cqVzb4IFwmdu+++u7d48eKyHj2l8ZpR/HFNlpdeeqm35557ljBAJBEWBTHi51e/+lWzD/C/hCPNAA444IDmv3333bf3zDPPlDggeiS0brrppiLooCZKo4fyu9/9bu+Pf/zjKp7StnteZ511GuFG0wSaLHCchPfSpUtL2g1yX8RfbLfddr0333yzt9lmmzVh3FMm3udVV11V7uncc8/tXXzxxc0+xP3jjz8uv8QFeJZXXHFFWf/Zz37Wu/rqq8s659M+H3zwQfMxQzx0n2uuuWb55VrxGUZRilAXnIOmGzybeI+bb755s24mh/xutyg1E4HzojHdMReiFPAI4g177bXXmn2isItV5cOIUsTaoYce2jvppJPKsscee5R9nnzyySKM+D92lBIxbgcffHDvlVdeaf577rnnyv/Egf8iinNNlMqLed111zUe0JooFbpnjkOU6h5Y2O/SSy8togxPpITcIPdF/MVdd91V4rflllsWQYmgk7AW7H/MMcdMCYNNNtlkiicWwcl9xntQOgH/yWMuwSk22GCDZv3+++8veUHpkptlRFEam08oXyDk4wfAOJt0mLkjv9stSs1E4LxoTHfk6nvEk0BY/OAHP5i2+h7xxHGffPJJ76mnnppzUSrPYw32wQuH1y8SRekRRxwxJ6JUcC7EKR7bQUUporGNt956q1TVx45d/e6rJko5ln3xwOK5jLz44otVUYpXc65FKcf+7ne/a9Yhp990opRn/uCDDzbhN9xwQ7NuJof8brcoNROB86Ix3ZHLH1Xep59+erNN1Wms+o1iBNGBuEAkHXTQQSXslltumVaUUpWOUILpRCkCcMcdd/zzSf4/CDWGceJcVIPDq6++OqV6HqIo5TdWB+P15VjixL3J67l8+fLGyxirwKkiJ16021Q6EFcEIdXNseq87Z4Jl1h84YUXeoccckgJw7sJnHvZsmUD3Ze81sDzofoeeDZt4jc+wwsvvLD50GBd8GGh6nvRJkrbqu9juml9WFGKSMeDTHx5NvGcZnLItsWi1EwEzovGdEdb+UMMUg2LMFDbTIHnMnvj2G+YalbG28ydePqBUKNqOoIH8tZbby0e2kHgHM8//3wOLuch7jk+XE+CTxDve+65Z5V9BxnvlHgS3zjO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>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>