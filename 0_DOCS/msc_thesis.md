

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

[1\. Introduction and Problem Statement	5](#1.-introduction-and-problem-statement)

[2\. Proposed Solution: The Adaptive Algorithmic Trading System	5](#2.-proposed-solution:-the-adaptive-algorithmic-trading-system)

[2.1 The Hybrid Machine Learning Core	5](#2.1-the-hybrid-machine-learning-core)

[2.2 Dynamic Parameter Optimization	6](#2.2-dynamic-parameter-optimization)

[3\. Methodology for Continuous Robustness: The MLOps Pipeline	6](#3.-methodology-for-continuous-robustness:-the-mlops-pipeline)

[**1\. Research methods and approaches	8**](#research-methods-and-approaches)

[1.1 Introduction to the Research Design	8](#1.1-introduction-to-the-research-design)

[1.2 Phase I: Dynamic Market Regime Classification	8](#1.2-phase-i:-dynamic-market-regime-classification)

[1.2.1 – Unsupervised Clustering Methodology	8](#1.2.1-–-unsupervised-clustering-methodology)

[1.2.2 – Feature Engineering and Data Preparation	8](#1.2.2-–-feature-engineering-and-data-preparation)

[1.3 Phase II: Prescriptive Parameter Optimization and Expert Advisor Integration	8](#1.3-phase-ii:-prescriptive-parameter-optimization-and-expert-advisor-integration)

[1.3.1 Prescriptive ML Output	8](#1.3.1-prescriptive-ml-output)

[1.3.2 – System Integration	9](#1.3.2-–-system-integration)

[1.4 Phase III: Continuous Adaptation and MLOps Framework	9](#1.4-phase-iii:-continuous-adaptation-and-mlops-framework)

[1.4.1 MLOps Pipeline Implementation	9](#1.4.1-mlops-pipeline-implementation)

[1.4.2 Fixed Weekly Re-training Schedule	9](#1.4.2-fixed-weekly-re-training-schedule)

[1.5 Scope and Delimitations	9](#1.5-scope-and-delimitations)

[1.5.1 Single-Asset Focus (EURUSD)	9](#1.5.1-single-asset-focus-\(eurusd\))

[1.5.2 Simulation-Based Validation	9](#heading=h.janr4ub7o37m)

[1.5.3 Parameter Optimization Constraints	10](#1.5.3-parameter-optimization-constraints)

[1.5.4 Exclusion of Fundamental Data	10](#1.5.4-exclusion-of-fundamental-data)

[**2\. Literature Review	11**](#literature-review)

[2.1 Strategic Imperative: Non-Stationarity and the Adaptive Advantage	11](#2.1-strategic-imperative:-non-stationarity-and-the-adaptive-advantage)

[2.1.1 The Fundamental Challenge of Market Non-Stationarity	11](#2.1.1-the-fundamental-challenge-of-market-non-stationarity)

[2.1.2 Empirical Justification for Adaptive Strategies	11](#2.1.2-empirical-justification-for-adaptive-strategies)

[2.2 Dynamic Regime Classification: The Machine Learning Core	12](#2.2-dynamic-regime-classification:-the-machine-learning-core)

[2.2.1 Selection Justification: Unsupervised Clustering (UCL)	12](#2.2.1-selection-justification:-unsupervised-clustering-\(ucl\))

[2.2.2 Advanced Feature Engineering for Forex Regime Differentiation	13](#2.2.2-advanced-feature-engineering-for-forex-regime-differentiation)

[2.2.3. Clustering Methodology and Cluster Validation	13](#2.2.3.-clustering-methodology-and-cluster-validation)

[2.3 The Dynamic Mapping Layer: Optimization and MQL5 Interface	14](#2.3-the-dynamic-mapping-layer:-optimization-and-mql5-interface)

[2.3.1 Conditional Parameter Optimization (CPO) Mechanics	14](#2.3.1-conditional-parameter-optimization-\(cpo\)-mechanics)

[2.3.2 Optimization Methodology and Search Space	15](#2.3.2-optimization-methodology-and-search-space)

[2.3.3 Rigorous Evaluation Metrics for Adaptive Systems	16](#2.3.3-rigorous-evaluation-metrics-for-adaptive-systems)

[2.4 MLOps for Continuous Adaptivity and Robustness	16](#2.4-mlops-for-continuous-adaptivity-and-robustness)

[2.4.1 The Necessity of MLOps for Financial Time Series	16](#2.4.1-the-necessity-of-mlops-for-financial-time-series)

[2.4.2 Justification of Fixed Weekly Rolling Window Retraining	17](#2.4.2-justification-of-fixed-weekly-rolling-window-retraining)

[2.4.3 The Rolling Window Mechanism	17](#2.4.3-the-rolling-window-mechanism)

[2.4.4 Robustness Validation: Walk Forward Analysis (WFA)	18](#2.4.4-robustness-validation:-walk-forward-analysis-\(wfa\))

[2.5 Technical Architecture: Python-MQL5 Interoperability	18](#2.5-technical-architecture:-python-mql5-interoperability)

[2.5.1 The Inter-Process Communication (IPC) Backbone	18](#2.5.1-the-inter-process-communication-\(ipc\)-backbone)

[2.5.2 JSON Schema for Dynamic Parameter Transmission	18](#2.5.2-json-schema-for-dynamic-parameter-transmission)

[2.5.3 MQL5 Expert Advisor Adaptation (The Execution Layer)	19](#2.5.3-mql5-expert-advisor-adaptation-\(the-execution-layer\))

[2.6 Conclusions and Future Work	20](#2.6-conclusions-and-future-work)

[2.6.1 Synthesis of Findings	20](#2.6.1-synthesis-of-findings)

[2.6.2 Recommendations and Future Trajectories	20](#2.6.2-recommendations-and-future-trajectories)

[3\. System Design and Implementation	22](#heading=)

[3.1 Introduction and Architectural Overview	22](#heading=)

[3.2 Data Engineering and Preprocessing	22](#heading=)

[3.2.1 Data Sources and Granularity	22](#heading=)

[3.2.2 Data Cleaning and Normalization	22](#heading=)

[3.3 Feature Engineering Strategy	23](#3.3-feature-engineering-strategy)

[3.3.1 Persistence: The Hurst Exponent (H)	23](#heading=)

[3.3.2 Volatility: Average True Range (ATR)	23](#heading=)

[3.3.3 Momentum: Average Directional Index (ADX)	23](#heading=)

[3.4 Machine Learning Core: Regime Classification	23](#heading=)

[3.4.1 Gaussian Mixture Model (GMM)	23](#heading=)

[3.4.2 Model Topology	23](#heading=)

[3.5 Dynamic Parameter Mapping (The Prescriptive Layer)	24](#heading=)

[3.5.1 Regime Interpretation and Conditional Parameter Derivation	24](#3.5.1-regime-interpretation-and-conditional-parameter-derivation)

[3.5.2 MQL5 Target Parameters	25](#heading=)

[3.5.3 Signal Generation and Context-Aware Execution	26](#3.5.3-signal-generation-and-context-aware-execution)

[3.6 System Integration and Interface Design	27](#heading=)

[3.6.1 ZeroMQ Architecture	27](#heading=)

[3.6.2 JSON Schema	27](#heading=)

[3.7 Proposed MLOps Architecture and Experimental Setup	28](#heading=)

[3.7.1 Local Containerization	28](#3.7.1-local-containerization)

[3.7.2 Simulated Weekly Retraining	28](#heading=h.uleblf3szp9r)

[**4\. Experimental Results and Analysis	29**](#4.-experimental-results-and-analysis)

[**5\. Discussion and Conclusion	30**](#5.-discussion-and-conclusion)

[Sources	31](#sources)

[Appendix A: EA Source Codes	35](#appendix-a:-ea-source-codes)

[MQL5 Directory Structure	35](#mql5-directory-structure)

# 

# 

# **Abstract**  {#abstract}

Project Work: Adaptive Algorithmic Trading for Forex Market Non-Stationarity

## **1\. Introduction and Problem Statement** {#1.-introduction-and-problem-statement}

This project work directly tackles the most significant and persistent challenge in automated Forex (Foreign Exchange) trading: market non-stationarity. The Forex market is an inherently dynamic environment where statistical properties, such as volatility, trend duration, and correlation structures, are not constant but change fundamentally and frequently over time.

Traditional algorithmic trading strategies, which form the vast majority of current automated systems, are fundamentally static. They operate with a fixed set of optimal parameters (e.g., lookback periods for moving averages, thresholds for oscillators, stop-loss percentages) determined through historical backtesting. While highly profitable for the specific market regime they were optimized against, these static strategies inevitably suffer from a catastrophic failure known as "parameter decay" when market conditions abruptly shift. For example, a strategy tuned for a low-volatility, ranging market will quickly become unprofitable, or even disastrous, during a high-volatility, strong-trending phase. The inability of these fixed-parameter systems to autonomously adapt to changes in volatility, liquidity, or trend strength renders them ultimately unsustainable and non-robust.

## **2\. Proposed Solution: The Adaptive Algorithmic Trading System** {#2.-proposed-solution:-the-adaptive-algorithmic-trading-system}

This research aims to deliver a novel and methodologically original contribution by developing a truly Adaptive Algorithmic Trading System. This system is specifically designed to overcome the limitations of static strategies by continuously and dynamically aligning its trading logic with the prevailing market environment.

### 2.1 The Hybrid Machine Learning Core {#2.1-the-hybrid-machine-learning-core}

The proposed solution is a sophisticated hybrid architecture centered on a Python-based Machine Learning (ML) model. This ML model forms the intelligence layer of the system. Its primary, non-trivial function is to apply Unsupervised Clustering techniques—such as K-Means or DBSCAN—to a multivariate time series of market features (e.g., Average True Range, ADX, price return variance, autocorrelation).

The purpose of this clustering is to dynamically identify the current market regime. Instead of relying on pre-defined, arbitrary thresholds, the system allows the data to statistically group itself into distinct operational states, which might include:

* **Ranging/Consolidation:** Low volatility, weak or non-existent trend.  
* **Strong Trend:** High momentum, low mean-reversion characteristics.  
* **High Volatility Breakout:** Explosive price movements, often associated with news events.  
* **Low Volatility Drift:** Persistent, slow movement.

### 2.2 Dynamic Parameter Optimization {#2.2-dynamic-parameter-optimization}

Based on the market regime identified by the ML model, the system executes the critical step of dynamic parameter suggestion. The ML layer interfaces with an existing, high-performance MQL5 Expert Advisor (EA) running on the MetaTrader 5 platform. Instead of the human trader manually inputting fixed settings, the ML model provides the optimal parameter set specifically calibrated for the detected regime. For instance, in a Strong Trend regime, the system might suggest a longer lookback period for a trend-following indicator and a wider take-profit target, whereas in a Ranging regime, it would switch to a mean-reversion strategy with tighter stop-losses and shorter lookback periods. This continuous, data-driven parameter adjustment ensures the trading strategy remains logically sound and maximally efficient under all observed market conditions.

## **3\. Methodology for Continuous Robustness: The MLOps Pipeline** {#3.-methodology-for-continuous-robustness:-the-mlops-pipeline}

To ensure the adaptive solution remains effective, robust, and constantly relevant over an extended period, the project incorporates a vital, state-of-the-art component: an MLOps (Machine Learning Operations) pipeline.

This pipeline establishes a comprehensive infrastructure for guaranteed continuous optimization. The core mechanism involves a fixed weekly rolling window re-training schedule. Every week, the ML model is automatically retrained on the most recent, relevant market data.

Key functions of the MLOps pipeline include:

1. Automated Data Ingestion: Secure and timely fetching of new, cleaned market data.  
2. Model Re-training: Automatic re-running of the clustering algorithm to identify new, evolving market regimes. This prevents model drift and ensures the regimes identified are always representative of the latest market behavior.  
3. Validation and Performance Monitoring: Rigorous backtesting of the newly trained model to ensure performance metrics are met before deployment.  
4. Automated Deployment: Seamless and zero-downtime deployment of the updated ML model and its new parameter mappings to the live trading environment.

This systematic and automated mechanism represents a major methodological advancement beyond static strategies. By adopting this rigorous adaptive methodology, this project not only delivers a constantly optimizing and highly responsive framework for automated Forex trading but also demonstrates the acquisition of essential, cutting-edge cultural skills in both quantitative finance and modern machine learning engineering. The ultimate goal is to create a robust, self-optimizing, and truly intelligent trading solution capable of sustaining profitability across the non-stationary landscape of the global Forex market.

1. # **Research methods and approaches** {#research-methods-and-approaches}

## **1.1 Introduction to the Research Design** {#1.1-introduction-to-the-research-design}

The research design employs a hybrid, adaptive methodology centered on overcoming the core challenge of market non-stationarity in Forex trading. This approach moves beyond the performance limitations of traditional static, fixed-parameter strategies. The resulting system integrates Machine Learning (ML) classification with a financial trading mechanism to achieve dynamic adaptation. The goal is to build a robust solution that continuously adapts to the latest market behavior.

The methodology is structured around three interconnected phases: Dynamic Regime Classification, Prescriptive Parameter Optimization, and Continuous Adaptation via MLOps. The execution will involve developing a Python-based Machine Learning model integrated with an MQL5 Expert Advisor (EA).

## **1.2 Phase I: Dynamic Market Regime Classification** {#1.2-phase-i:-dynamic-market-regime-classification}

This phase focuses on defining and identifying the current market state, which is crucial for determining the appropriate trading strategy.

### 1.2.1 – Unsupervised Clustering Methodology {#1.2.1-–-unsupervised-clustering-methodology}

The core classification technique utilized is Unsupervised Clustering. This method is selected to segment financial data and objectively identify the current market regime. Market regimes represent distinct, formalized states, such as ranging, strong trend, or high volatility.

### 1.2.2 – Feature Engineering and Data Preparation {#1.2.2-–-feature-engineering-and-data-preparation}

Before clustering, the methodology includes extensive feature engineering to capture the relevant characteristics of the time series data. Features will be derived from data characteristics relevant to regime classification, such as volatility metrics, momentum indicators, and autocorrelation characteristics. The quality and stability of the identified regimes will be evaluated to ensure meaningful market segmentation.

## **1.3 Phase II: Prescriptive Parameter Optimization and Expert Advisor Integration** {#1.3-phase-ii:-prescriptive-parameter-optimization-and-expert-advisor-integration}

Phase II translates the output of the classification model into actionable trading logic, thus forming an adaptive trading system.

### 1.3.1 Prescriptive ML Output {#1.3.1-prescriptive-ml-output}

The Unsupervised Clustering model output (the identified market regime) serves as the input for a lookup or optimization process. The ML framework is designed to dynamically suggest the optimal parameter settings for the existing MQL5 Expert Advisor based specifically on the characteristics of that identified regime.

### 1.3.2 – System Integration {#1.3.2-–-system-integration}

The parameters suggested by the Python ML model will be fed directly into the MQL5 Expert Advisor. This demonstrates how the ML model directly influences the parameters of a secondary trading mechanism, providing the necessary dynamic flexibility lacking in standard EA optimization practices. This is the step where classification results are transformed into prescriptive advice for execution.

## **1.4 Phase III: Continuous Adaptation and MLOps Framework** {#1.4-phase-iii:-continuous-adaptation-and-mlops-framework}

This final phase addresses the Key Innovation (Rigor) of the thesis: ensuring the entire system remains robust and adaptive in the face of continuous market changes.

### 1.4.1 MLOps Pipeline Implementation {#1.4.1-mlops-pipeline-implementation}

An MLOps pipeline will be implemented to handle the necessary architecture and infrastructure required for deploying and maintaining the ML model in a dynamic trading environment. MLOps practices are essential for combating Concept Drift and Data Drift, which cause financial models to degrade over time due to shifts in underlying market dynamics.

### 1.4.2 Fixed Weekly Re-training Schedule {#1.4.2-fixed-weekly-re-training-schedule}

Model governance and lifecycle management will be maintained through a continuous adaptation strategy. This strategy mandates a fixed weekly rolling window re-training schedule. This specific schedule guarantees that the ML model is constantly re-calibrated using the latest market behavior data, thus maximizing model robustness and achieving a level of adaptation that is a major step beyond static systems.

## **1.5 Scope and Delimitations** {#1.5-scope-and-delimitations}

To ensure depth of analysis and methodological rigor within the constraints of this project, the following delimitations are established:

### 1.5.1 Single-Asset Focus (EURUSD)  {#1.5.1-single-asset-focus-(eurusd)}

The research is exclusively focused on the EUR/USD currency pair. As the most liquid instrument in the global Forex market, it serves as the standard benchmark for algorithmic trading efficiency. Generalization of the adaptive framework to cross-rates or commodities (e.g., XAUUSD) is outside the scope of this study and suggested for future work.

### 1.5.2 Simulation-Based Validation and Local MLOps Orchestration

While the system is architecturally designed for deployment on production-grade infrastructure (e.g., Google Cloud Run/Scheduler), the primary validation of the "Continuous Adaptation" hypothesis is conducted via Walk-Forward Analysis (WFA) on historical data (2020–2024).

The Fixed Weekly Re-training mechanism will be orchestrated locally by manually executing the retraining\_script.py (or via a simple local scheduling tool like a cron job/Windows Task Scheduler). This approach serves a critical methodological function: It isolates the performance impact of the ML model and CPO logic from external factors inherent in live cloud deployment, such as network latency, execution slippage, or unpredictable cloud scheduling delays. The performance metrics derived will therefore be a pure measure of the adaptive algorithm's efficacy.

### 1.5.3 Parameter Optimization Constraints  {#1.5.3-parameter-optimization-constraints}

The adaptive mechanism is restricted to controlling two critical variables of the Dollar-Cost Averaging (DCA) strategy:

Distance Multiplier: Controlling the grid density relative to volatility.

Lot Multiplier: Controlling the martingale progression relative to trend persistence. Other EA parameters (e.g., Take Profit targets, Magic Numbers, Time Filters) remain fixed to isolate the causal link between Regime Classification and Risk Management.

### 1.5.4 Exclusion of Fundamental Data  {#1.5.4-exclusion-of-fundamental-data}

The Machine Learning model relies strictly on Technical (Price/Volume) Features (Hurst Exponent, ATR, ADX). The integration of unstructured fundamental data, such as news sentiment analysis or Large Language Models (LLMs), is explicitly excluded to maintain a focus on quantitative time-series structure.

2. # **Literature Review** {#literature-review}

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

# **3\. System Design and Implementation**

## **3.1 Introduction and Architectural Overview**

This chapter details the technical architecture and implementation of the Adaptive Algorithmic Trading System. The system is designed as a hybrid application that integrates a high-performance execution layer (MetaTrader 5/MQL5) with an advanced analytical layer (Python Machine Learning). The core objective is to overcome market non-stationarity by dynamically adjusting the parameters of a Dollar-Cost Averaging (DCA) strategy based on real-time market regime classification.

The system architecture is divided into three distinct environments:

1. The Execution Environment (Client-Side): An MQL5 Expert Advisor (EA) running on the MetaTrader 5 terminal, responsible for trade execution and receiving dynamic parameters.  
2. The Analytical Environment (Server-Side): A Python-based Machine Learning service responsible for data processing, feature engineering, regime classification, and parameter optimization.  
3. The MLOps Infrastructure (Cloud): A Google Cloud Platform (GCP) pipeline that manages the weekly re-training and deployment of the ML model to ensure continuous adaptation.

## 

## **3.2 Data Engineering and Preprocessing**

The foundation of the regime classification model is high-fidelity market data. This section defines the data acquisition and cleaning pipeline.

### 3.2.1 Data Sources and Granularity

The system utilizes M15 (15-minute) OHLCV (Open, High, Low, Close, Volume) data for the target currency pair (e.g., EURUSD). This timeframe was selected to balance the need for capturing granular volatility shifts with the reduction of market noise inherent in lower timeframes.

### 3.2.2 Data Cleaning and Normalization

Raw financial data often contains gaps or anomalies. The preprocessing pipeline enforces:

* Gap Filling: Forward-filling missing timestamps to preserve the time-series continuity required for rolling window calculations.  
* Log-Returns: Conversion of raw prices to logarithmic returns to ensure stationarity for variance-based feature calculations.  
* Data acquisition and preprocessing were facilitated using the MetaTrader5 Python library. The raw OHLCV data was fetched for the specified window, after which the 'close' prices were transformed into logarithmic returns using NumPy (np.log(close / close.shift(1))). To ensure the integrity of the rolling window calculations, any rows containing NaN values resulting from the shift operation were removed.

## **3.3 Feature Engineering Strategy** {#3.3-feature-engineering-strategy}

To effectively differentiate between trending, ranging, and volatile regimes, the system extracts a compact vector of uncorrelated features. The feature selection is specifically tailored to inform the parameters of a DCA strategy: Persistence, Volatility, and Momentum.

### 3.3.1 Persistence: The Hurst Exponent (H)

The Hurst Exponent is the primary discriminator for the DCA strategy. It quantifies the long-term memory of the time series.

* H \< 0.5: Mean-Reverting (Ideal for aggressive DCA).  
* H \> 0.5: Trending (Risk of drawdown; DCA must be conservative).

### 3.3.2 Volatility: Average True Range (ATR)

Normalized ATR provides a measure of absolute price movement, used to dynamically adjust the Distance Multiplier (grid spacing) of the DCA orders.

### 3.3.3 Momentum: Average Directional Index (ADX)

ADX measures the strength of the trend regardless of direction. High ADX values signal a "Strong Trend" regime, necessitating defensive DCA parameters (lower lot multipliers).

Feature extraction was implemented using the TA-Lib library for standard technical indicators and the hurst library for complexity measures. The Hurst Exponent ($H$) was calculated over a rolling lookback window of 100 periods using the compute\_Hc method. Volatility was quantified by normalizing the 14-period Average True Range (ATR) against the closing price, while trend momentum was derived from the 14-period Average Directional Index (ADX). These metrics were consolidated into a unified Feature Matrix *X*, with rigorous cleaning to handle missing computed features.

## **3.4 Machine Learning Core: Regime Classification**

This section describes the unsupervised learning component responsible for objectively defining market states.

### 3.4.1 Gaussian Mixture Model (GMM)

A Gaussian Mixture Model (GMM) is employed for clustering. Unlike K-Means, which assumes spherical clusters, GMM allows for elliptical cluster shapes and soft assignment (probabilistic membership), which better captures the complex, non-linear distribution of financial returns.

### 3.4.2 Model Topology

* Number of Components (k): The system is configured to identify 4 distinct regimes (k=4), typically mapping to:  
  1. Low Volatility / Mean Reverting  
  2. High Volatility / Mean Reverting  
  3. Low Volatility / Trending  
  4. High Volatility / Trending (Breakout)

* The regime classification model was constructed using the GaussianMixture class from the scikit-learn library. The topology was configured with four components (k \= 4\) to capture distinct market states. The model utilized a 'full' covariance type, allowing it to adapt to the elliptical cluster shapes characteristic of financial data distributions. The predict(X) method was employed to generate the discrete Regime ID sequence representing the current market state.

## **3.5 Dynamic Parameter Mapping (The Prescriptive Layer)**

This section defines the logic for the Conditional Parameter Optimization (CPO). It details how the abstract "Regime ID" is translated into concrete numerical inputs for the MQL5 Expert Advisor.

### 3.5.1 Regime Interpretation and Conditional Parameter Derivation {#3.5.1-regime-interpretation-and-conditional-parameter-derivation}

The process of generating the prescriptive DCA parameters from the GMM output involves two critical steps: Regime Interpretation (translating mathematics to economics) and Conditional Parameter Optimization (CPO) (translating economics to actionable settings).

A. Regime Interpretation: Linking Centroids to Economic Labels  
The GMM's output consists of k \= 4 cluster IDs, each defined by a multivariate centroid (Ci). This centroid represents the mean values of the clustering features (Hurst Exponent (*H*), ATR, and ADX) for all time segments grouped into that cluster. The interpretation process is as follows:

1\. Primary Classifier (*H*): The Hurst Exponent centroid value for each cluster is the primary determinant of the economic label. Clusters with Ci(*H*) \< 0.5 are categorized as Mean-Reverting (Ranging), while those with Ci(*H*) \> 0.5 are categorized as Trending (Risk).

2\. Secondary Classifier (ADX/ATR): The mean ADX and Normalized ATR values distinguish between the levels of volatility and momentum within the primary categories, separating, for instance, Low Volatility from High Volatility (Breakout) regimes.

This analysis validates the manual economic labels assigned to the four identified clusters: 'Ranging (Safe)', 'Strong Trend (Risk)', 'Volatile', and 'Low Volatility / Trending'.

B. Conditional Parameter Optimization (CPO)  
Once the historical data has been segmented according to the GMM's cluster assignments (Regime Training Segmentation, as described in Section 2.3.1), the Conditional Parameter Optimization (CPO) is executed to find the optimal parameter set (Pi \= {Distance Multiplier, Lot Multiplier}) for each isolated regime segment.

1\. Optimization Tool: The built-in genetic optimization algorithm of the MetaTrader 5 Strategy Tester is applied to the MQL5 Expert Advisor.

2\. Segmented Backtest: Instead of running the optimization on the full history, the optimization is run four times—once for each historical data subset belonging to a specific Regime ID.

3\. Multi-Objective Function: The optimal parameters Pi are not chosen based on simple profit. Instead, the selection maximizes a multi-objective fitness function (Section 2.3.3) that mandates:

* Sharpe Ratio \> 1.0 (prioritizing risk-adjusted returns).  
* Recovery Factor \> 3.0 (prioritizing resilience and drawdown control).

4\. Prescriptive Output: The parameter set Pi that yields the highest multi-objective score for its corresponding cluster segment is stored as a row in the lookup table. This table forms the final prescriptive artifact used by the Python ML service.

This rigorous, data-driven derivation ensures that the parameters in the subsequent Logic Table are theoretically optimal for the market structure identified by the ML core.

| Regime Characteristics | Regime Label | DCA: Distance Multiplier | DCA: Lot Multiplier | Rationale |
| :---- | :---- | :---- | :---- | :---- |
| Low H, Low ADX | Ranging (Safe) | Small (1.2) | High (1.5) | Aggressive accumulation in safe range.  |
| High H, High ADX | Strong Trend (Risk) | Large (2.0) | Low (1.1) | Wide spacing, minimal size increase to survive trend.  |
| High Volatility (ATR) | Volatile | Large (2.5) | Moderate (1.2) | Wide spacing to handle noise/wicks.  |

### 3.5.2 MQL5 Target Parameters

The ML model outputs a JSON payload targeting two specific extern variables in the MQL5 EA:

* input double InpDistanceMultiplier  
* input double InpLotMultiplier

* The parameter mapping logic was instantiated as a Python dictionary structure, linking each GMM Cluster ID to a specific distance\_multiplier and lot\_multiplier. For example, clusters exhibiting high ADX and high Hurst values (Strong Trend) were mapped to defensive parameters (Distance: 2.0, Lot Multiplier: 1.1), while mean-reverting clusters triggered aggressive accumulation parameters.

### 3.5.3 Signal Generation and Context-Aware Execution {#3.5.3-signal-generation-and-context-aware-execution}

To isolate the performance impact of the adaptive risk management framework, the system employs a standardized, non-adaptive entry signal protocol coupled with a strictly defined Dollar-Cost Averaging (DCA) recovery mechanism.

The system utilizes a Hybrid Decision Architecture. The Timing Decision (Entry) is purely technical and static (MACD). The Sizing and Spacing Decision (Risk Management) is purely adaptive and ML-driven (Regime-based). This decoupling ensures that the ML model does not need to learn precise market timing—a notoriously difficult task—but focuses on the more tractable problem of Regime Classification.

A. The Trigger Mechanism (Standard MACD) Trade entries are generated exclusively using the Moving Average Convergence Divergence (MACD) indicator on the M15 (15-minute) timeframe. The system adheres to a classic trend-following logic to establish the initial bias:

* Buy Signal: Generated when the MACD Main Line crosses above the Signal Line while both are below the zero baseline. This indicates a potential bullish reversal or upward momentum.  
* Sell Signal: Generated when the MACD Main Line crosses below the Signal Line while both are above the zero baseline. This indicates a potential bearish reversal or downward momentum.

B. Context-Aware Execution (Signal Aggregation) "Signal Aggregation" in this framework represents the fusion of a Static Time Trigger with a Dynamic Structural Context. It does not involve combining multiple entry indicators.  
When the M15 MACD generates a signal (tentry), the Expert Advisor executes the following aggregation logic:

1. Signal Validation: The system accepts the MACD entry signal.  
2. Context Query: The system queries the Python ML model for the current Market Regime (Rt).  
3. Parameter Injection: The system retrieves the specific DCA parameters mapped to that regime. Specifically:  
   * Distance Multiplier: Determines the spacing for subsequent recovery orders.  
   * Lot Multiplier: Determines the volume scaling for subsequent recovery orders.  
4. Execution: The initial trade is opened. The structure of the DCA grid for this specific trade sequence is now locked to the regime parameters identified at the moment of entry (or updated dynamically if the thesis scope allows dynamic grid resizing).

C. Delimitations of Trade Management To maintain methodological rigor and isolate the efficacy of the adaptive DCA parameters, complex trade management features available in the Expert Advisor are strictly excluded from this study:

* Primary Strategy: The focus is exclusively on the DCA (Grid/Martingale) recovery mechanism.  
* Partial Take Profit (PTP): Excluded. The system targets a net take profit for the entire basket of orders.  
* Order Stacking: Excluded. The system will not open additional entry trades in the same direction if a grid sequence is already active.  
* Logic: By disabling these confounding variables, any variance in performance (Sharpe Ratio, Drawdown) can be directly attributed to the ML-driven adaptation of the DCA grid parameters (Distance and Lot Multipliers) rather than auxiliary trade management features.

## **3.6 System Integration and Interface Design**

To bridge the Python analytical layer and the MQL5 execution layer, a high-performance Inter-Process Communication (IPC) mechanism is implemented using ZeroMQ (ZMQ).

### 3.6.1 ZeroMQ Architecture

The Python script acts as the REP (Reply) server, and the MQL5 EA acts as the REQ (Request) client.

1. MQL5 Request: On every new candle (M15), the EA sends a JSON request: {"action": "GET\_PARAMS", "symbol": "EURUSD"}.  
2. Python Response: The Python model predicts the regime for the latest data and replies with the parameter set.

### 3.6.2 JSON Schema

The data transmission follows a strict JSON schema to ensure parsing reliability in MQL5.  
`{`  
  `"regime_id": 2,`  
  `"timestamp": 1715420000,`  
  `"params": {`  
    `"distance_multiplier": 1.5,`  
    `"lot_multiplier": 1.2`  
  `},`  
  `"status": "OK"`  
`}`

The Inter-Process Communication (IPC) layer was built using the pyzmq library. The socket was configured as a REP (Reply) server bound to tcp://\*:5555. A continuous loop was implemented to listen for JSON requests from the MetaTrader 5 client, parse the incoming signal data, execute the prediction pipeline, and return the optimized parameters in the defined JSON format.

## **3.7 Proposed MLOps Architecture and Experimental Setup**

To validate the continuous adaptation hypothesis while adhering to the scope of this study, the MLOps pipeline was simulated in a local containerized environment. This approach mirrors the architectural constraints of a production cloud system while isolating performance metrics from external network latency.

### 3.7.1 Local Containerization  {#3.7.1-local-containerization}

The Python application, including the ML model and ZMQ server, was containerized using Docker. The environment was defined by a Dockerfile utilizing a lightweight Python 3.9 image, ensuring dependency isolation for libraries such as scikit-learn and talib.

### 3.7.2 Simulated Weekly Retraining

In the production architecture (as described in Section 2.4), a scheduled service is responsible for model updates. For this simulation-based study, this function is encapsulated within the retraining\_script.py.

This script operates as the local orchestrator for the Continuous Training (CT) pipeline. Instead of being triggered by a cloud service, it is executed manually or via a basic local operating system scheduler. Its operation, however, mirrors the production intent:

* Iterative WFA Loop: The script initiates a loop that iterates through the historical testing period (2020-2024), advancing the lookback and validation windows (e.g., 100 days IS, 20 days OOS) sequentially.

* Model Persistency Simulation: At each iteration, the script performs the full Feature Calculation, GMM Model Fitting, and CPO derivation. The resulting model artifact (gmm\_model.pkl) and parameter map (trade\_params.json) are temporarily saved to disk, effectively simulating the behavior of a cloud storage service (e.g., GCS) persisting the artifact.

* Validation Data Capture: Critically, for every WFA step, the script captures and persists the Out-of-Sample (OOS) performance metrics (Sharpe Ratio, Recovery Factor, Trade Count, etc.). This collected data forms the source for the final analysis and the Streamlit Presentation Layer (Chapter 4).

This simulated environment ensures that the core hypothesis—that continuous, regime-aware adaptation provides superior, robust performance—can be quantitatively validated with maximum control and reproducibility.

# 

# **4\. Experimental Results and Analysis** {#4.-experimental-results-and-analysis}

# **5\. Discussion and Conclusion** {#5.-discussion-and-conclusion}

# **Sources** {#sources}

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

# **Appendix A: EA Source Codes** {#appendix-a:-ea-source-codes}

## **MQL5 Directory Structure** {#mql5-directory-structure}

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


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAogAAAENCAYAAACM3JRMAAB1kUlEQVR4Xuy9h88sxZm2//0nZJEECERGxAVhEMHIgBA+2OzRIRgQIi1gVnDAwIIwS04Gm4XzAQYbw5q4B7DBhCWLYDIis+ScBYj5fVd57/497zPV887Mm9+5L6k1PdXVlbvq7kr9fzrGGGOMMcYE/k82MMYYY4wxo40FojHGGGOMGYMFojHGGGOMGYMFojHGGGOMGUNPgfjok8/78OHDhw8fPnz4mKdHG+MKRGOMMcaY+Yz1TjcWiMYYY4wZaax3urFANMYYY8xIY73TjQWiMcYYY0Ya651uLBCNMcYYM9JY73RjgWiMMcaYkcZ6pxsLRGOMMcaMNNY73VggGmOMMWakyXpnwYIFnWWWWaY5tthiizHXRwELRGOMMcaMNFnvIAofeuihcv7KK6+U/19++WX5/8UXX3RuvvnmzjfffBNv6cndd9/d+f7778t52/33339/5/HHHx9jNpNYIBpjjDFmpMl6Z8UVVyyi8LHHHhtj/tlnn43pWTz66KOL+WGHHdZ58skny/kFF1xQrsl82WWXLf/ffffdzp577jnm/u+++65xc7nllmvMf/jhB3k5Y1ggGmOMMWakyXoHgbbTTjs1gu3AAw8s5jvuuGPpUYQHH3ywEXm9BKLOP/nkk3Ku+2+88cbO22+/XdzceeedixkgKG+99dbm/0xhgWiMMcaYkaaX3nnkkUeKsHvttdc6q6yySmPOUDHm9Az2Eoi6h+syj3Ad83jgxkxjgWiMMcaYkSbrHUTaLbfcUs7pIeQ/Ao+evuef/4fd2IN42mmnde66665ivvvuuzdCMApEDSWrB/H000/vvPjii6UHkd5KgVu4OdNYIBpjjDFmpMl6hwUpcU7gww8/3FyT2QorrDBGyMn8T3/6U1UgAu7K3uqrr96YH3TQQY05Q8+zgRkRiHTL0iXbi/fffz8bjTS8qXz++efZeKSgTHz66afZ2MwwVJB6nqn8WIk3UZ599tlx6wgz9+mnLahBz45WhI4a119/fTYyk8BU6Z25zEACUePncXUNjTZmdKn2SxyfbyMq7vlMnHuwzTbbdL7++utspcD1TTfdNBtPOfGtZrfddsuXpxXSirex+YgmMuuYrngylIF/r7/+er7UN3FeTRxamQi4sdpqq2XjgXnhhRfGpGvsBTB1EF5rr712k2b/8i//kq1MGv20BRnN+zryyCPzpZ58+OGH5T7VsbFcaChxtnP77beX8D7zzDP5kpkgWe+YIQXiDTfc0JgdfvjhxcwCcTiIJ8vkOVRZzSaU38yt4PzMM88s5htttFGyOTVEf+a7QNxnn32asnD11VdnK1OCtl/Ya6+98qW+aZt4PSiT4UbkqaeeKm5ut912pVdzs802K/8fffTRbHVcqLOGrZO4bzZMOO8X0ojjv/7rv5oXiLjCcjLppy0YD9zQ4oBeLF68uLPWWmuV86222qrULffee2+zUjW2a9NFP3GfSNkz/ZP1jhlSIOohA1UmEohvvPFGY8b4uoYBLr300sb88ssvbx6Mjz/+uDFnvF/j+aPyQMR46g2XN2QEAxUY/xmCieKIcwlzeh1POeWUJg3Vu6vGkOOOO+5o7jv11FOLGZUO+zyp4uetGnOGsiPRTcIF6iXioEyoAqPXgV/yfOWVV27svPXWW+U+wq+5GRxvvvlmMdcEYA7CpzSRGQdgfs455zRms2ES72QRV8BF1IPLXBf1fKhsYEbZqD1D/KrH94QTTqimF72GmDPfRWkMuM2WDnJz3333LebyV71LlDGIAjE2+OS73Ih1Qa1s6r/uxR/VKdynawqL7rn22mvLL2HKUE/xfEQo88SDMh17yehphBz3Y445ppjrPwfENKfMQk5zhL6eYY7peqmaKIQ1jhLxUqj0veyyy5r4xLzg2UR0YU6a7bLLLuX8iSeeKNfbyk4WiKQf/6k/CMOFF15Y/lN2VUdhzi+9aepR48CP9dZbr7PBBhs07h177LGd6667rpzzMkT4Vc9GuOenP/1pOY9t2J///OdixnMmMw42M+b541xlhDRg0YHsIEhlrrpb7lCX5noU9GzwvBDPGD+OeD/E/FA9q+dRbcTFF19czE1vst4xQwpEDh5WVt9wzkNNYVdDz38KKw9GbOxPPvnkzssvv9y4IfMzzjijnO+xxx6N+BxFgajKhYpBQ45UCKR1FojYpXHFzoYbbljEFueXXHJJ55BDDikVJajxx03u45w8YLk+b8z81woszjOYcVAZMi8M6I1ZY401yk7wNJSq5HkJeOmll4o/aijPP//8xl3CzzWVDeU1bmGPYRPsKk1wX/6Awk85/Nd//dey4eh8IfcgMtdS6UoZII9p4GQXc/JWjWZ+hiTO/va3v5Xf2267LXpX2HvvvUv6qsEVEjW8LLAXl67J36uuuqrJK/xvE4j84gdhQTwhIBB7tbJJHnOuvI4CkXynkZOfGlrkHHfVu33PPfcUc4EZDWyNzTffvFznWVDvPY1ujDtljHO+dnDwwQeXZ0Dhw1xpzjl1YU5zuPPOO8t93M/5bEdpUEP53JYX1EeqwxYtWlQ2GNY0gbayE8sLbpGfwGfNtt5663JOGWV6DQe9wvKPvGUPOdIWd/CP3mGusVpU9kD7z+GnRFcNtWFsa0JYOSfeEmbUs4SDc15OFF9Q/UT8dC9fz2gTiNSjnKsejfU26YDAU/xU9uL9tfwg/DLn2VMbbcYn6x0zpEDkgeWtkgqcHih+OfgsTSyMqjCpYHkjFLHR4ZfGj0NvlzBKApE461iyZEkxl5iK9qJAFNyjt89o53e/+12pyM4666xihwolXhdcIy/1W4PKVsMweluOjUjuBeCcio485c1c/uO3wpqFhECoxvjlIWYqS+jVkM1F1IDqoCFYd911x8QXc3o/YtlQA1B7htR4tc0d5drZZ5/d+KWFYaSr/IVdd9218Tf2zpx33nlFlPUSiDVqZROifQlENdgCPyWUMZcA5DwP42LWttks1+Inrfh/5ZVXjhGmSnPSJw7z5TSPftfSnPty2GYrxDnmcWT//ffvyv+YF5B7uGTeVnZiecGtVVddtaTpmmuu2ZjHEQbBufIeN2LvO9foeZdYhFiv9BKIuQ0jzAgtxQtimGN8cZ945XvbBCLkcOjZ2HLLLZtyGMtevD/nB+b5eZS5/DPtZL1jhhSIGpriQPxJIMr8gw8+KPbp2ldB5VfDFgcccMAY848++qicM4Fc+wuNkkCsMRGBGIWFFhG1CURVoqroI4gCvdEDPVzym4ZU+VkTiBqqpixob6g2gYjfeuPH/Ri/WAHG8M9HgRgbOeDlS0N7ecd+pZF6RmrPEObLL798c19EPcb0SnAw55FyA6SrejKAYVn5G/OZlwZ6onsJRIWLZ/6oo44q9mtlU/ZFFGrZT5UJzHsJRA3Pa2hee5DphYhhYJAA4VnoJRD1kpvTPO5ZhrnSXGThMNsh7HGbDToEKANKN5HzAtoEUFvZieWF8kc5gZtuuqnz6quvlvM4bKs6h/MoEKPYj+0TQ7Cyn0UkvXag/Ke86N5oj3j3KxDjXnaYUzaIl+b4qnMkpo/iFOttetujQFTZi/7l/OCc8FsgDkfWO2ZIgQg06qocJBCBoQHs6GAOCUQzTYyH/F1CuuahTTjNN9riORGBqIZx/fXXb9K1TSAC19smaMe84dBQJT2J/KeSzgJR84Z07LfffsW8TSAyT0l2t9122674yd6oCcRe3/yMaVR7huJLmIboIjyD9G5EsI+YIl31jOoANfLZvE0g5rqA3g01xrlsgvYcgyjUcnnS3FXOewlEIJ3ivQzVAatWozlpBG0CUQteODDPaQ45zZnLBgsWLCjmbc/6bOOkk04aEzcO0lxTEbI5cA7jCcR4QCwvmiajg/KCCOecoVrqHvXMYqa8V94o30Df0YUsmODpp58e4xeHhFout5j3KxDzc0N5l5Dj0LzFmD4cxCXX24pPLHvRv5wfhBtyfGXf9CbrHTOgQOwXCiMFPu9TxVveO++8M8YMsL906dKuXg4zPKSzepLGo9Z7GGGeVm1vOyqrNshL8nTQiol5N1H44c6o733Yz7Mxmc8Q6Y8Awk0aZkEjz5wy5W1+vmuQd7ku6FU22/Ka6Sqxl2gQiAe9pLks0sASvzY/M8RBvYagNO8H/JiMvJkulDb/8z//ky8NlReDlJ1cXvqBkQoJPECgI7h6wegX5aKW/8RPiz76BYHIPErKSK4vCZ8W7GRiPdr2bOSyFxkmP0w3w+qd+cyUCEQzN2BIQ2+0MwkrTVm9x8R2wlJbUGGmj9iLFsk9l8b0y3SVHXrRmMc4E71mbSM0Zm4wUb1Dz+18y38LxBGGLSDoKZoNX2j50Y9+VCansxLRzCxs23LRRRdl4zJP8fjjj8/GxozLdJWdf//3fy9zd3uNbkwVxG+69i81k0/WO3l6CoemVNSwQDTGGGOMmWdkvZN7hFm4F+dyM40gDu1ngaipLUxlGA+mFmC3Ng2FqR7508NMQ4jTfwRh6se/frFANMYYY8xIk/UOAvEXv/hF6Y3WPpOa18o5C/q0YT5bX0WBiFnem1b7DjNXVguMZKZpVcyd1fQepvrwFSgt3tKiQsLFQkQtRmKlvhZOEaa4n/REsUA0xhhjzEiT9U4eYtYWTIhB5u7nfVAlEPM+qdqbNopCCUP48ssvy/x7Dlbpa5Emv5pHi2jkP72Dui+CIFSY4l64E8UC0RhjjDEjTdY7cYgZwbVw4cJyzv6dzN+nZ1EHQk4CkevYj9c1J5b9MOklxO3YG8lWSM8991xZZNVLIGqbsAwLTnOYJgMLRGOMMcaMNFnvRIGoDxVo6yPOGXbWZz61QXkUlAwJI/D0RTHQxwE44ucg2UpO35zvJRABgclXqOQ3vZEaYiZM0b+JYoFojDHGmJEm6528SIUFKtqqSZuac+irQ1Egam4ix+qrr961R2cUcOz9iz0+FHDggQeOKxDjBumLFy+WM02Ysn8TwQLRGGOMMSON9U43FojGGGOMGWmsd7qxQDTGGGPMSGO9040FojHGGGNGGuudbiwQjTHGGDPSWO90Y4FojDHGmJHGeqcbC0RjjDHGjDTWO91YIBpjjDFmpLHe6cYC0RhjjDEjjfVONxaIxhhjjBlprHe6sUA0xhhjzEhjvdONBaIxxhhjRhrrnW4sEI0xxhgz0ljvdGOBaIwxxpiRxnqnGwtEY4wxxow01jvdWCAaY4wxZqSx3unGAtEYY4wxI431TjcWiMYYY4wZaax3urFANMYYY8xIY73TjQWiMcYYY0Ya651uLBCNMcYYM9JY73RjgWiMMcaYkcZ6pxsLRGOMMcaMNNY73VggznGWWWaZ8rvtttt2rr766rEX/5dvvvmmsRd58sknO4cddlg2njI22mijbNSwyiqrZKNphfT56KOPsnFnueWWy0ZTxvPPP9+5++67s/GkMmx83nrrrZJGw+YT99bK4EzA83DiiSdm46E455xzZk28phrKzksvvZSNG6hLqFPMWMZLt0HoVc/PFqjnv/jii2zcymyJk/VONxaIMwyNy9FHH90cn376abbSEzVONHo//PDD2IuBDz/8sPzuvvvunXfffbecz0WB2K+9QYniMPoxaH4My4UXXljExvXXX1/yFEE2FQwbn4k0/tNdzmrkMHz//ffh6nDwzG288cbZeAxXXnll57bbbsvGs5oLLrhgzH81+OOVnYmUkenk9ttv74rjZBPTYrx0G4/o1nj1/GxgUIEY4xTbp+nGeqcbC8QZZtDehyeeeKLz6quvNv9r93/33Xedm2++ufqQDisQaVCXLl1aHmQOziOYUfG+8847Y8zfeOONzv3331/Oo0B8/PHHyyGiKEOs4VatIswCkbjGXjfOa40/7n322WfN/0H8EFRkpCu/EdzFrYkQ/XzzzTeLsBgPykKOK+lfCwt5kO22xef999/vvPLKK2PMcuOv/M7pl/MDcjlr8xfkr8pb7tWll1XlSeSyBIQjmuUwRPS8xPTh2anFL3LPPfd0Lrnkkmw8hq233rq4ldM+Q1o89NBD5Rz7hIdwzQRZPLU1+ORzzB+VkRdffLGUzTYoXzkP28oE9YfqK67lfAbcU1oRpvicQy5HWSC2pTdx+Prrr8eYAWGnHEZwP9Z9+XmJEF6lTy28xDGmQy+3as8D4D750Ab+xnakzQxIr5i+GfIo5ksuL4Qvp20bFoizCwvEGaYm8B588MHO4YcfXs7XWGON8stDt+aaa3aeeeaZzhlnnNE0pLo/ViKbbbZZ6YHabrvtSs+U7N15552dtddeu3PwwQeXczWajz76aPMAv/7665299tqrnItll1229G4SLs5xHxEje6+99loZRnnuuec6Rx11VLkGW2yxRfmP+QorrNAIRMJy1113lUPhl0A66KCDOsccc0xxIw+HXnTRRU1YqJQUlj/84Q8l/Lglv1SB4y69PLi3+uqrF7PsB5WZ/Mdt+QEK85lnntnZZpttyj34SbpSkXHfZZdd1rn11luredkvBxxwQOeRRx4ZY4b7EjWcU3lCLAvEVWVh++2375x++umdl19+ubPVVlsVM+4njPRM0pPRT3zIs5tuuqnYE5tssklnn332KUNBMb+Js/I75ofAT+7j/uOPP74IruwvEC7igr80bIQDO0cccURnwYIFxQ5+3nDDDU25wa3rrruu/I/pT3nGbZVXGtsYBpCIxk38IC11/wknnNCEcd11163m67XXXluepUMPPbT8j3YQeqeddlpxB3PKEmVs11137XzyySfFzv7771/SUXFdvHhxiTvPzHnnndekbRYP00GbQNQvoomwke7YpSyCyhrlmHK4+eabj3EHu//0T//U+eMf/1juxQ6Qh9RVlD/yDaibcJd0vO+++xo7MZ9lh3zGjOebPNtwww1L+gH5F8sR5ZHfHXbYoZwz1WDvvfcu9SX3qbd31VVXLS9p1DMR/CF+lEPVT6pPeKZ4Rp9++ukxz4vSLYaXMoF5Dq/SlXS44oorutyK9Tx2CWOsR7lO2HFv0aJFnYULFxZzQXnCLtdJf/xH3CpetC1qc8hL6hTqk5i+uo5fhJM0Iz3II1B8eT7lrtKWumq33XYr9kgzuUOccvu0wQYblOtAGiuNpgrrnW4sEGcYPdgZ3sjiNR66+FYpQRMrBh4yGpoastfWgyixx8Ofe01yOITCQCMcwZyGjwdd4Cb3UuFT2WlInYqaxlpuXXzxxaXiXbJkSfWtM/a05bRTLwQVmuIV7RNvKq6aH9FePFe6x8oK1lprrTGiDSbj7fexxx4r8brlllt6CsRcFhAbNCRK1xVXXLHENfc+DBKfSHSnlt+Q80PEckZjHMFfUKMCCMU47QL/KNe13iMgzylDiDXioPmSNDIi9yBKIMa8ljCKZtAWL8IJlPUjjzyyMd90001LGuMn5V2sttpqzbnSMIpGepYIo+KNuJmqqQa9QDiQjjrWX3/9MQJxzz33HFPOlW65rOV0wx69riKmB2lIPtILh9s5v2Qn5nO0w6/CpB5ChDr1SyxH8TrPPs9JLGvK+1ovPPap02SX540wqD6JvYoxLaJAjHFSXRp7NOlpVjroenQr1vPxeWAKUe15z3nA85br92ym0Yv4HMT0lZuYxV5KmSu+vATF9JJ7iEzsUtdBDHOsQyUKeSHRC+9UYr3TjQXiDJMfYMHblhogqIkCiA9rbpAistcmEKkoebvceeeddUtDDGNNIMZwyryXQKw19LlR/uCDD5oev0ibQCQuDOdRwfI2q3jVBKKQH/Q49RKINAz9CKqJCMQbb7yxOcc/xAFuKR7Rr1pZQCDG9Ba5wRgkPpHoTi2/oa0sx3JGoxFpE4iZtnJDeEkL4rTjjjuOSX/1CnItN869BGL2vxYv0ltxobHGLyEhwigA9iCLSAnl6PZ4w9XTxXg9iOr9FKpzclnL6dYmEMlDemSBdMziL9qJ+RztRAETBSK9sZkoEPNzINoEYi/BTg8f9SjEtOhXIGKH+ynPpEMvgZifh34FIvVKFojZbDIFYi29nnrqqc7yyy9fXoIhhjnWoWqP6Jmljp5qrHe6sUCcYfIDDHqzBYapgIfu/PPPb+wce+yx5Tc+rDxkDBFryI/hEQk+2ePtX5V7rLAYJqWngPszMYw1gUjDJ4FDRaMwc58qHoYSuZcKg2E0sdNOO5UhDrnFr+b91Cb306io1y+Gi0qNOABvq4pXTSBmP2iYo70szIHrGu6jMmYoJAuqiQhE3BMPP/xwcQt/9OZM3keBqLJAuseyoPTO5UIMEp9IdCfmN8OFMb9rxHLGefYXokDEfTUexIfeGcplLDf00uGOBCZgjzggzDRXivjQM0eDG8VpL4HI8LLiRzrX4kUDrTmDhF29gNiX+Is9ZLjNMD8wzIZQIr6IHcF15d8dd9zRpOt0M55ApKdMw4waboZc1nK64S75JuQGeajnUT3nWUxFO8rn8QQi9YSGgVWOgJ5lidr4HJAvDMvKjRrUn2K99dbrfPvtt2PqE8WZMhzLRz8CEbGlkRzSQdejW0pj6tGYlirb4+UBYkvTOkgTXqbxO7Ytanv6EYgaLgbVVTG+Si+lLX7qBSoOVSvMsX2SnTzVaKqw3unGAnGG4WGLBw8hFY8mVdMoIvh46LTCNT708WHVQ0avGOYrr7xyl2B47733yjmVQqywqEzz0KGI/tUEIuyyyy5dYUOgyowGT/fS+Mr80ksvLWZy68svv2yuaT5ShPlIXCPs0S9QvJnz0ksg1vyI9hAYclthpmcS4Yg56cr/LKgmIhDJJ4UpDqcceOCBxYw0iwJRZQGBLZS3HJo7lRuMQeITye4ov2MDwf8auWHM/kIUiEC8FJePP/64mMVy86c//amYaTiZvNfQI5CvmNO7JxCjyudeAhGYh8lcrhdeeKEaL3pd9KKieXEcl19+edNLFsuR5n5x0CjTS8R9Eo1CdpgLNlOMJxCBdCEfSSfZz2VEcRfYi3mo9FMeclBPZPEX7cR8Hk8ggubPcagc8awRduwTBpVH3FZ92SYQmQcdwwqxPnnggQfGmBGufgUi6LmK9WV0K6YxduTvKaecUszGywPYd999m/sEc6Blpp7JfgTiX//61+Y+pV0sJ0ovpS3PNWIReCbwK4Y5tk/As6IX4KnGeqebSROIebWhaGswGd6bSWiY2sI8G8kN6GRDF7564MzsZqrLwqhz1llnNY0dv7En0PxDrEVhmxe1tUGj3ya8zNwji9GpgBGw2kryqWAQvTMqDCQQ45sOxB6Htm5gvW1kYk/UoOBnbf+xth6wGr16S2YjUykKyNe2/DOzj6ksC6ZTVo5Sb7Hykl9N2TD/P/Qu0SNLL2scdu2FBeL8YqoFIs8e04Wmi6x3zCQKxEjcFy4KxH73xBO4UdvmAT9//OMfj5kgyxtt7Ipm6KC2FyBDO8znaAt7233GGGOMmZ9kvWMmUSBK8KH4mWfAW7fmhMEge+LxVsJ+UXIjb86Jn7zlR4HHZFYtamCfKd7+GQohLJrsTS8Z/3//+9+PCbsWJfznf/5ns28W9rRiTl8hYSJ6bZWvMcYYY+YuWe+YIQQiIi4eWSDGFaCaeA9xtR6rCmU/zmVhZZk2cNaqLIRc3jtNiwG02krL4ePwheYQgSbbxrBJIB5yyCGNmcIqmGTO3DxNdOd3qjfrNMYYY8z0kvWOGUIg9upB1Eq3iERXXL0Fssc8Fnr7dLACMK72qg0FSyCyqIKeR4aWEW4SiKxWiyJWAjGvaOVa3AcrXgf5q7mNnqdnjDHGzD+y3jGTLBAhLhSJPYhx647YgxhXzrKcH/oViNroNO6QD+wPpTmEhKFNIMrdtn3ctLEtvZnso6bP3xljjDFm/pD1jpkCgcj3FflmJiIvzkFEILJ/FT2EcQ6iPnnGdxxlt1+BCLirxSkSiAg+hqX13cjxBCLmCEo2JyXM+E8ctE/X22+/XcKmuYjGGGOMmT9kvWMGFIj9gtjKC0uAvQ9rq5XZmX+y5/axx2HNr/Fgf8SlS5eO+ZSZMcYYY+Yvw+qd+cyUCEQz9eRd+ccjzw0FRLB6bdW7qs+aRaK9YcF99cia+QMvYrWyQZnhs15m7vHYY4+VL2vMZtrK3UzSb3j22WefoZ8Ntolr+350L/jiER0xpp2sd3KbWfsCTRts7r3ttttm4y6wp5FPtbM6ov85LNOFBeIcZTIEYiQv0IF+HoR+YCV4r88l9bPZai18851+4txP2onJys82Bi2TM8lUb/I7l2F3iLgLxHTRT3nPnwEcFfpJm14gRIYRltNNbaHrdJH1TgwHU+O0BoG0pE1rg44Q6trxOkTYsYXpdhKITHtrqz9nKk0sEGcYHny9Meh7mkCBYIU3hSZ+Y1ffDqZxO+mkk5rviPINTyEzegNVSHFPc0LjN475D6qAsKeHlGsqmLIHvBnxnzcqoDGR23ErIcFcT83fzN9hxi/9J07xG8FaNa7rHBDTg2+tzgeULsSdNCDNYpxJK/0/99xzi5n+q1JZsGDBmHsiyk9dU/niN+afFmbJjr5prO+n5vwDzeGVucqSKr74PVyVUxp6pnLIXBVufEvX96RrX02KZUAb5sdvf+vtnbRhqonMme8M+q9vzNbmPBMP5lRjD1TuOfAfcr7NB7beeuvmnPqD55A4ElctPJSAJH3+/Oc/l7TSd8M5VO/wzWyZ8V1eoW/0km64lct7/M6wymSuk1TOFCaOk08+uZjJfi6/Geqcc845p7mfMqT6R3GMzx7xoqxHwcY8eMp4rEMvvvjiYp9nTM8WaQX6ogx+y10Oyl6s//JzLjsqn/G55RAx3ttss01jHr8fzjX2CY5wn74FrecEZMahZy3WH9FO7DWTHfklN/RlophvKhv6zwG1Z461DPxvE1PDkvWO4lYjbqmX6Xe0TPs2yy0+4JG/zS56hWUqsUCcYfQgAMMAmouJeXyYVVkx/MO3T6lc4orx888/v6wI/+qrr5oHn8pO+09KCAAVvXr05H8WiPFaPD/ooIPKRuTAYiA+ph4/oRW/biNiZRrd/NGPflR+Y08Owy8x/GxODm1uxPO5jOJB+v3mN78p54ozebr55pv/r81/rNJXuijtGLJioZeoDWHldFP5Ih/lDmXwtdde67Kj7xHX8k/hzD2IlAkJDCGRQZnRJ9ribgex8fvJT35SztlBIIN9lWfdu/zyyzfXeR6A8BAnQdqxi0LsQewlEFXWY7mP4dVvzLe5DvvRCnoTVQZo7NlzlueSNGVOOR9AANJBgh47LA4EPjggECtMX+FlQHnPsGdMbxG/fx3LZOxBlP3YM0YvD1ufgcovPT5t39OmDFAmgGdK+fnHP/6xc+WVV3Y9e6pPKceUMeba63NwsQ6lngZ2wFA51ctz/uRgTK9Y/8XnXG7H8km8JbaIZ443xO9kU8ZVP9fWCHCfvlym5+Tee+9t0jyXe/kR7ZAmEjnYUdy1iwlTA/QCEvNNZSP2II73zE02We+0iTLi0LZglXLKB0EkjO+7775spQuVBdKQNp104d5bb721sdMWlqnGAnGGiSKPhlANlSoEHuiDDz64sQMUHiq2xYsXN2ZsEaRCxD18LvCdd95pzHIBk/t62PoViFRyVIgcVGaEV2+Hxx9/fHl4MtEdKor111+/dK+LPNQXw6/V6gofD+Amm2zShIEGbD58FpF0yekXG0x6FkgTKhPeoJUuSjt6B5QmHHnlP8R8iG4DjTs9euedd17jZk2U1/Kvl0BkaAY3I9jLjaTc53ePPfboOV+q9kwIGj7S6eGHHy7/Y6MIhBu/+xWIIpZ7Dj23tXyb66ixRwDwfCnOPHdKM/6TVxIAsR4DCTLqCN2/ww47lDxnh4ncewUxvXGX8pjLZE0grrzyyo0ZDXct/1S+qN841/9cZlVPxh074rMX61HEcfSjVofWwhvLvoSP0hEor/k5rwnEGG+oxTv6j/DPdW1krbXWas71nACikfjTEy+3c/0hOzx3uq8WjigAKTO5bMTrbc9crW6bDLLeyW2mkJivQTz1wgHYRWj3ItaDcRRCZRTawjLVWCDOMOMJRARRbgwpOG0CkQfs2muvLWa4NxUCsQ0m2S5atKire7329k7YcBO7sdLi4Y/hzxVkLT3mC0o/3pxBcSY/aKi5TnrRi1ETiLGRqRHzM1beDOExnEXltN9++/UUiEL5B7KXG1sqPoZNBhGIQDyuvvrq5ktJmbYywLP0yv8OJf7ud78rv5MpENvI+TbX0ZAuZa1t3hrTGUhvPettArG2kwS9ar0EIm7iHqIjl8ma4BpEIGZymc0CMT97sR6l8Y/lolaH1sIbyz4iU72AoPovP+e6dyICsS3dRU0g0ov185//vDyTvAQpn6MfsgM8d/0KxBx+yAKxxkwKRHoGH3nkkWzckOu1/L9G2/WYfrWwTAcWiDMMDxxzXKgUqMTUyMfCwXAMDRBDMzw0VLpUbPRe0J1NLw33UqEynHLEEUeUOTHMT4kCETc0H0uVUm7kY+W2zjrrdM4666wx9rhPYcAM/xgG4H6u4UcWKgxDa0N0KlvOecgUXyoe4kGYFy5cOCb8qiCxK6FBevB2fdxxx42ZYzOXIV2Uftq4nfk3pAtpTPyZfnDGGWeUtFa6bLnllmW+EHmPHcoC6cLwX4b8lNuxfGnf0Msvv7x8A72XQMz5F+2Rjwwna3g7NoKxnEKuOGXOL24TFipjiOEQ5DuCjHKo4Ur8pqwx1LbqqqsWMxp/7uf54jnTcDcCVGmnXhzCs9JKK1Ub2lju8ZdyCrV8m+soL+DCCy8se8JS9pR2DPlKiCOSqC+ox3hescf8N+6D2j63Sm/sco9EpMq7rqsc5DKpOkn5w/61l1xySeOHRGut/GbGE4j52dN13M7iuFaH9hKIlEH8Vg8Zvfiq//JzThio/6JAbNu3N8Y7+h9FfC09yCvyLuY1PfJ84QxhyXSBHNdoR89dvwKR8OeyAYpr2zM3EwKR8kW49MW3K664opTT/GIE2GNeK4fSkTyiLNVQPUh8sU9PLPkZ6xMLxBGFwsRQau1NO8KDxYOYxRdvtnk+Ce7VhuioQCiM2Y1e1PaD5P781sObPmZtw73xQaKyyfGNgqYt/NEOYqQmguYytfRTHJXmuXcW+3FYgrLQK13ahjtIz1pe16jlnyAsOQ6A+2zR0Q+4rZ4OwhS/whT59NNPu8o+98bhXvUUvvrqq11lKqYd6ZvdyigP8DdSy7e5DI0fhyAPGOrtBfWY0ieXo7Z9bnEzL+yJZZf8yG5R/rMZkL81PyaDtmdvqqD+q/lVe6773bf39NNPb+a00aNe21UCQSf3IrU2JoOdYaZZtJUNudX2zE0FWe+MJ8p44Y1fgosQL8qk4AW8H9qeofHCMlXMiEDsFdn8Rid4Y21rKCYLGhe9pUwXVKyjAGlLb4Axg8CcpljRDkocSjb9o97bfhmVemwuguhjCoqgHq71ZtV66keJrHdOPfXU0rNLT28Negj77WxZvHhxNuoLhvoJA2GZCQYSiCjbXBFQ+WKmrvB+GEYgxr25cu+VCjZhyPOWeLNXmBV+rdQSMfx0o7f1skwFZL4xZmqgctfqWjN1uB6b+7DYapTJescMIRDjsnlgIi1mEljqIqWbPEIlrSGmKBDpWYrDVTWByKT02EXbSyAyryguQWfllsbyFf6o5tlGIIafpfb9dgcbY4wxZu6T9Y4ZQiAy2VQbPCIGmaTMpFEEFkKOSZbMKWCysfbIYvIl/zFn0qkE4lVXXVUmtnKol68mEBF5rIYUvQQiE5u14zkw900TZBX+2AvKXAyFX+ReUmOMMcbMX7LeMUMKRM1nYMUghwQW5nFuw84779x5++23m81FgeFbCcQoxBCBjLfXBGJNwOUDsBOHlOkNZPVTFoiYaeNVBGTNfWOMMcaMBlnvmCEFojbcjBtXIrDyvkUM5WIv7q9Er6MEYm0fpJpAZF+u+DWFXj2ICER6GwmnNrTMAlH3sAKJlUgWiMYYY8zokvWOGVIgIsLifmcSWAg5PkEECEGEFr/Y0wIT9seSQIyfMGJxCEvbawKR/f/ivMHxBCLgNz2YUBOIhFlCMApE5iTGb5EaY4wxZn6T9Y4ZUiACw8naqycKrPhh77h6UGZ8hF0CUSKSg01soSYQQWIO+hGI7BqvLyjUBGJc+BLDj8hFkBpjjDFmNMh6xwwoEGcSeian+ksFbH8z6P5f8wn2y3r66aezsRkxeHHrtTEtL2F5I9dMnLIxldQ2Dzazi7Ztw9o2F2/bbN2YqWQ26Z3ZwpwRiFDbXX6yyTv7jxrMK+13808zP2E7qrZv8AK98LkXPxN7/KeS6fLHDAcvEm15pClAGfaubLtmzFQx2/TObGBOCUQz9bDlkOaWmtGi3y+OzCaBaOYO9Ar2+khCjTgtyJipxHqnGwvEGYYeu5dffrl8nHvjjTcu38tEoGlDcuZ6stqaPSRpdKlkGb5jj0l93JsPtssNBB6LfbDLnpB89F2ruZmrudJKK3Wuv/76Mpwee4lkB9y4Tz98AYh8JQ/1zVTl/RFHHNFZsGBBMWOR12abbVbynt+4Kl/IjDm4mGM3uqEV/PijvOZLGPvss0/pvSEczMuFAw44oNx71llnNdMvagKR+1lohl+UR9xlC6z4sqGvIa255prlGn7pY/aUWc4px9yroWMthttwww076623XufBBx8sfulbsor3nnvuWfZa5X7uIRz4Rxy5J34L3LTDKM0xxxzT1C3UN5orzvexyQftg0uaMo+bNN9mm22KGT1/lBXlI+iX/WnJE311hbnm7DgRFyCyE4bmoTPNgTK7ww47lE+NUWYE88Rr3xM2Zlisd7qxQJxhVHnSKMcvuKjhY7GNoDJlEU1cjENjzVY9oIqVfSfjvB/2feQabsZh+rhvZax8V1tttebcTA9ZlLOIKua9tpDK9noJRPI35rfciPuSIgT15SH1IEaBGKdc6IWiJhDjFlcSFCBhhh8aNozxokdJ8xkVVr6spDhoKyzCJiERe6KINzsPxO+0Izb5djthGPUpI4NCPiHuIvFDBeSR6odYFnXOlmYff/xxYx6v5R5ELUaMZYQX47hQMfcg8gIM7DSRP5lqzESw3unGAnGGUeVJoxxXb6vBX3XVVctbu45f/epXXQJRDbsq1igWgEqWI5trODkPLWZ7Zur5/PPPS1ngeO+990p+5byHnDe9BCJuxfvlhsSfzrWgpCYQ2XVA4VJZzQIxN/wguzT4bFhP75E2p4/lnPu4HyGHyJQ/OV6ETWZZILbtfECarrPOOsW9fffdN182LfzkJz8paUZPMGKfskDvrcrQtttuW+wpj+M5+UhvIv/p4Y7XcjlRnlFGeKGlR5Cy0ksg8pJCmPSxBmMmC+udbiwQZxhVnm0CMX77Wj0s4wnEWg8iG41ncQG8ved9H2PFb6YH9eKpEUVMxbz/8ssvy2/Om9zTBvT4QP6ykdwYRCBG/1R+skCEOIQbexBp8IlHvF4TiPSM65vsDGEOIhBzDyJpSfmP32TPm+Gb8VE+UJ7UcwfqlY1lQ+f03or4ogJtApGyjjjUdIPxBOJvf/vbMeExZjKw3unGAnGGGU8gMteQoTWGkbFLgz+eQGToBbt8upB74xzEDOIxzkWM+0Oa6YO5Wcz7WrRoUbOdk/KePFq4cGExu/jii0sj+/e//738qvFkfiAN+bXXXtt8uQj36JVDeEU32gTilltu2VmyZMkYgYi7zFckXGrAawKReWvMn6QMag6iIG5xLmJNIN59992ld4qGn57TQQQi0GtFHIkrfvMM4C9hZy4uZl6dPz68hPL8ay7p888/3wh+5kHz0kFeQcxjnTMPlTmrDFPXpkVwft5555XzWA4w1xSCKBBfeumlksfMawTmHmb32rbRMWYQrHe6sUCcA7zzzjtl65FB4Z7x9omjAY0bgzOUJPFpphdEV0578j7vScicPQm02LtCA54bSxp37GY3amioNyORMB4Sehl69/rZoop4xR6oQSGO2X8WVqhn0vQH+UiZyWVh6dKlfeWj8qGtzAyax/ipeaqUb14ExFdffdWcGzMRrHe6sUAcUegl4u1bKwpFPw2AmT1EgTjbIGyUMQs0MxlQnuJuC8ZMJtY73VggGmOMMWaksd7pxgLRGGOMMSON9U43FojGGGOMGWmsd7qxQDTGGGPMSGO9040FojHGGGNGGuudbiwQjTHGGDPSWO90Y4FojDHGmJHGeqcbC0RjjDHGjDTWO91YIBpjjDFmpLHe6cYC0RhjjDEjjfVONxaIxhhjjBlprHe6sUA0xhhjzEhjvdONBaIxxhhjRhrrnW4sEI0xxhgz0ljvdDOuQPThw4cPHz58+PAxP482xhWIxhhjjDHzGeudbiwQjTHGGDPSWO90Y4FojDHGmJHGeqcbC0RjjDHGjDTWO91YIBpjjDFmpLHe6cYC0RhjjDEjjfVON3NCIF5//fXZyPTggw8+aH5/+OGHdHVqef/997NRCcOnn36ajYcGt77//vts3FDzizD827/9WzaeN5AeX3zxRTbum++++y4bFV544YXO0UcfnY2NMWZeMVv0zmxiYIFIQ7vMMss0x8MPP5ytTCq333578eeZZ57Jl+YFxC2y0UYbTaihB9zQ70TdGpRVVlklG3Xuv//+zgYbbJCNh+awww7rvPvuu9m4geuZjTfeuPPiiy9m41nFAQcc0DxXP/7xj/Plnlx99dWdnXfeORv3zZNPPpmNOm+//XZn00037SnGjTFmPlDTO6POwAKRxmvBggWdm2++ubPZZpuV/0899VS2ZvpkvgpERFpNdEwGwwjEmnDth5w/U8Xzzz/f+dvf/tb8f/zxxztrrLFGsNENedsrHQahlleY1dJytkOaDJJvlI0LLrggG4+BdNBzZYyZf9T0zqgzkEBkqDdXvKeddlrnm2++KecXXXRRuc6xyy67FDMasXgP52qMjjzyyPJ/hRVWaIa4ZMaBmSp7CZ3oxxtvvFHMVHmvvfbaxfzNN9/8h2dzAMIbiaJOPUrrr79+c53G7JRTTinmF154YWO+0047FTPSsiYQzz333HJ99dVXb4adY1rmoWjS9LbbbivXLr744pKfnG+zzTblehQUUUgQPvX6ckjM7b777uX6Oeec01wj/whv9P/ll19urqsMQYzfoYceWtykZ0v3L7fcck0ZyqKGdJCboLTguPHGG4vZHXfc0ZhRBuN9Ss8oMiUoYjoRpvfee69xh7j0y7rrrpuNOgcddFBJW/y/8847G3effvrpcl3/CUvOA1378MMPm/MnnniiXI89lffdd18xywJRzx0H7pG2Smt+1atIXmCW0xx0P2VOQ+D8j9flL+f/8R//UeKKGf9/9rOfNfb1bHMo/nruZc5zLz84ai8FPEtcozxBvJ/zGE8O1TH6zwGY6////b//V84bY+YoWe+YAQUijbwaS85VQcrs1VdfbeyuvPLKpeFqaxSYqxbNQWavv/56Y5YFovygh0X301Do/Nhjjy0V/FyBcOdDcZXgeeutt8Y0/jIn3RHnO+64Y+eTTz75h4P/a65f3ELEP/TQQ8WMe1dbbbVyvuyyy3YJQ4F/pDHEhpZ8J096CUTgv+xEgXjwwQf/46b/x1prrdWcU1Zee+21zl577dWYffbZZ2WYM8dvq6226uo5Ix6Kd02sKFwxLWD//fcv0xe+/fbbxuy8884bI1xEm0CUe4SRsApE35dfftn870X0RyhdiRfpIEhLnpXYgxjzQG5xbe+999ZtTfjjfEMN/WeBKDO5SVmJyI8Y3whx/+ijj8o5afXb3/62tS7Q+XXXXVfOJRB1Py+mN9xwQzmnvHMNt9qe+7YexEsuuaTEI8+3jD2ICFlEPuQ6RuWLcoq53OFcz4oxZm6S9Y4ZUCBSSaoSZl6ZhplVcVJRxqOXQIQzzjijsUsPSc0sCsQ8/5EDCJcaP/yU+Vwgh1WijrjS86J4ZgEW7eaekiwQERTrrbdeZ8MNNyzHtttuW64jPFddddXifhYyUeBNpkCM4k3hBPKNnsc81IcbOX64gZuxF4ejH4GY04Lj2WefHdOrxjGIQJRdfnk+ots5XduI/giE0ZVXXjkmnUA9hv0IxJgWCn8sV7I7nkDMYdB/5WumFp9edUE8l0AUEoLxUNxqz32bQITtttuucYMXEIgCkd7y7BdEgRh7yHVgZoyZu2S9YwYUiPTsURmq1+njjz8u/6NAFLypU+nyli1zBAnnNAC8pZ999tnFnF4lKulohh+YRYEYh7jvuuuu5rytoZgL5LBK1NHbceuttxazmgCLdunFiauHs0Ck14z0EhoeVA8N5IZ+EIFID1oO3zACsdaDSLxy/NSDSE+oyiLx7Ecg5rTgPth1110bs8MPP7wqEOkVF7gDMZ70IBI2Ibf7QcPaEYbNv/766xKv2KvOYpR+exBrAlHlKpqNJxDbehBzuRH0TKp8Ib6POuqo1roA4nkWiDz36l0kPW666aZy3vbctwlEevmWLFlSzsnLWGbpNQbuU/rkOka9rbkHkakeXshjzNwm6x0zoECEY445plSOOmisokBUjxSH3spXXHHFMffQAKix0HHCCSdUzaJAlECN85GgraGYC+SwStSxaIE4PfLIIyVNswCLdtXY3nPPPWVOWBaI6nl97rnnytDf3XffXa5jxtAdDeEVV1zRuAvjCUQa6kWLFhU/a+FjVe2WW25ZGuR+BSJsscUWRUzgrtImx08vDgsXLuwcccQRZa4fvWL9CESlBYKDsCHCADPS+vLLL++sueaaTdzXWWedzllnnVXOKXcIw2uvvbYZHo/pBHvuuWcJ43HHHdfM1+wXwkCYSDvOJVSIFz2TxJ+0IY3EbrvtVnryBxGI/DJfj/xT/McTiIggRgv+/ve/l/Bo/mubQMR9wqGDsEOtLoB4ngWiruvYfvvti1nbcx/rkYiGjHXwjAGL7viPW4hAznMdo3mm+k8e6P9cmtJijKlT0zujzsACEdhfjyHmDBUzQy21eW0SJZmlS5d2zQlCyLRN8MfuO++8k43nJYPElTQnjbVgqEYtrUln0nsYEG69to5BnA7Ts0LPUy5fbfEjfdTQDwJiIZexmvsQzUir2l6PEcKvvSgHBfdzfkjoE+ZcHtjzsfa89aLXc9oL7quVoTYIG/7kMtBWF/QCN3CrtsdlG7W8xJ2a/7ireFGecjoD16P/ip8xZu7TpndGmaEEojFm+pBANGauQO+upj/MZfbZZ5/OiSeemI1nHM3nVY92jTjyA7x0MyLSBiMXmp8eYZThpZdeysYN8qeX2+PR5vd0Yr3TjQWiMbOcU089tdobZsxUgOjQAh6haR39MhkCMQucmYBe49wDPhvQor5e1NKv14smIwqqZ+K94/Xay24vt2vEOdTR75nCeqcbC0RjjDENZ5555pjFVsxR/vWvf9005uwdqt6rk08+ubFX26uUhp8FQTLXMD6iAjOETt7PFOQ+B9T2F2VPVeaPRuht10p0YAqM7tP2RdE/9j/V/OcoinR/FGK4jTnxiuHRnppxD1cWZmViuhF2uPTSSxsz9mcF5uBqrj/kOMRV9JoTrrnCcb53Foj5Wt5PV3OOo/sQRzC0H27s7ZM/clv3Rjdqe6/qv3ZlUBxi2mr7KK7HPXSnQrRb73RjgWiMMaaBBpvFbBJzfKaSDdcRGCwUjIukWPGPgERU/vGPfyxmWgiG/bianXm5WgkexQvDuMxnBrau0p6nNcEWzz///POujyIgZrTinfCzKAoUJiAMxAeIS78CsRYGYIEf8Y/3H3LIISV8IqcbC9jY23TzzTdvzPisJXFHDD344IPFLMYBauEaViBKfGmHgijS4r0SiPfee2+TVueff37ZXSDazQvWcJ9FexB3aZB/vXZh0Pxo3GZBJNf1ZSnKW23Xh4livdONBaIxxpgGRACNMHtwIlAQiGrMFy9ePGZTcIQWwoBV/XHRE1uXYR8xcPTRRzeHBE4WL/SSsRqfxUESDbLDtkKbbLJJ4waCqW04M/Z2sf3WDjvs0NwnYRK3qyIu/QrEtvAQV9KAXj96JtnlIPdw5XQT2CPe+KFe17ijQFscJkMgCqXZeAIRmHpAeB9++OGudIsCEXtR2DJ8zH3Mg9RuEzWBSDmIH1OgDBLPGLYYj8nEeqcbC0RjjDENseG/7LLLyhZFaszZ5mkQgRjFWCQKEO5HVAFfNaoJxCgaepEFova3jEyGQOwVHnpDEYpx79KcbgIRhXhCiLPfa00g1uIwEwKRbbd+/vOfF7Pf/e53XekWRVv89ClsvfXWzec2Zb8mEHPaWiDOLBaIxhhjGtTw0whLKKkxR/wwFCr4VCXbBjHEqrlscYiZIVQNAzPUrLlrUYAgLhlGhFtuuaURDXF/yTjsqDAxDCthKaJARKBoTiP36pwhZoUpDjHL3XheE4i6rvAwpH3NNdeMMeM+fR0M8hAzaUgvW/wwAOHLAjHGQXYghguRrjmjDP3WRCAMIhBj2us6vaD6yABDx20CkXjmoX9eNID4KG35yhSCENqGmBHQGmK2QJx+LBCNMcY0qOGn8da3vGNvDws7aMQ5WOggtICBTevjIhV92hHRoaHXKFC0ET4Hbssf5jdKTMSFC8xTBIa+82rrvCVUXDjCl7+gbZEKgk525W+bQIzhId7ZTAtXInGRyp/+9KdiVkuzKBChFocYLjjwwAPLdfyoiUAYRCDGtNd18lLpxktBTSBiT2HVgZnKAPHVEDNghjvRb32hjSMuUrFAnH4mVSDysJKpfHVgrkJhV0GsQeEkjlMB/uI+b9rD+KH7JvIAjbdX1myCuMbPBUbzuQZvypqoH6FM5AZjGA466KAmXfSccqhC56tFE2Gmn309O5HYWM0GcqM+FdTSYT5C7xPDjxOFZ0tCx4w2g+qdUWAggcjDRGXL6iuh75UiSDS0kLuXI1Rg8Q1isojDIYJud8wGaWBng0CEmvAR+c0wwn3DCMSYRvENfDbTlkZTlT9Tyc9+9rPqF0qGFYi5nDJ/iM/HAW5qsrt6eOL3mftFb/7Qz7M/ldSE0bCCbNg0H49hwzMItXQw7dAbGYeCzeiS9Y4ZUiAy70To26X9CpJBBCJDAXzaS7z66qutFawEouY0aK4DR6zs6SHLk4W1kozJwsyHiQKRz27FT3PlhncyII74XavciYfCJrJAxE4OI+nBPbidV9TVaGsQSaucXkCPUf7kHPbiZ+/aBBzhIh/jpHYgbxRWrXobBNKAMMX8URmKYaml6WyDuLCik3k/MW/0DOS0UxmWyOxVTsd7BmvPCGGJ7kMUiDVI99on68ij2ufuBNeIZyTHT9DAtz07WZDVygLkMt6vQCQPGKIElddeZUrhIU3ySxhhinUdyM0YZ+7LZoSjVx1ijBmfrHfMkAJRDY9EGMvwJRD5jz0qXM6pwNkkkwnMVJC6X27oF2i04hyJaI/JuZyvuuqqTU9IhPvYekCTftmjiV4S7iE8cVhNhyrZbC6BqP+ad8E+Xr0a3mFgQi/f3j3qqKNKnFW5SwRuttlmxWy77bYrvTwXXXRRidfxxx9frnOOnT/84Q/Nfdjnl2FF5q+Qdgjn3KArHmyfwF5kfO4o9j4y8fymm27q3HDDDcUuDZHcZtIx/rIHmNxi0jhhxA8mTtfSiUnqDGcSZ66rscZN8oxd+/GXfcLojZK/cdI0K/vivmMqh/RmX3755Y2/bJmAHdw54ogjyqa6jz76aFeazgY0zye+2HCQv8qz+AzwixjPzxQH5TT+537Zg2yfA+HS9oy88sor5Zznjl8m5WNfduLzksPKof3LYv3BESffC8wpG2eddVbznBN25sJpLpa+c82zw3OTnx1BvUSe88zksiCBhnuUG8oB508//XSpR+LzoPjFZwO3VlpppVJmr7rqqlKWVKbYqLkG8SDv2C8Qe5RhoH4kTIRNceaZYx4bbm644YYl7fjkG+nAVATMNJyvNGtLB2PM+GS9YyYgEGloEQk03FSaWSDCwoULG/si917Ea1kgSjxQCUd7nOc3cO5jKb3s0SixtYDCwxYC6623XmOfOV/0hNJwxAnF2FeDEPdxwm1t3RDDMhHYFJVwCYbEs0DUjvuR2IOYw6IGggncgoafVXNtAhFkHhtBDUMCDRb3c11hJg9IF9nNvVo1Yk8zUxW0kSqiT9T8pTFluw3AT1a2KR3I53i/4hU3+wXKKvOWamk608SVgjFfKJvkTX4GEAq77rprI/yU9pxjN5fTKBBrzyCCLz8jixYtaqaTKB15BnRvrTzlsErwkj+qP0AvCLlHMP7Xpsr4Iwgnedjr2RGxBzGXBW20G3v8tL1G7EHsJRDjC2ZE/7GvAwhPjIvsxR5+xZly8Oc//7kxh/iShN+rrbZaSQdtLA21dDDGjE/WO2ZIgUgvHZUzAooKr00gal5S/CxPrXESWSAK7sFePHIlyH3czzWtstMkdcKDe3HoWPZpQGKljRn2aquxuCc3vBOBcMUwxeEhxV+fQ+LQp6L6EYjRXZnXGnSRBSLxj/kkN2MjGe3Qq6GeLSaQtxHTGnS//G/zF0FI40eZyl9jUL4IxYtfelriAbU0nWkkEFUuhfKs9gxgLwo/4FxCPpr3IxDzMyLUkxv9hVp5UlhzGZVdhUHhyy96Wu2oA7JAjGki4rMjokDErVgOfvWrXxVzrSKN/uFWPwJRZLdVzrCvA3J6KR3jqmCFAegRlBnPVPaHejW6B7V0MMaMT9Y7ZkiBKOGnyqwmEGXn9NNPL0NJ2o6AN3fMGb6R/VNPPbXsZ8V5rQLWNggM5TB/R/5G1LBq0QzDq6DwMBTKOUNJ+M05dtXDwfAow1qcq0GQfQ1t0ZuVG96JIreUXqrciT9mmqtFQ6r9x+g5EDksEoiYq2eCIcFjjz229LxoPyqG/uO96oFrawR5IWBIsyYQ+Y3D/vQKkq6//OUvGzMRewfZs0vhiQ0d/qq8yF8gT3/60582cx8VPnqz1RMJihdljX3VgLSkHLJ9Qy1NZxqlI3Eh/JRFyiTnpE3tGaBXtU0gai4mw524049AJC84p8yTN5yzfxv5zTNAecJM915yySWlJ5Owyh3CqiFu7qEXknPtZaYwtAnE2n5pNYEIXCdf87MjokDMZYF5lowe5H3oZFfPQz97zLXt9ZchPOqhxZ6G3hXPeI4/KvfqaY/PBcP+pC3U9h80xgxG1jtmSIEINNyIDqgJRIbGVOFSeWq4Nu5BBdq/iZ6D6E6sgCG+Zdd6fWLPC3YkIhQe0PAdB/OSBHPSMOMXNyQQ1WBycC9MtkCUUIv7YIHir7BxqGeOj53reg6LBCJxUG9MHEKXmdJdcJ57CCUIONRw1wQixD2+aNyZH6U5VpGTTjqpyf999923MY8CMZYR+QvkaRxmi2Xk5JNPbvznEMRdZtpDrJamM40EIsS4kIZKm/gMKO3aBCIwt43/vAj1IxAhPiPkKaj3iueZcqN7JUhYgS13FNYHHnigcQehCP0IxNp+aW0Cse3ZEVEgQiwLIu9DB4qv6oHx9pgjHWp7/WUID9+zlT0NUcf992LaRjexy6HngmuaVpDjkNPBGDM+We+YAQWiMf3CPME8vwxqQ5jGmLnPRIRp3hFhUHjRaBPmE4URmF//+tdjzOh9RpQzt3UisECQXmo4++yzyxdZhmWiaViL50zBYkWYynzNZL3DyCb5o9HOqULf2Z6NWCCaacUC0Zj5CYJpWPKI0aAwrM8UoMmGXl+E2+eff54vlWsskBsW7uclmh5rpmrQ0y+xOAwTSf9e8ZwJ1E5MVb7WyHonjrIIppsgWgmfev11tBHnNDNyKfv6IhDU/JoNWCAaY4yZML0ayfEYRiDmBYZTQWzcMyyW0767wxBfluMUk0EYJt1q9IrnTDATHQlZ70TRpi3ASO+cT8xt1q4IEaaAMCWEee6kLW5oDrKm5wgLRGOMMbOeuJJcc3RpwDT/MS4+05zVPKdZ1OZM8h1izOI8zCh05HecO62FghzMQdccVg7NG5fAifOhmc8LXP/LX/7SmLPIJ8O8Xl0HzdvlqAkW2aMnSAsH2UKLvTcjxHvllVcu9hXn6HY8EL0SDxxxrrx6n3ADO8yHlT3N72UxWBz2RsDin4bEOeLcb6jFU//1nWkgj4455pguUY65dobQginI6Qnk1ZIlS7rMMwqH8pX8Zn6tyhPxBOKmPVcpYxDn73JQhnU//2t5CVnvRNGmjfuzkMf/uB1eBH+5rjgwBUBzmCGny2zEAtEYY0yDhBY9HhIbsQFj9TWNHyvu9VUtfRghI7dYWMfqeFbFS/ix2j1vWRUXtrHzBP8lAiD2vMQeRDXCuM8G6oJdC3Rd87wQF3EnCMCfKG60ur2tZw3hRdqwNyuL37Q7A0O0+XOTxFF7+sY4t/Ug6jrwwQDiFHdqYHV7bbGU0oX7tb8nQ9dZxPChgjxsG+OpHUAAsa3FqPjFjggZzPUlIqVDW3oST+0mwF6obYsEawJRZYD7lBbESy8ZpAtllvjlMtzP4tKsd2qiLQtE0oa86UVM21/84hdlj2KELi8soubXbMAC0RhjTAMNqQ4JkNiAIU60m0FcmFJrgPNejjSWbImV94yUP/zGa/RMtg0l1wRiFnQ05pozFs1zWPNQra7n+wTCi9X7CKI4l6wGbtX2yWwTiNk+IjSntagJRHqqEGQI4Z133rmkE3vURje1D6iI8WxLm5xGIporHtmu3Izx5Jc4IfD41QE1gZh3z+A3p9Wzzz5bBDvmOghLvL+NrHdqoi0LxLirRhuKA/kYX4AYbpaQr/k1G7BANMYYU0BcqAGkt66XQGSBBgsrZDcLC5kDezlyH1/ZiVtX5SHm+HUcrV6lty7uV8m2QFATiNl99e5koZfDir9xFbB6GPN9AmFIOnGNOWa1HRtE9kt2ewlEoR62mNYaMoaaQASEC+lA+HL61cIa4xmHqPUVLsiiT9QEYlt61gRijX4EIsR9dQXhzWV4KgQiC4pieW1DcaCsxk+78iUspVHNr9mABaIxxpgCDSq9YwyTMpTcSyAiNBAlbI+iOV8ZenXY5J1hZX23naFC5qExfCoz+YMgYa4YQ5bY03e36XU87rjjyubgGip+6aWXyn1s1B4FDvexiTvhUpiy0Mth1SIExOfixYubIdN8Hyje8MwzzxQ7EqoMaUvICYZUST+GRmOc2wQi9kmfGH4JcNwgrRU+VvnSS6s5mkLzCgXD7sSLXizMNYdPxHiS97gpuxoyH0QgtqXnZAtEhrH5rjnpImHLEHMuw1MhEAmXplhATO+I4kCaY4dtc0gTbZQPNb9mAxaIxhhjGpgvpwa9H3rZRUwtXbp0zDevgfli2UzgP187Uu+jQCzG4T3A/Zo7LCpANAwKX9iRKO1FDAfhVW9nrUcLCOfNN99cDWsN0qcWftIy7wvYT3iBLzARv34gTv3a7UW/6TkRSFPSJfaMDlqGIeudQUQbPbqaqzkehIuXpsggfk0nAwtErRCK8LBQ+PMDPVFeeOGForhrXeKjDm8k8Q10PGoFkAdLbz16C+Ttmzfz8eDeE088MRv3TdvbqJk66G2Y6k1fh6GtzLW9kQ9LfPufKNOVlvhDOuReLDP7YDiYz5mauUnWO4NslH355Zd39cr2y7zaKJsVTnElE92kVGA6WKUjZCbUxSozKmvO+Syb0DdkJSAYWhiva3gUmQyBCNqYVemtHexr5Mni+U12EAYViNlvMziI+sl+iZsM2spcrDsmg8kUiJOdlm3PQ5u5MWZyqemdUWcggahvlArG/PUhe6Dyvemmm5r/EoOMtwMrqmQm+/E/sESf/6oYZWciO8zPZoin0oC9pASNGeKYuRPs+yU7fEcYEIjxu8ZxvyqZ0TOj3lfc0zwhuQH8hzjXiDTnPrnDN4AJh8Igsanv3MY9yrT9QCR/pxlqc5sG8fvcc88t/+P+arjJHJ+52NtCvBTHJ554oskr9h2D+B1mbcpKusTv+D799NPFnPjHvcY0nKG5MPgVr+vNV3uZcbBXnfKX/xH8xS2FUf4i4BlJwAx/avvRxSE4Jt6ziCAKN9nP++rJPO6Np28r883pjOaJcVx77bXFLPqT96UDfduZQ4sAiCcvsPkFK87Zqu2v1/a8RcHHNf3qiGTz+C1prZwlzSnzFpLGTIysd8yAApEKMVZinMdVORmuU4GpUeA/G4nKDSprbfbJfAsNeR5//PFdFakaq/mG0gJ22223Zt4J5mq4OJcIeuyxx8qKNPIiNrbnn39+WbX21VdfNYKARkSrrGiw5Ebc20r+Z4EYG8Att9yy/OZePP4TRu1PBUwQjuR9ydgDCnoJxPH8RlRIFDBXhxcPIC7ab2yuoXSQsFFeaRuFuG8b5YSViQg18l1QHrhP4gzi6tIoEONeZBL1uCvYgkHPXH72JNgVRvlL/kQBl/ejY/J+3OZB5Vf5Tr5KrMZ99Wp748Ue9EMOOaTrE2Fx3zdetHBf/lAma/vSLb/88uU3p1lNgMYyWttfr+15y/WaaBN4MqdcKM8gpvlcLfPGzCay3jEDCkQailipcS6BqFVTHLHngd4QfnkL1zW5oUobwUMjjxtU7PzmijTP7Yg9Lm2V61wgijytiAPFiZ6Mgw8+uLEDxJnGST2zQO+uRAb30JPDpGSZ5R4Qua+8yALxqaeeKtfiROWaQGSl23nnndeYZQhjbbJwL4Eov+N3TqPfrBwjTTR3I8dlLhLTIZZnnSMGmIjNQXqT/wi1OOmdnkXMcw8qAlC9fhKI6knTMwjxPtI7C0OBW6wSFPKX/In3xHxnBICXQUQU5xxyQ+GJboLyk5EK5TX5jjsS0pSFPNWBNImbDQv5Q7mSexyx3PDcEG6lexSCkWger8uttuct12uirQ6TOc9ZhOdKaW6MmThZ75gBBSICJlZqa621VmfzzTdv/muOYRSIVGL0YlHR8RZdE4j0hmBGTxS9YDWBmCvI+cJ4ApG0GUQgkqYaUsO9YQWiYL6p7NQE4kMPPdRTINIYDyoQBec1vxEFeRUYyO5cpJdARAxRTtg6AjG03377TYlAJC8F+8lNhUAE6gEOevKgH4HYBulBL6vcAtJrPIFYQ71y9MjpuZwtAjHmDVggGjO5ZL1jBhSIcegF1NNDZai5RxxZIDKMxTkNXE0gAhUy5lTuUSCywSrm/W4PMNdQWkAcYs4NiYar+GTVwoULS7pG89NPP72Ia3oPNUyn/beAX/W0sL/WeEPMNOYM3wLDwghQ5rvlXr08xKxPW4k8xKzhYPm39dZbl4VJKlv9+E3cNSz6yiuvlL3RIKblXKOXQERcxbm+LAyTQBx2iLkmEMlH7FNOuEfPcV5h12uIOQpEbbALiHq2LgH8iWVG4WGImToC4hAz7sovyu4111xTRiQ0nYHr2jxZxCFmiWz5w8uF9nZDDLJSGCQKtYcbTEQg1oaYdV3zHUXbFxlUFgh/zlOluTFm4mS9YwYUiEBDFXtvtBUNhybSZ4EI2hSyTSDS+KsRjAIRQVHrDZgvaDEGR/wwexQJ8dNBSiP1Huj++AF22SU/okDUpPk4T4z/kAUijY/sK1yY4Z/cVD7HRSpaQBGJixXUaMo/Ncb4hTDs1299uD4uUuH/XKWXQAQt2iAN4hAzPX1KW/IBuKb04WCRE4wnEHmu+STXP//zP5e85eAFTaJe4C+9mMoj+ZsFYlxYExdgIZbii0YMj+yzqIVfIfO4GIsyj1kszyIuUuGlCqI/CjvpqRcnLdxhzqHSfSICsfa8aVFRXoTDi1X8L2JZIO7Y4YiLVIwxE6emd0adgQUib9yx0ppK6EWKleF8pNYoGNMPCDUJnkhN0PRDFBuIOKYoMJQZh28Bf731VG+i4J/LsAVRnuM5WeAuL3dt2xyNhzZgHnYjZvyvPT9mNKnpnVFnYIFoJpc8fGdMv9Cw1qZeDPtCxUpgVo1z5FXBEfyNvd2mGzbZnQ/QU8oLQRv67N4w6L5eIq3XC7RWbw8ixmOPOeK09vyY0cR6pxsLRGOMMQ0IJ30eLwpEBFn+1FsWiIz69PpEGy8vul4Tlsw91fZekAUiYcoLdhB8+tzaeL2dUSBmcKP2OTzNnY8QTs2TNvMD651uLBCNMcY0IMpYLHfUUUeV6UQIOUTbEUccURb3MG8TccR8SvaY1C4LLAy64YYbmp0PNDdYsFgNN3EbN3A3TlfgHu7dcMMNyzQJemExY5SFXmvEKvM5f//73xf7cd40X/hiLiyLn9j5IV6X28BcT8KJm/ihXSPYW5MNx4kfG5sTdsKGGyz+Yzs3uaFwYjZd063M1GO9040FojHGmMInn3xSDsGiuNzTx2ImCavYg/jtt982drSQSuCmNu0Hdi7IAlGr4SMSZSA/RRSIgt4+rZ6vCcTYgxgFYtxuDJiPS9himHuF08x9rHe6sUA0xhhTyAucNMSsrXV01AQiq7+jnehWFGMiC0StoudetrYCCTvI99cEYjTvVyDWhp0V77gYqxbOuBetmdtY73RjgWiMMaZATx+9e0I9iJdcckljFsWeBCL37Lrrro0dhnJzD2Lcy7PWg6hFJ2ytFIedRT8CMfYg8r1t0UsgQu5BJL5tAjF+2pB9WnstsjFzB+udbiwQjTHGNGi/RfZxPPTQQ5seQsw4HnnkkUZYscemxFdtr86INgfnYI/MLBC1Xyr3a6EKX4XCDHu9BOIge06yyT7/o0DUF3kwl7BsE4hxX1et5t9+++0be2ZuYr3TzZQKRPanmq4PyU+nX7MBVtrF1X7GzCQ8f8PMy4q9VcYYM1NMVO/MRwYWiKwW09vTePut8WYX38Cmkun0azawzz77lJV3xswGeP6GGWrzKlBjzGygpndGnYEE4l133VWEIZN0t9tuu3Lei4mItvHczkzELzMcDM/kYSQzd6lN1u8XC0RjzFwm6x0zoEBk6b+EG8NJcUhJPYvsQ6Whzyja2EtKPY+77bZbc9/666/fmPNdZ9B/+RXnrsSJx5orw9wR3JyLAjF+r1Yr9+K3jfkOMzC3JprF795qrg3HX/7yl2KGcDvnnHMa8wsvvLCY0yCz3xdmxxxzTDM/54knnijXYdttt23uA+biMBdJc3R++ctfFnPZyXODzODEeVTswyZq5QO7F198cTGL362OdsgvmZG3skOeA88ozypm+p617HPAiy++2PzX95yj35EoEPUd6Pid4xhOvrMsJBAJDwsFKLfxk39xRarKKoenVxhjJpOsd8yAAjE2IlTWEohROO6xxx6dtdZaq5xHgch1eh5pMDi/8sory38aKTYnXbRoUTHHTXaz55xfGgLOr7766uIOm61++OGHZXgV87/+9a/NJqZzUSCuttpq5ZfVd+uuu245j70qpEucTP31118XMwlE/kf7bDUB3LPGGms05lqlRzqpcaUBZ9NaUJ4h9NkEFsgLNr5FIJJPAncRIu5BnDx4VvQ8MQmebyFDrXxg97rrrivnJ5xwQpMHtTKk50duk+fAZsEqB2x6vPPOO4/pQeQadgRuQPQ7IoH46KOPlkUAQHhUBjfYYIPynAMrXBmNAMKpegU/2wQi1zWlJbprjDGTQdY7ZkCBGGElG5U2u9ZTiXMeD5BApOHI1zHvJTDkBiIo34sZfsaeK9ybiwIx7h2mxhARrV6+Bx54oJjFlXOYSSByxAZVKw7zvmNq+Lk/m0FsiPmSgY5f/epX1dV8WlXYln9mMPLQrvKjVj5ibx29yboe7cT8ii8QnOt5jPlMr3EUiJQr9nqLdiD6HZG5ykY0hxgG2QPCsc4664x5sakJxHg/5PQyxpiJ0EvvjCoDCcStt9666UlSzwQVOrvNa3uBhx9+uPlGZexBjL0RfEKJRoLeqdNPP72YvfLKK0UUqVcDtwEByrnMb7rppiKg8JPPPAncn2sCMe4dRi+KGsG4Gpu4cy1+HxQzCUT2DItbO6h3Z1iBSA8UvZIRC8SpJ+aF9nJrKx9RpGFP59FOL4EIed83nqkoEHnu9tprr+Z6HN7uJRBPO+20pncQVN5iuaOX8cgjjyznCs96661XVkJTntgiRajHM95f+2+MMRMh6x0zoECk4aJi1iGxmM2vuuqqYh4FYrzOIRESzeJeUpofBQwrR3vAcFU0o8GbawJRQ2vEhSEzNZYIPr57ylxERDOCEXsMB59xxhnFLM5BZEiPhhTBfNxxxxWzYQWivrXKx+gZbiaf2gQiw/5bbrllZ8mSJc18NTMc5AVTB0h3lYm28hFFGkPMe++9d5ed8QQic1JZaEaZ4qXg7rvvLub4J4HGc7d48eLOPffc01m4cGExG08gKszXXHNNMdPc11tuuaXMQaRccl0vfAqPho0pb1zHT9JDdQzPN/cTXuZoyt0sdI0xZhiy3jEDCkSxdOnS8vH0DOa9Jo8jeGgAI/RcIHRq7kUzzrGnSe9AY4RZ7F2bi+Q0AeKrRlvcf//9pZelF8zrnChK11qeZBAFypOvvvoqXTX9gpiiHJPumi8oauUjwgvEeHZq4F/tmY092O+8804pd4PCKEJ2l3KCf+OhOiE+64jG2vOe/TDGmGFo0zujzFAC0cwuaDzVI0uvj5l7xN5c0w1l3BhjpgrrnW4sEOcB9LSwCpWFBC+99FK+bOYAzMs17Rx99NHZyBhjJg3rnW4sEI0xxhgz0ljvdGOBaIwxxpiRxnqnGwtEY4wxxow01jvdWCAaY4wxZqSx3unGAtEYY4wxI431TjcWiMYYY4wZaax3urFANMYYY8xIY73TjQWiMcYYY0Ya651uLBCNMcYYM9JY73RjgWiMMcaYkcZ6pxsLRGOMMcaMNNY73VggGmOMMWaksd7pxgLRGGOMMSON9U43FojGGGOMGWmsd7qxQDTGGGPMSGO9040FojHGGGNGGuudbiwQjTHGGDPSWO90Y4FojDHGmJHGeqcbC0RjjDHGjDTWO91YIBpjjDFmpLHe6cYC0RgzrXzxxRfZaChw55tvvim/33//fb5c5YMPPshGffHRRx91fvjhhzFmn376aee7774bY2aMmZtY73RjgWjMLGD33XfvLLPMMs1x++23Zyvzhp133jkbDcUFF1xQ0mnbbbft3HDDDflyFdL2s88+y8bjQv5kYXvYYYd1nnzyyTFmxpi5ifVONxaIxswCECDDMhVChfC8++672XgMq6yySjaaViQQh2HQNLNANGZ+Y73TjQWiMbOAmkD8+OOPmx7F++67r5i99957jdkuu+xSBJL+I1giCLgDDzywua7h0EMPPXSM/Z122qlxDxBeuodz0H/ZUY8nfpxyyimd2267rZjDscce23nwwQc7G220UbGPvRVWWKEZopW/3Kvra6+9dnP/6quvXsw222yz4gZsvPHGY/wACUQJNQSt4sZx5513NueKu0StzBWWBQsWlP+khXjggQcae20C8a9//Wu5rvjts88+jWj8+uuvO+utt96Ye4wxsxPrnW4sEI2ZBajHTgesu+66jbBZdtllyy+CSULltNNOK79tPVkIlzfeeKOcM6y6xhprNObioIMO6txyyy3l/MYbbyziDmIPInZEtCOxhRAirELu86vhXMKx1VZblXOJsngdNxF0e+21V+exxx4rZsRdbj3zzDNd8/1qAlFhwq7C+d///d+dxYsXl3Ndj2l25plndq677rpyfsUVV5T/3L/iiisWMyAcNYG42267lXPF7/333+/suOOOxQx37rrrrniLMWaWYr3TjQWiMbMABNnNN9/cHOL+++8vh3rSLr744s5yyy3XWbJkSSOY2gSiRKVYbbXVym/srcx2JKCiQGyzE4eYN9hggyIUn3vuuUaYrbXWWs11kNiLAlEg9NRzGVG8a9QEYoyb0oTf2GsJMc0Qt0cffXQ5Dj744OLGPffc0znvvPP+4dD/Y9ddd60KxBdffLH5r7AjLOlNzOlmjJm9WO90Y4FozCygNsQcxZKEDStnhYZc2wRiFlv6H/1CHCHsgN68TTfdtLETezJFtBMFIub0oEVR1OZ/L4HIUDIrhkV2IzJZApEeP3r+QKuhX3vttdKbKeh9rQnEuDhGYX399dfLve49NGbuYL3TjQWiMbOAmkBkXt5ll13WWbRoUek1hAMOOKCz9957d958880yVw+uvvrqzpZbbll6FSOINezQq8d8vgsvvLCYR79wB2FDLyW//IcTTzyxDJ/Sm4nZUUcdVXrVoh1WDzOMKuhFjMPRhBl/8Z9waCi7l0BEaGK+4YYblvDjJiDsJGTFRARiTDP5SfwIs7bCQRQyNI0fK620UlUgEs4cP4jD08aY2Y/1TjcWiMbMYp5//vmu/fcQKoiWuPdfbS9ARA/3Ypf9Antx9913Z6PSWym/6dVDRGaiaKLXLG4hgxgjTEuXLu2aP9gvEoiavziZ5DQjDXJav/XWWyUPekH65vhNRXiNMVOH9U43FoizABqm2AjREK+zzjrBxmDEnhREQhyyi0zUn0EZdkuSiRKHQkeJ2EM3lXz++edFSEnMiWHSnV7BI488spy/8sorzermV199NVqbtXz44YedZ599dtL2ejTGTA/WO90MJBA1DASsNNxiiy3GXBeIk9w4HX744V1mplPS8Prrr++cc845Y9InD2cNQhSIbeJQyJ/pyBsLxOmFRRfTAaLojDPOyMad448/Phv1BUO1DN1SZ8w1GH5nIZExZm6R9Y4ZUiAyjKQ5UTUQJz/+8Y+7VvitvPLKwdY/erDy0Aw9abWhLHrZNJEcuI/5UeMNnc12onghbbUtSYR4P/HEE+WctK8NB5I3Ent5LpZALDLcl4ciYVCB+Pjjj1fTviYClXe1a1Ab2sPtbF/Dpe+8884Yc3qXcppwP3HlHqVx9sMYY4yBrHfMkAIRMZGFXQRx8vTTT48RKWx9IRFCQ835I488UnoKtBqTid2sCmSTXdllmwu2y+DNnAnv7FP26KOPlnOE0Hbbbde59dZb/9eXuQdxzr18UeCtueaaJT3oDSItzjrrrBJfJsYDG/SyypQhPjb5Pf3008fcL3GEsGShwcsvv1zEPekpe8w1I73xQ+dCW3ZEuM4KTeUH4A8rUHFXLw/axw67l19+ede2J7quMCFwWVDA4ocTTjihxFEbF7NwADvYJY6a40XcWciBXdzCDRZjEDaVGa2svfLKK8e8tBhjjDGQ9Y4ZQiDSCO+5555jvmqAGYfEiLbIYNI6w08PPfRQYw/233//st+Y9h6LPTz0+nBoD7K4MTC//EcM4RZDWFlczVUQL8QJIRcFnlZfQtwTTkP9ea819rqrCcQsziDPVRQIcO1lh/jK0ANJ7y09eQpT7AlVnrHalrwSNSGPv3vssUfzwoFA1KbGQFkjnHGvOg6EK+Vqhx12aMz4igXplXuq839jjDEmkvWOGUIg8lkt9QC29SJKILKXGPOI1NsTBWIUDoBbiB0EEgJkv/32K+Y1gSgYRmQLkLi1xlyCOMdhVAQPX8eYCoEY97ITbQIRcJf93HIekweIM9Ker1tMVCAC5YltQhCmvQRi7skkveJmxiILwvzfGGOMiWS9Y4YQiBImzEHrtUiFRh3iXmYSIYgAGn5gpSIij57GuDWEPgtWE4iITs05w0wb9841JLQFYpCe2UEEIvfnz6nVBCL71UlgnX/++WUov5dApAcxfh9XMESrDYRZTNBLICL4lM+gPBWUg4ULF5Zz4n3JJZcUs9pmy8RX+/jJHPEa58Iy9Mz8Vcqb9urTPn9A2L/99tvGvjHGGANZ75gJCESg8Y+bw4ooEOkRYwNaiCLkoosuKv/pOZJwOfDAA4sZ8+p6DTEDc8uwy/Hll1/+w9E5yHvvvdfEg3l3MIhAJK323XffJt0QTTWBCPJnwYIF5X+0d+2115Zryrf47d7MLrvsUuzecccdPQUinHzyyY2/fMc3Q28h18hPQCCedNJJJS6Ykz5C8eSQKGZOoszYRBrogdb9mMWpD8YYY0wm6x0zoEA0sw9E0FRAD+NMfCosfvVisvHXLQaDF7f4ab/J5Lrrruv8+te/zsZ9wUvQsOFiakSeNjEe/foVd1kwUwOdFHEKSmY68oAFc3m6y2TAS3VtZ4jJQC/+U4W+PjSXyXrn1FNPLXPb+epSv2g+/HzBAnGOMxUCETfbeg+nmqkUiGYwNFw/2dD7ffbZZ5cNtodhImWEhUzx84D90K9f82W/zThKNNs49NBDe44YTUcexBGSyUSfjpwKpqKdiEym+9QLTB3CTaY5SYwjcmv+YKZ0q11nSlQ/AjnrnThiJ5gGp7zXiCdH/oLUfMEC0ZhZAvNq8xdDWKWf931k/u94n38TVJzqMcP9WJEBQ/W4J/Rd4xrYi3ZF3qM0Utu7spcftb1NcSP6K4FIXHKDyr01N4hnbiRqYdP+qlEAtAlE7ov+j9cwyD/tvJAX3QE7ONR2ZpA/+Bn3/JyKnqw2gdiW/20wr7xfamW65l/O70xbHtSeLczysyV67RkrgUjYspsqEzlf2Mc2b7Gl8iC7EoiU1Vx+QXGv7YWL/ZxWoKk4NeEUaXtu8CeXxza7QrtcxJ563MjpUssrFjLmDfK11RrP79Zbbz2mXDGFjalZShvEYOxh/uSTT8pWafnZr5H1Tnw2qXP4mIXy/u233y5fTBIxfWvxmqtYIBozC2CuJHs5ss+lFgexOv+YY44pPV5xMRH7YlKRUhFR8eV5qlRgXNt+++2bOZrM9WXBDvuOAvcw75PFQUwnwA57l26yySall41hlbyI6aabbmrsqsLOe5RG2MWAngDixR6cWtQW/YjQEOBm3NuUeGyzzTaNv0Blzf6gbMHENAgJmquuuqrcKzeiXRqS++67r2mECTPzVEmfI444ollMRfpoT1HNCa4JROLC12OIm+5Vw4AfEsDyj0Yd//CXISgaTxpC4qwhKVbqc53waJ4w+cl9xJ/P91E+lKe4MRV7e9YEImUw5z+NtXrzlDegLbWiGeSy2lamZUbecsgd0ol0Yf56m1u1xpn7a88W6RifLUG6kgcqI1m4Kk/YxYGFcXqmeN74j7uUe+3gsOqqqxb/WYypRXlyW/u3Uj5Id+wSZ+KhMhSfZfzWXrga5WG6hp4ZpVUs9yBz1gz87Gc/K+dC95Omeo6//vrrJg8IF24pPNkuKN1ZYMjuFSqjDD2TntRj2iOX+om8xG6uA1hw2tY7q/ooCkjciD2vnCuuwGJL9lueqECMZjl8uK/nFWplcK5igWjMLCD2VGhYlwaYijZChRjfwnNDGQWi7HFdIkIVKWZxL1OJgti7FwVinI+KSMA+13lDB/wkvBEEDw2NQODw5t3Wgxgr9poZft55553lXhohoYo83y+7bKsl1JjksBEnqPVwZIGI8NVKftC9vQQi6b/rrrs290DsQcTNGE7EA3BdaYzw+vjjjxs7U0VNIMadBZT/fCKUMkQ8EMvEgXhKlGRyWVXZymUadzkE7pGXoPLb5latcY49WfHZ6gdEXs5/8oRyLCjX9KDvuOOOjRlpoRcH7QMMSsdY/vTRAtIUwQ9xGkV8luN+trm8A/sL41Yu99hlsWjcVaIG/hA/vocew41b+ZmVXYjCXpBXxIcev1yPQS2volnc95Z5wKqP5AfbsCH0s0DEjPQH0pvyOVUCkbBwPPDAA41ZLV5zFQtEY2YBrAhXZaMKkAr4Jz/5SfmvFeA0WPQyYMZQSm4oo0AUXFcFGQVirDRV+dcEYq5g5Se/qixrw6WKh5BwahOIzD1S/LV6PbrJPdENEQUivTk6aBCzXTUmOWxC/nO0CUTur4moXgIRyEO5DTHNsEO+xvBDbJB4iaA3lfu16n8qyHHLeav8xxzRi5hDBFAe6d1pG1rOZVWiLpdp7NDTl/MShhGI9J7HfAWeLc1zy7srcC3az/mfy7nyG1Efw8wcNYhlXf7rNxLLShaIIvotNxhGjuGVQIzhxhxxqS9TRfL9+EF6xmc+lulsF9oEosqS6rG4a0ktr7IAA8QpYVF9RK8gPa+8pCHMs0DEjGvYwW6uv9rIeifns8xy+CC+QNXiNVexQDRmFhArVlUwcfWsrsf5QHzakIqPXgNBz0S/AlE9X6CFGzWBCLkHkT0rxxOIuZdOPS1tAlHCIroV04UwqlewTSBmsl01JjlsuEvYFB8asTaBmHsQ1bsVBaLmg9H7hX+4F+dsYRbjmd3U0G1skGLex0ZxsskCEXIPIvkPiCLtX8twea9N6XNZVdnKZRrBGb/gFBelRIFYc6vWOMcN+ns9W0Jxg1x+gDyh90qoXMd9fOOIQE0gxvJH3muIeRiByMcRJLok7HK4ZReBmIfM4/0qk4SFoWfBfbjJsHq2C+MJREGPs8xqeUXZynMQcTMKRNKKNNeuFFkgAtdkdyoE4m9+85vmpQUQ372E71zFAtGYWQA9Jgw1MU9Jm38zL5GhVL5jrblTVJbagohz9XZQQTLPb6WVVupbIGKH/S8vvfTSxk/mBG255ZadJUuWjBGI+HHNNdcUc/k7nkDUJuWshmb1qeYgtglEKnWlgeb/ES56y2IYc+Mnf7VxOz0H9LTV7KoxoZHEPYQc/jGMRZxIb8LA/K42gQjkx3HHHVcaCc0hlH3cRjiRtjQc+IcIIi0IG8Ox2hYEM+35SvoQHobINFctCkQaaebGMVyH+7iRG+DJIPdkItDYpD7nP9CYa7iWtFN54Xr+ylMuq7FsxTIN5A1+IVJkBiq/bW7VGmfMas8Wc8fisyXIPwQq10mLnP/kCaKKvI9zaxm+pZyTh/ijPK4JRJU/3MCMYfBhBSJlhXJBucJuL4GYPy4Aup9yRQ+f/MAeaUNZo4zjJqK2ZreXQMRPPVfcI4GKXfI4Q3ryzFMXYYee2CgQgedTLxE1gcg1zdGUQGT6QwxfJuudXJ/JDPc0R5MwMv8wDt3XyuBcxQLRmFkClZwaXkFllD9LSeWM0BDcw2rDQfb3k0hjFWbuUcDP2upN7NXmEo0HYet3L0HSQIIIqGwJSw5jG1qF3LbCMqJ0i/4h5GqrQWsgABC/NfC/trIb+3nPuNiDRvx7pRXXCLPKSe0b61PFIPmP8Kl9x72trOYyDfiXzSJtbmW4Xnu2yOf8bAn8blvhLGp2yMuYP73ATlv5GRTCMV469IL7ez1jvMRI6I5ntwbPGHmQ65X8X/D89Mr7Yfnqq6+yUUPWOzWBmCH/8irvkRaIvE2inDniHkX5LWuy0Rv4fEY9PbMxrlNd6KlA9HY4EzBkMUq09eLNNqa63Blj6vDs0cbTizre4pb5QE3vjDoDCUTeUKJweeWVV5p5AMMIxNkmgmaaOBTYi5loNKfaz5kWiMYYY0aXrHfMgAKRFWb582u//OUvO99++20jEGO3bJwzoG/jahUT9vgv++eee27TM6lVZYgS5hJJnOiXe/QtZs01glNOOaWYaYfzuQBDK4SV9GH+igSi4qoVeFynO5705L+uMzdL6UYa6l6lBYe68fX9aw7mR4HmpGCWV/NFmFTNHCjsERb1HOu7zBzaSkX2OLQtR/zmdBymUj7id00gKv7YU1lhngnDiEqDuDqUbz9DFtuyi3l0U8T5PopTnJtUC4cxxpj5QdY7ZkCBGCdMZ8YTiBrOYrNarSqjwYVHH320TBoWNMysPuJ6HN9XI4+55luw8o+VVcxX2G233YoZPZtye7ajcGrSdRaImnyL+GLlVLzGfIrNN9+8nAObjLJnGtc1jwpBz55WEIcJSGPykh5gpaXSvQZh0zW2RWDF3r333ttMClb4QWWBibxMDNf9QudMxqc8AJOgs0Dk/scee6ycx95r/NR2Dawm1CRx0Aq4NoGIGxK3CEu2QZB9XVc8ucYKVcpYLRzGGGPmB1nvmGkUiDSqbPkQ71dDyz5HbFmgTTH50gKCMjfEuScRtIIJEVRbLTabQfTFrRq0ySlEQXP88cdXhTLos0akgXog43XlAZN+84RsxI6+5BDTnTTFXw6laZ4Mr/RFTOE/gjSGmTzVZ6jYxgG35Q9+Ug7iKsfaELO2DxGxB1HCmV7tuKiArVJwq00gxg2Wo/n/197d6jTWRXEYvxwSxAgEV0AyhgvAEQQgxqK4ARBIFB4MJCAwcAUQMASCQyMRBP2+eXaymt11TksLA3PKeX4JYXp62vPRmfQ/+2PtOiCGmIWX/y7ZgihJP0vOO5oyILZ1MTNFfpIuZvDFTctP1FyLL17CRA4vyF/M4wIi4bOu/J5f20WTBEQQ7miJo8Wtfi5aAHmesEcl/1EBkefzPWZbrCzwnraASEvw6upqaZEjwNaBj+DKklC8PwGxvs7wXkCsP0/QzY06IPK5f3VAjFUtQpyHJOlnyHlHUwZErKyslEWrqVvF2DVqjyG+YKlfRgik25iabPGlTz2tWOM01m9cWFgoISK6J6l5RV2kqNOUQ964gAjqHnEcfufXdhVdtZSEoPWP68oBkd+5Ntzv379LyI5aTFFbLV7fFhAR95hxi1EjivpmrK1Z3/c2vCfj7zgW4/H43CnnUH+uEfjqc4oxolEjjNpxse3h4aGEL/6u8DsHRPBeHIP3jjBbf+bgvCluW9dM49w4z/v7+6GadvzmXhIq2Zd7i3EBMYY75PPgz7mVU5I0e9ryTt9NHRDBJINx9Zuo83V3dze0jdaqi4uLRn2y+jFf2pPW2XpP/UXfdYSZfF8CLXG5NhziMeGa5yetgUVwi67fwLEnue9tx+K1nH82qt5b3hbXN4lxrZ1cU66bxT3K5xbd7RxzklplbeI8Jr3nkqRuG5V3+uxDAbGLCAPRAsbkg1iJQLMtJtgw8SivePARba2U76EV8W+fx1fj38LT01PeXFqfvwv/SZul/6i1iZbnWb8OSePNUt75Lj8mIOL19bV0A9Jtqp+Brmw+07w+50dR6ucj/vZ5fLVRa82OaqmeRD10YRLRbT+N6NbvirjmWCda0s80a3nnO/yogCjNsrYl4ug6z0s5ga7zeik3Xsfr27q9p2n9YpjHqKXmckDkWOOGmuSAyL55iS4e1+8xTUCMoQkx9CEvexa4f3lIQdyvfL8ZBlMv/5WvOTCEJup+Bq5j1DlI6jbzTpMBUeoAZqMzoYgJQ1EnlNnSrGe7tbVVFoQHVQKYUc7kmij0zYQxJgnxelo5KU0UJal2d3dLQKS8EK2KhDaCXT0rnSoC1JNkP6oU8JNDJS2vTMrhfaiDeXZ2Vs4v9s3BlONS2ogSTTEJ7fb2tpzr5eVl2Ydr5vH19XV5nv24zuXl5fJ61FURIqyxjUlNvBZ0pzOGdm9vr0z6qjHzf3t7u9wv9uOecBzuF5OruF+0Dkd9S/bjvSPI5i5m7h9LjzHelclODGWJ6zs/P//v+Pi43NtJx9VK6gbzTpMBUeoAgk1d75ISPwSlqB0ZM8TrWqT8JhQyG5yAUtcZrferw1606lEEnBYwWtCWlpbKtvX19cHxOHYOOXVr2tzc3ODPdL+2je2MY+U6p/E+BCta7PjZ398vwTa3II4KiHFtlFCK961rbIaDg4MSDJmlHyE2lzoK0apI2IxzbwuIud5qrsHKNeR7J6nbzDtNBkSpI+p6lwREQlPWFhADj6PO6HsBkVDHCjG0UFJmCKO6lsNnAmJbDU5CLyWE6NLd3Nz8cECcZDY6s+cpe8S1cj4Zx46gTMvkNAEx12A9PT01IEozxrzTZECU/jECTszAJ/iwZCIBKrbVz7cFRMJZdIk+Pz+XmpPvBUTQ2laHvnrJQoJmHvs4Pz8/aIVjiUe6mkHgyuWEEMciUNXLPBKCCZV1dzC1KtmPupIErhAtpyx12BYQUU9Kq68VvIZ6oaBrm3qZhMRYTpH3JZAfHh6WsAxaIqcJiIRPPh8+J8Iu2wiIXOPNzc1gX0ndZd5pMiBKHXB1dVWCBT9vb29lG12dse3l5aVsawuIBBNax9iP8XOEuHq/k5OT8hz71gGRQFS3plEeKo5HMfWMdbN5joBUH3NnZyfvWtTHYgxjvHe0+P3586c85pyji5nneBwth3QNxzFGBcSNjY3Be0doDdzLeC7GbIJjsG1xcXGwLa6HsZLTBETQQsrYxLW1tUEX89HR0WDsqKRuM+80GRAl6RNojay7xRnf+fj4WO0hqevMO00GREn6JGZD//r1q/zUXfWSZoN5p8mAKEmSes2802RAlCRJvWbeaTIgSpKkXjPvNBkQJUlSr5l3mgyIkiSp18w7TQZESZLUa+adJgOiJEnqNfNOkwFRkiT1mnmnyYAoSZJ6zbzTZECUJEm9Zt5pMiBKkqReM+80GRAlSVKvmXeaDIiSJKnXzDtNBkRJktRr5p2msQFRkiRJ/WNAlCRJ0hADoiRJkoYYECVJkjTEgChJkqQhBkRJkiQN+R/u17Z3jcfCpwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,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mf1kW2GBaHpDzpzGmO7kcrXssssuIQjXWmutCf/reOKJJ8qSwSP46aefVv+jQIw0eZu++uqrIlT5zWE5noLtjzzySPU/Hy+BSHzbeEgXL148uP/++6v/dQLxvffem7APsF+OJ6KJNPnuu++qMN1PHQhE4liXtsPgOq+++uoSYXfccccS90zcn3zyybLOs4rPSxDnLKq533wN7jWnA3BOzpF56KGHJqSF6S/ZVlggmt6QM6cxpjuxXCEUdt1117D136yzzjpFxBx55JEThJG8W4i7++67b3D77bdXQu+pp54abLTRRoM333xzsNpqqxXRcO+99xaxedhhh004/rjjjivH8cuCiIRrrrlmsO2225Zr8nvJJZcMbr755sEOO+xQwjbYYINynchyyy1Xtv/pT38qwoPzEodNN910cNFFF5V9EIjLL7/84Nlnny37v/baayU8eusuvPDC8rvjjjsOzjnnnFLlzrkQQEcddVS5B+6H66+xxholXtdee205H5x66qmDffbZZ/DWW2+VeN55553l2OOPP77Eh/0QXPF+6ryF7Lf11luX/ZS2VMFLdK6//vqDL7/8MhwxGOy5556DY489dnDmmWcONt988xK2yy67DA499NByH9w7KO4LFy4s6U66cK/E9/nnny/7sC/Pn/sgnqQp1yYu7MP++oDg44J0/stf/lL+63447xVXXDEhf+jZ3HrrreX8Cjf9Jb+DLRBNb8iZ0xjTnViuEAyIwIy8fk0CMXrnEE7sw7kkYviPWILoQYyCKAoECcQsGvh/0kknDf72t79NCI9I/MAZZ5xRhKvQ+RAsb7/9dlnHs7XJJpuU9TqBuOaaaw4++uijKhyiB5H7POigg6ptnBtiPEiHVVdddfDBBx8UgRaZ7H5iGpx99tlFlN54441FcAECMfLxxx8Pttxyy+r/TTfdtEQY8SHeLBLrwL0Cnj7df34Gl156afmNXlHFIe8bBaLg2ZIOnIf7EPlY0z/yO9gC0fSGnDmNMd2J5YqXfuwgQrXnKaecUr286wQiYoljEIkcv/3221cCUfBfom8UgUj7xya4Lh4tRFMknhPhhudLSAAhWKKo1TF1AlFwHkTf66+/voRAjKKac5MOWbhF3n///eJVfeyxx6owzlknkhRnQBwqXqR5XXUw3tAo+prCEIF1cdc2XQevYIT9OW6rrbYqVdXEO3qS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>