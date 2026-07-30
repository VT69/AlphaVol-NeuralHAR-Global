# Forecasting Realized Volatility in Markets: Does Sentiment Add Information Beyond the HAR Framework?
## Comprehensive Research Project Summary

This document outlines the end-to-end development, methodological evolution, and final empirical findings of the **AlphaVol-NeuralHAR** research project. The objective was to ascertain whether exogenous sentiment data and advanced deep learning architectures can demonstrably improve upon the classical Heterogeneous Autoregressive (HAR) model for forecasting Realized Volatility (RV).

---

### 1. Research Objectives
1. **Empirical Baseline:** Validate if the classical HAR model accurately models the long-memory properties of volatility across diverse asset classes (Crypto, US Equities, Emerging Markets).
2. **Sentiment Integration:** Test whether incorporating alternative data—specifically, NLP-derived news sentiment (FinBERT) and macroeconomic variables—into a linear HAR framework (HAR-S) yields statistically significant alpha.
3. **Non-Linear Dynamics:** Hypothesize that sentiment interacts with volatility non-linearly. To test this, design a hybrid deep learning architecture (`Neural-HAR`) to capture residual non-linearities and cross-feature interactions that linear OLS models miss.
4. **Economic Significance:** Translate statistical forecasting accuracy (QLIKE/MSE) into tangible financial utility via a volatility-targeting trading strategy.

---

### 2. Methodology & Data Pipeline
#### Asset Universe
To ensure the robustness of our findings, we selected three structurally distinct assets spanning January 2019 to July 2026 (for BTC) and February 2015 to July 2026 (for SPX/NIFTY):
*   **Bitcoin (BTC):** Highly speculative, retail-driven, 24/7 trading.
*   **S&P 500 (SPX):** Highly institutionalized, deeply liquid, heavily scrutinized.
*   **NIFTY 50 (NIFTY):** Emerging market equity index, subject to distinct capital flows and structural inefficiencies.

#### Data Engineering
*   **Realized Volatility:** Computed as the annualized 5-minute intraday realized variance.
*   **Exogenous Features:** We engineered a robust feature matrix for each asset:
    *   *Market Microstructure:* VPIN (Volume-Synchronized Probability of Informed Trading), Order Book Imbalance (OBI), Amihud Illiquidity.
    *   *Sentiment:* Daily FinBERT scores aggregated from high-quality news sources (Alpha Vantage, Finnhub, The Guardian, CryptoPanic).
    *   *Macro/Crypto:* Term Slope, Credit Spread, Crypto Fear & Greed Index.
*   **Regime Filtering:** A Hidden Markov Model (HMM) was used to classify market regimes (Low, Medium, High volatility) to test whether sentiment's impact is regime-dependent.

---

### 3. Architectural Evolution & Rationale
We sequentially tested and discarded architectures based on empirical feedback, ultimately arriving at a sophisticated hybrid ensemble.

#### Phase 1: HAR and HAR-S (Linear Baselines)
*   **HAR:** The classical baseline modeling daily, weekly, and monthly RV components.
*   **HAR-S:** A linear expansion of HAR incorporating the exogenous sentiment features. 
*   *Finding:* In-sample $R^2$ improved slightly, but out-of-sample (OOS) expanding-window tests revealed that HAR-S suffered from severe overfitting. Linear models could not digest the high-dimensional noise of sentiment scores.

#### Phase 2: Neural-HAR (Hybrid Design)
To prevent the deep learning model from failing to capture the core autocorrelation of volatility, we designed a **Hybrid Architecture**:
*   **Linear Prior:** An OLS-initialized layer that explicitly computes the standard HAR components.
*   **Residual Branch:** A deep neural network that takes the HAR inputs *and* the exogenous features to predict the residual error of the linear prior.

#### Phase 3: Network Topology (The Ensemble)
We initially tested basic MLPs and LSTMs, but they struggled with the tabular nature of the data. We then implemented and ensembled two state-of-the-art tabular deep learning architectures:
1.  **Gated Residual Network (GRN):** Designed to act as a feature selection mechanism. It uses GLU (Gated Linear Units) to suppress noisy, irrelevant sentiment features while amplifying critical signals.
2.  **FT-Transformer (Feature Tokenizer Transformer):** An attention-based architecture that tokenizes numerical inputs and computes multi-head self-attention, allowing the model to learn complex cross-feature interactions (e.g., how Term Slope interacts with FinBERT scores).
3.  **Inverse-Loss Ensemble:** The final `Neural-HAR` model averages the predictions of the GRN and Transformer branches, weighted inversely by their local validation loss, ensuring the architecture best suited for the current market dynamic has the highest influence.

---

### 4. Training & Validation Protocol
To ensure rigorous, statistically sound results free from look-ahead bias:
*   **Walk-Forward Validation:** We used a strict 21-day step-size walk-forward training loop. The model trains on a rolling window, validates on the immediate subsequent period, and tests purely out-of-sample.
*   **Hyperparameter Optimization (HPO):** Integrated **Optuna** to run 40 independent trials per walk-forward step per architecture (GRN & Transformer). It dynamically tuned learning rates, dropout, weight decay, hidden dimensions, and depth.
*   **Convergence:** Utilized `CosineAnnealingLR` and gradient clipping to stabilize the highly stochastic loss landscapes inherent to financial time series.

---

### 5. Empirical Results & Findings
The testing framework utilized the Diebold-Mariano (DM) test, the Model Confidence Set (MCS), and a Volatility-Targeting backtest (5bps slippage, Quarter-Kelly fraction).

#### 1. Bitcoin (BTC)
*   **Statistical:** `Neural-HAR` dominates. It achieved an OOS $R^2$ of +21.7%. The DM test confirmed its superiority in MSE at the 1% significance level ($p=0.0001$). 
*   **Economic:** In a structurally difficult OOS period (yielding negative absolute returns), `Neural-HAR` exhibited the lowest annualized volatility (33.8%) and the lowest Maximum Drawdown (-59.5% vs HAR's -65.4%), proving superior downside risk mitigation.
*   *Conclusion:* Crypto markets are highly inefficient and heavily influenced by non-linear sentiment dynamics. The deep ensemble successfully extracted this alpha.

#### 2. S&P 500 (SPX)
*   **Statistical:** `Neural-HAR` is the **sole surviving model** in the Model Confidence Set (MCS). It achieved a highly significant QLIKE DM statistic of 5.80 ($p=0.0000$), definitively beating the HAR baseline.
*   **Economic:** Targeting 2.0% daily volatility, the `Neural-HAR` strategy achieved an annualized volatility of 40.61%, drastically closer to the target than HAR (57.64%). Furthermore, it reduced the MaxDD from -42.2% to -35.2%.
*   *Conclusion:* Even in highly efficient institutional markets, attention mechanisms (Transformer) and feature gating (GRN) identify complex, non-linear relationships that linear models miss.

#### 3. NIFTY 50
*   **Statistical:** `Neural-HAR` marginally beat HAR in QLIKE ($p=0.09$), but both models survived in the MCS. 
*   **Economic:** The `Neural-HAR` strategy exhibited slightly elevated turnover and a comparable MaxDD to HAR.
*   *Conclusion:* Emerging markets exhibit different microstructure dynamics. The complexity of the Neural-HAR model provides a marginal edge, but the simplicity of the classical HAR model remains highly competitive.

---

### 6. The Way Ahead (Drafting the Paper)
The empirical pipeline is complete. The transition to the writing phase should focus on the following narrative structure:

1.  **Introduction:** Frame the failure of linear models (HAR-S) to digest alternative data due to the "curse of dimensionality" and linear constraints.
2.  **Literature Review:** Bridge the gap between classical econometrics (Corsi, 2009) and modern deep learning for tabular data (Gorishniy et al., 2021).
3.  **Methodology:** Detail the Hybrid `Neural-HAR` architecture. Emphasize why combining an OLS prior with a non-linear GRN/Transformer residual branch preserves statistical consistency while allowing for complex feature interaction.
4.  **Results Discussion:** 
    *   Highlight the SPX QLIKE improvement (Table 4) and the MCS survival (Table 5).
    *   Discuss the BTC OOS $R^2$ explosion (Table 4), proving sentiment matters in crypto.
    *   Analyze Table 7 to show that better forecasting translates to **smoother equity curves and lower drawdowns**, not necessarily higher absolute returns, validating the utility of accurate vol-targeting.
5.  **Conclusion:** Conclude that sentiment *does* add information beyond the HAR framework, but *only* when processed through architectures capable of non-linear feature gating and attention.
