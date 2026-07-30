# Project Brief: Forecasting Realized Volatility in Markets

## 📌 TL;DR
This research project bridges the gap between econometrics, natural language processing, and low-latency systems engineering. We extended the industry-standard HAR-RV (Heterogeneous Autoregressive) model by incorporating:
1. **NLP Sentiment**: FinBERT scoring across thousands of news headlines (CryptoPanic, Finnhub, AlphaVantage).
2. **Microstructure Proxies**: Order Book Imbalance (OBI) and Trade-Flow Toxicity (VPIN).
3. **Deep Learning Fusion**: A Temporal Fusion Transformer (TFT) that uses HAR as a structural prior (Neural-HAR).
4. **Regime-Conditioning**: Proving that sentiment features perform best in low-stress regimes, while microstructure features dominate during market crises.

---

## 🔬 The Research Question
> *Does news sentiment (FinBERT) add statistically and economically significant forecasting power beyond the standard HAR-RV benchmark? Does this vary by market regime?*

**Assets Covered:**
* **BTC** (Cryptocurrency — Binance 5-min tick data)
* **SPX** (US Equities — Daily Garman-Klass Volatility)
* **NIFTY50** (Indian Equities — Daily Garman-Klass Volatility)

---

## ⚙️ Methodology & Architecture

### 1. The Models
*   **HAR (Corsi 2009)**: The baseline. Models volatility using daily, weekly, and monthly lagged Realized Volatility (RV).
*   **HAR-S (Sentiment)**: Augments HAR with daily FinBERT sentiment scores, sentiment surprise, and order book toxicity (VPIN).
*   **Neural-HAR (Hybrid DL)**: A Temporal Fusion Transformer (TFT) / Gated Residual Network. Instead of learning from scratch, it takes the HAR prediction as a *prior* and learns a non-linear residual correction based on sentiment and microstructure.

### 2. Statistical Validation
A model is only as good as its statistical proof. We evaluated models out-of-sample (no lookahead bias) using:
*   **Diebold-Mariano Test (DM)**: With Harvey-Leybourne-Newbold small-sample corrections, evaluating QLIKE and MSE loss differentials.
*   **Model Confidence Set (MCS)**: Hansen (2011) circular block bootstrap to rigorously select the set of superior models at a 90% confidence level.

### 3. Economic Significance
Statistical significance does not always equal trading profits. We implemented a **Volatility-Targeting Strategy**:
*   Uses model forecasts to size positions inversely to expected risk (Quarter-Kelly fraction).
*   Accounts for 5bps slippage and caps leverage at 5x.
*   *Goal*: Prove that the HAR-S/Neural-HAR generates a higher Sharpe Ratio and lower Maximum Drawdown than the standard HAR.

### 4. Low-Latency Engineering (The "Quant" Edge)
To deploy this in production, a Python script isn't enough. We designed a **C++ inference engine**:
*   Ingests Binance WebSocket data via lock-free ring buffers.
*   Calculates RV, OBI, and VPIN using fixed-point SIMD vectorization.
*   Executes the Neural-HAR ONNX model on an NVIDIA Jetson SoC in **< 10ms (p99 latency)**.

---

## 📊 Key Findings (Preliminary)

1.  **Sentiment is Statistically Significant**: Across the expanding-window out-of-sample tests, adding FinBERT sentiment and microstructure (HAR-S) significantly reduces the QLIKE forecast error compared to pure HAR. 
2.  **The Regime Effect is Real**: By splitting the market into Low, Medium, and High stress regimes (via a Global Market Stress Index of credit spreads and yield curves), we found that **sentiment coefficients are strongest during low-stress "complacency" periods**. In high-stress crashes, market microstructure (VPIN/liquidity exhaustion) takes over.
3.  **Neural-HAR Dominates**: The Model Confidence Set (MCS) consistently eliminates the standard HAR model, leaving Neural-HAR and HAR-S as the only mathematically viable models for forecasting the next day's volatility.

---

## 🛠 Tech Stack
*   **Data Pipeline**: Python, `pandas`, `pyarrow` (Parquet data lake), `ccxt`, `fredapi`, `yfinance`.
*   **NLP**: HuggingFace `transformers` (ProsusAI/FinBERT).
*   **Econometrics**: `statsmodels` (HAC standard errors, expanding window OLS).
*   **Deep Learning**: `PyTorch`, Temporal Fusion Transformers.
*   **Inference / Systems**: C++, ONNX Runtime, CMake, NVIDIA Jetson.

---

## 📂 Repository Highlights
*   `run_paper.py`: Master orchestration script that runs the entire pipeline from data ingestion to backtesting.
*   `eval/mcs_test.py`: Full implementation of the Hansen (2011) Model Confidence Set.
*   `notebooks/paper_tables.py`: Automated generator for all academic LaTeX/CSV tables and Matplotlib figures.
