# AlphaVol: Neural-HAR with Microstructure & Sentiment Features

## Overview
A high-performance quantitative volatility forecasting engine. The project extends the industry-standard HAR-RV (Heterogeneous Autoregressive model of Realized Volatility) by combining it with a Temporal Fusion Transformer (TFT). It leverages Limit Order Book (LOB) microstructure features, NLP-based sentiment shocks, and macroeconomic regime conditioning. 

Originally targeted at cryptocurrency markets, this repository has been generalized to forecast realized volatility across multiple asset classes: **Crypto (BTC), US Equities (S&P 500), and Indian Equities (NIFTY 50)**.

## Architecture

This project implements a Hybrid-Inference Methodology consisting of three pillars:
1. **Persistence:** Baseline memory modeled via HAR lags ($RV_d$, $RV_w$, $RV_m$).
2. **Liquidity Pressure:** Microstructure features including Volume-Synchronized Probability of Informed Trading (VPIN) and Order Book Imbalance (OBI).
3. **Information Shocks:** Contextual awareness through NLP FinBERT sentiment scores and macroeconomic stress indexes (e.g., VIX, Deribit DVOL).

### Neural-HAR (TFT + HAR Prior)
Instead of relying solely on pure deep learning, this architecture uses the classic HAR model as a structural prior. A Temporal Fusion Transformer (TFT) with a Gated Residual Network (GRN) learns the non-linear residuals dynamically based on the prevailing market regime. This captures complex market dynamics while remaining statistically robust against overfitting.

### Low-Latency Edge Inference
The model is engineered for production-grade speed:
- **C++ Inference Engine:** Targets a sub-10ms p99 latency for the forward pass.
- **Hardware Target:** Optimized for edge deployment on architectures like NVIDIA Jetson Orin (ARM SIMD).
- **Position Sizing:** Executes a real-time, Kelly-adjusted volatility targeting strategy directly from the inference output.

## Repository Structure

```plaintext
AlphaVol-NeuralHAR-Global/
├── data/                       # Data Lake
│   ├── raw/                    # Parquet files for Tick/LOB data
│   └── processed/              # Cleaned features (RV, OBI, Sentiment)
├── src/
│   ├── data_pipeline/          # Engineering: The "Plumbing"
│   │   ├── ws_ingestion.py     # High-speed WebSocket ingestion (Binance/Equity APIs)
│   │   └── processor.cpp       # C++ L2 Order Book aggregator
│   ├── nlp/                    # The "Context"
│   │   └── sentiment_engine.py # FinBERT pipeline (quantized for speed)
│   ├── models/                 # The "Brain"
│   │   ├── neural_har.py       # Hybrid TFT/GRN Model
│   │   └── baseline_har.py     # Classical Corsi (2009) OLS benchmark
│   └── cpp_inference/          # The "Engine"
│       ├── tensorrt_deploy.cpp # C++/TensorRT inference logic
│       └── CMakeLists.txt      # Build configs for ARM/Jetson
├── backtest/                   # Strategy & Risk Management
│   ├── vol_targeting.py        # Volatility-scaled position sizing
│   └── tc_optimizer.py         # Transaction cost/Slippage modeling
├── eval/                       # Statistical Proof
│   ├── dm_test.py              # Diebold-Mariano implementation
│   └── mcs_test.py             # Model Confidence Set calculation
└── notebooks/                  # EDA and Paper Drafting
```

## Data Sources

| Data Type | Source | Frequency | Purpose |
| :--- | :--- | :--- | :--- |
| **Price & Volume** | Binance / Market Data Providers | 5-minute | Compute Realized Volatility ($RV$) |
| **Microstructure** | Crypto L2 / Equity L2 | Real-time | Compute Order Book Imbalance (OBI) & VPIN |
| **News/Social** | CryptoPanic / NewsAPI | Event-driven | Compute FinBERT sentiment scores |
| **Macro Stress** | VIX / DXY / DVOL | Daily | Regime-conditioning (Low/High stress) |

## Core Methodology & Math

1. **Microstructure Integration:** Computes toxicity (VPIN) and imbalance (OBI) using lock-free ring buffers in C++ to avoid standard Python overhead.
2. **Diebold-Mariano (DM) Validation:** Employs formal statistical tests to rigorously demonstrate the alpha value of the hybrid model against standard OLS HAR benchmarks.
3. **Volatility Timing Strategy:** Outputs prediction uncertainty quantiles ($q_{10}, q_{50}, q_{90}$) to scale sizing and optimize the Sharpe Ratio.

## Setup & Deployment

*(Further instructions to be added as individual components are implemented.)*

## Original Research & Methodology

This repository represents an original research architecture fusing three distinct domains:
1. **Classical Econometrics:** Leveraging the Heterogeneous Autoregressive (HAR) framework for baseline volatility persistence.
2. **High-Frequency Microstructure:** Incorporating Order Book Imbalance (OBI) and Volume-Synchronized Probability of Informed Trading (VPIN) to capture real-time liquidity shocks.
3. **Deep Learning & NLP:** Fusing FinBERT sentiment with a Temporal Fusion Transformer (TFT) to dynamically learn non-linear, regime-conditional responses.

By unifying these signals into a novel `Neural-HAR` model and deploying it via a custom sub-10ms C++ inference engine, AlphaVol delivers a state-of-art solution for real-time volatility prediction and dynamic position sizing.