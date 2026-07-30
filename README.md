# Forecasting Realized Volatility in Markets: Does Sentiment Add Information Beyond the HAR Framework?

> **Assets:** Cryptocurrency (BTC) · US Equities (S&P 500) · Indian Equities (NIFTY 50)  
> **Models:** HAR · HAR-S (Sentiment-Augmented) · Neural-HAR (Hybrid GRN + Transformer Ensemble)

## Overview

This repository implements the full empirical pipeline for the research paper:

> *"Forecasting Realized Volatility in Markets: Does Sentiment Add Information Beyond the HAR Framework?"*

We extend the industry-standard **HAR-RV** (Heterogeneous Autoregressive model of Realized Volatility, Corsi 2009) by adding FinBERT-derived sentiment scores, macroeconomic signals, and a sophisticated non-linear neural correction ensemble layer (combining a Gated Residual Network and an FT-Transformer). The central research question is whether NLP sentiment contains incremental, actionable information for volatility forecasting beyond the linear HAR baseline.

### Key Empirical Results
1. **Statistical Superiority:** The `Neural-HAR` ensemble overwhelmingly outperforms the baseline `HAR` model. For **Bitcoin**, it achieves a massive Out-of-Sample $R^2$ of **+21.7%** and significantly lowers Mean Squared Error ($p=0.0001$). For the **S&P 500**, it is the sole surviving model in the Model Confidence Set (MCS), achieving a highly significant Diebold-Mariano QLIKE statistic ($p=0.0000$).
2. **Non-Linear Sentiment Dynamics:** Linear models (HAR-S) fail to extract out-of-sample alpha from high-dimensional sentiment scores due to the curse of dimensionality. Deep learning architectures (GRN + Transformer) successfully isolate and extract complex, non-linear relationships.
3. **Economic Utility (Vol-Targeting):** In a volatility-targeting strategy (Quarter-Kelly, 5bps slippage), the `Neural-HAR` model's accurate forecasts mitigate risk better than the baseline, producing the lowest Maximum Drawdowns and keeping realized portfolio volatility much closer to the target mandate.

---

## Methodology

### Model Hierarchy

| Model | Description |
|:------|:------------|
| **HAR** | Corsi (2009) — OLS regression using daily, weekly, monthly RV lags |
| **HAR-S** | HAR + FinBERT sentiment scores + Macroeconomic variables |
| **Neural-HAR (Hybrid)** | HAR linear prior + Deep Learning Residual Predictor Ensemble |

### The Neural-HAR Ensemble
To prevent the deep neural network from failing to learn core autocorrelation, we designed a hybrid architecture. An OLS-initialized linear layer computes the classical HAR prediction. A parallel neural network branch takes both HAR and Sentiment features to predict the **residual error** of the linear prior. 

The residual branch is an inverse-validation-loss weighted ensemble of two state-of-the-art tabular models:
*   **Gated Residual Network (GRN):** Acts as an endogenous feature selector, using GLU to suppress noisy, irrelevant sentiment signals.
*   **FT-Transformer:** An attention-based architecture that tokenizes features and learns complex cross-feature interactions (e.g., how the term slope interacts dynamically with FinBERT sentiment).

### Training Protocol
*   **Walk-Forward Validation:** Strict 21-day step-size expanding window to explicitly prevent look-ahead bias.
*   **Hyperparameter Optimization:** Powered by Optuna; 40 dynamic trials (tuning LR, depth, dropout, weight decay) per walk-forward step.
*   **Optimization Stabilization:** Incorporates `CosineAnnealingLR` and gradient clipping to navigate the highly stochastic financial loss landscapes.

---

## Repository Structure

```plaintext
AlphaVol-NeuralHAR-Global/
├── run_paper.py                   # ◄ MASTER RUNNER 
├── requirements.txt               # Python dependencies
├── data/
│   ├── raw/                       # Raw OHLCV and news CSVs
│   ├── processed/                 # Feature matrices + forecasts (.npy)
│   ├── paper_tables/              # CSV tables (auto-generated)
│   └── paper_figures/             # PNG figures (auto-generated)
├── data_collection/               # Data ingestion pipeline
├── src/
│   ├── data_pipeline/             # Aligns asynchronous data sources
│   ├── nlp/                       # FinBERT pipeline
│   └── models/                    
│       ├── baseline_har.py        # HAR & HAR-S models
│       ├── neural_har.py          # PyTorch GRN & Transformer definitions
│       └── train_neural_har.py    # Walk-forward Optuna ensemble logic
├── eval/
│   ├── dm_test.py                 # Diebold-Mariano test (HLN corrected)
│   └── mcs_test.py                # Model Confidence Set
├── backtest/
│   └── vol_targeting.py           # Economic significance backtest
└── notebooks/
    └── paper_tables.py            # Aggregates results into LaTeX-ready tables
```

---

## How to Reproduce Results

To fully replicate the exact tables and figures submitted in the research paper from end to end:

### 1. Environment Setup
```bash
# Clone the repository and install requirements
git clone https://github.com/VT69/AlphaVol-NeuralHAR-Global.git
cd AlphaVol-NeuralHAR-Global
pip install -r requirements.txt
pip install optuna
```

### 2. Run the Full Paper Pipeline
The `run_paper.py` orchestrator controls the execution of all methodological steps. 
Run the following commands sequentially:

```bash
# Step 1: Train the Baseline HAR and HAR-S models
python run_paper.py --skip_build --step har

# Step 2: Run the Neural-HAR Ensemble Pipeline (Optuna + Walk-Forward)
# Note: This is computationally intensive. Ensure a GPU (CUDA) is available.
python run_paper.py --skip_build --step neural

# Step 3: Run the Diebold-Mariano tests (Statistical Significance)
python run_paper.py --skip_build --step dm

# Step 4: Compute the Model Confidence Set (MCS)
python run_paper.py --skip_build --step mcs

# Step 5: Execute the Volatility Targeting Strategy Backtest (Economic Significance)
python run_paper.py --skip_build --step backtest

# Step 6: Generate LaTeX-Ready Paper Tables and Figures
python notebooks/paper_tables.py
```

### 3. Locating the Output
Once the pipeline is complete, all outputs are systematically exported:
*   **Tables:** `data/paper_tables/table1_*.csv` through `table6_*.csv`
*   **Figures:** `data/paper_figures/fig1_*.png` and `fig2_*.png`

---

## Key References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.
- Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS*.
- Liu, Y., Wu, J. J., & Zhang, G. (2020). FinBERT: A pre-trained financial language representation model for financial text mining. *IJCAI*, 4513–4519.