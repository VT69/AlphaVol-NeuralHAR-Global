# Forecasting Realized Volatility in Markets: Does Sentiment Add Information Beyond the HAR Framework?

> **Assets:** Cryptocurrency (BTC) · US Equities (S&P 500) · Indian Equities (NIFTY 50)  
> **Models:** HAR · HAR-S (Sentiment-Augmented) · Neural-HAR (GRN-Corrected)

## Overview

This repository implements the full empirical pipeline for the research paper:

> *"Forecasting Realized Volatility in Markets: Does Sentiment Add Information Beyond the HAR Framework?"*

We extend the industry-standard **HAR-RV** (Heterogeneous Autoregressive model of Realized Volatility, Corsi 2009) by adding FinBERT-derived sentiment scores, microstructure signals (VPIN, OBI), and a neural correction layer (GRN). The central research question is whether NLP sentiment — from news, social, and macroeconomic sources — contains incremental information for volatility forecasting beyond the HAR baseline, and whether this effect is **regime-conditional**.

### Research Questions
1. Does FinBERT sentiment (HAR-S) produce statistically lower forecast errors than plain HAR?
2. Does the improvement vary across market stress regimes (low / medium / high GMSI)?
3. Is the improvement economically significant in a volatility-targeting strategy?

---

## Methodology

### Model Hierarchy

| Model | Description |
|:------|:------------|
| **HAR** | Corsi (2009) — RV lags at daily, weekly, monthly horizons |
| **HAR-S** | HAR + FinBERT sentiment score + sentiment surprise + microstructure |
| **Neural-HAR** | HAR prior + GRN residual corrector (PyTorch) |

### Core Equation

```
HAR:   log(RV_t) = α + β_d·log(RV_{t-1}) + β_w·log(RV̄_{t-5:t-1}) + β_m·log(RV̄_{t-22:t-1}) + ε
HAR-S: HAR + γ_1·FinBERT_{t-1} + γ_2·SentSurprise_{t-1} + δ_1·VPIN_{t-1} + δ_2·OBI²_{t-1}
```

### Estimation
- **Expanding-window OOS** — no lookahead; model re-estimated daily on all prior data
- **Loss Functions** — QLIKE (Patton 2011, robust to RV proxy noise), MSE, MAE
- **Statistical Inference** — Diebold-Mariano (HLN 1997 correction), Model Confidence Set (Hansen et al. 2011)
- **Economic Significance** — Quarter-Kelly vol-targeting backtest (5bps slippage, 5× max leverage)

---

## Repository Structure

```plaintext
AlphaVol-NeuralHAR-Global/
├── run_paper.py                   # ◄ MASTER RUNNER (Steps 0–6)
├── requirements.txt               # Python dependencies
├── data/
│   ├── raw/                       # (empty — populated by data_collection/)
│   ├── processed/                 # Feature matrices + saved forecast arrays
│   ├── paper_tables/              # CSV tables (auto-generated)
│   └── paper_figures/             # PNG figures (auto-generated)
├── data_collection/               # Data ingestion pipeline
│   ├── fetch_all.py               # Master data fetcher
│   └── sources/
│       ├── fetch_binance.py       # BTC 5-min OHLCV
│       ├── fetch_yfinance.py      # SPX, NIFTY daily OHLCV
│       ├── compute_rv.py          # RV from OHLCV (realized + GK/PK)
│       ├── fetch_news_extended.py # Multi-source news (Alpha Vantage, Guardian, Finnhub)
│       ├── fetch_sentiment.py     # GDELT, NewsAPI, Google Trends
│       ├── fetch_fred.py          # Macro (BAMLH0A0HYM2, T10Y2Y)
│       ├── fetch_fear_greed.py    # CNN Fear & Greed, Crypto F&G
│       └── fetch_deribit.py       # Deribit DVOL (BTC implied vol)
├── src/
│   ├── data_pipeline/
│   │   ├── data_loader.py         # ResearchDataLoader — aligns all sources
│   │   └── build_feature_matrix.py # Assembles per-asset parquet
│   ├── nlp/
│   │   └── sentiment_engine.py    # FinBERT pipeline (batch inference)
│   └── models/
│       ├── baseline_har.py        # HAR & HAR-S (OLS, expanding window)
│       ├── regime_har.py          # Regime-conditional HAR (low/med/high GMSI)
│       └── neural_har.py          # GRN-corrected Neural-HAR (PyTorch)
├── eval/
│   ├── dm_test.py                 # Diebold-Mariano + OOS-R² (HLN corrected)
│   └── mcs_test.py                # Model Confidence Set (Hansen et al. 2011)
├── backtest/
│   └── vol_targeting.py           # Vol-timing economic significance backtest
└── notebooks/
    └── paper_tables.py            # Generates all paper tables & figures
```

---

## Data Sources

| Data Type | Source | Frequency | Asset |
|:----------|:-------|:----------|:------|
| Price & Volume | Binance | 5-min → daily RV | BTC |
| Price & Volume | Yahoo Finance | Daily OHLCV → GK-RV | SPX, NIFTY |
| Implied Vol | Deribit DVOL | Daily | BTC |
| Implied Vol | CBOE VIX | Daily | SPX |
| Macro | FRED (BAMLH0A0HYM2, T10Y2Y) | Daily | All |
| Sentiment (NLP) | NewsAPI, Guardian, Finnhub | Event-driven → Daily | All |
| Sentiment (NLP) | GDELT 2.0 Doc API | Daily | All |
| Fear/Greed | Alternative.me | Daily | BTC |
| Fear/Greed | CNN Business | Daily | SPX |

---

## Paper Pipeline (run_paper.py)

```
Step 0  build      Build feature matrices (data_collection → data/processed)
Step 1  har        HAR & HAR-S expanding-window OOS forecasts  [Table 2–3]
Step 2  dm         Diebold-Mariano test: HAR-S vs HAR          [Table 4]
Step 3  mcs        Model Confidence Set (Hansen 2011)           [Table 5]
Step 4  regime     Regime-conditional HAR (low/med/high)        [Tables 6–7]
Step 5  backtest   Vol-targeting economic significance          [Table 8]
```

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (copy and fill in)
cp data_collection/.env.example data_collection/.env

# 3. Fetch all data (needs API keys)
cd data_collection
python fetch_all.py
cd ..

# 4. Run FinBERT sentiment engine
python src/nlp/sentiment_engine.py \
  --input data_collection/data/raw/sentiment/extended_news_headlines.parquet \
  --out_dir data/processed

# 5. Run full paper pipeline
python run_paper.py

# 6. Generate paper tables & figures
python notebooks/paper_tables.py
```

### Single-Asset / Single-Step Runs

```bash
python run_paper.py --asset btc --step har
python run_paper.py --asset spx --step dm
python run_paper.py --skip_build --step backtest
```

---

## Paper Tables (Output)

| File | Content |
|:-----|:--------|
| `table1_descriptive_stats.csv` | N, Mean, Std, Skew, Kurt, AC(1/5/22), LB-test |
| `table2_insample_coefs.csv` | HAR / HAR-S in-sample betas + HAC p-values |
| `table3_oos_metrics.csv` | QLIKE / MSE / MAE per model × asset |
| `table4_dm_tests.csv` | DM stat, p-value, OOS-R² (challenger vs HAR) |
| `table5_mcs.csv` | Model Confidence Set membership + p-values |
| `table4_regime_coefficients.csv` | β_{sentiment} by regime × asset |
| `table6_backtest.csv` | Sharpe, Ann.Return, MaxDD, Calmar, HitRate |

---

## Key References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.
- Liu, Y., Wu, J. J., & Zhang, G. (2020). FinBERT: A pre-trained financial language representation model for financial text mining. *IJCAI*, 4513–4519.