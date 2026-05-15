# AlphaVol Data Pipeline

This directory contains the production-grade data collection pipeline for AlphaVol-NeuralHAR-Global.

## Execution
Run the orchestrator:
```bash
python fetch_all.py
```
Or selectively:
```bash
python fetch_all.py --source binance
python fetch_all.py --asset btc
python fetch_all.py --from 2022-01-01
```

## Structure
All data is stored in `data/raw/` in `.parquet` format using PyArrow.
- **OHLCV**: 5m, 1h, 1d price data from Binance and YFinance.
- **Realized Volatility**: Bipower variation, Parkinson, Garman-Klass, and HAR lags computed from 5m data.
- **Microstructure**: VPIN, Order Book Imbalance proxy, roll spreads.
- **Options/VRP**: DVOL, Options chain snapshots.
- **Sentiment**: FinBERT-ready texts from NewsAPI, CryptoPanic, and GDELT aggregate sentiment.
- **Macro**: FRED indicators, Fear & Greed indices.

See `manifest.json` for details on rows, coverage, and missingness for each generated file.

## Detailed Dataset Report

| File Name (`.parquet`) | Data Type & Purpose | Source Details |
| :--- | :--- | :--- |
| **`btc_5min`** | High-frequency 5-minute OHLCV to compute intraday volatility estimators. | Binance REST API (`/api/v3/klines`) |
| **`eth_5min`** | High-frequency 5-minute OHLCV to compute intraday volatility estimators. | Binance REST API (`/api/v3/klines`) |
| **`btc_1h`** | Hourly OHLCV data for medium-frequency trend tracking. | Binance REST API (`/api/v3/klines`) |
| **`btc_1d`** | Daily OHLCV data and Log Returns. | Binance REST API (`/api/v3/klines`) |
| **`eth_1d`** | Daily OHLCV data and Log Returns. | Binance REST API (`/api/v3/klines`) |
| **`spx_1d`** | S&P 500 Daily OHLCV (used as a TradFi baseline). | Yahoo Finance (`yfinance` package, `^GSPC`) |
| **`nifty_1d`** | NIFTY 50 Daily OHLCV (used as an emerging market TradFi baseline). | Yahoo Finance (`yfinance` package, `^NSEI`) |
| **`btc_rv_daily`** | Econometric targets: Realized Volatility, Bipower Variation, Jumps, and HAR Lags (Daily, Weekly, Monthly). | Computed internally from `btc_5min.parquet` |
| **`btc_deribit_dvol`** | The Deribit Implied Volatility Index (DVOL), the crypto equivalent of the VIX. | Deribit API (`/get_volatility_index_data`) |
| **`btc_obi_daily`** | Market Microstructure features: Corwin-Schultz Order Book Imbalance, Roll Spread, and Amihud Illiquidity. | Computed internally from `btc_5min.parquet` |
| **`btc_vpin_daily`** | Volume-Synchronized Probability of Informed Trading (VPIN) based on Easley et al. | Computed internally from `btc_agg_trades` |
| **`cryptopanic_headlines`** | Crypto-specific news headlines and community voting metrics (Bullish/Bearish/Important). | CryptoPanic API (`/api/v1/posts/`) |
| **`newsapi_headlines`** | Global macroeconomic news headlines covering stock crashes and broad market sentiment. | NewsAPI (`/v2/everything`) |
| **`gdelt_daily`** | Global database of media sentiment, parsing global news for "Crisis" and "Economic" taxonomy tags. | Google Cloud BigQuery (`gdelt-bq.gdeltv2.gkg`) |
| **`google_trends_weekly`** | Search volume for terms like "bitcoin crash" and "recession" over the last 5 years. | Google Trends (`pytrends` package) |
| **`fred_series`** | U.S. Macro Indicators: TED Spread, High Yield Spread, T10Y2Y Curve, and Fed Funds Rate. | Federal Reserve Economic Data (`fredapi`) |
| **`fear_greed_index`** | CNN Fear & Greed Index (Equities) mapping market psychology. | CNN Data API (Scraped endpoint) |
| **`crypto_fear_greed`** | Crypto Fear & Greed Index measuring digital asset psychology. | Alternative.me API (`/fng/`) |
| **`spx_vix_term_structure`** | 9-Day, 30-Day, and 3-Month VIX metrics to compute contango and volatility term structure. | CBOE Global Markets CSV Export |
