import argparse
from utils.logger import logger
from utils.manifest import ManifestGenerator
from sources import fetch_binance, compute_rv, fetch_yfinance, fetch_deribit, compute_microstructure, fetch_fred, fetch_sentiment, fetch_fear_greed

def main():
    parser = argparse.ArgumentParser(description="AlphaVol Data Collection Pipeline")
    parser.add_argument('--source', type=str, help='Specific source to run')
    parser.add_argument('--asset', type=str, help='Specific asset to run')
    parser.add_argument('--from', dest='from_date', type=str, help='From date')
    parser.add_argument('--to', dest='to_date', type=str, help='To date')
    args = parser.parse_args()

    sources_to_run = {
        'binance': fetch_binance.run,
        'yfinance': fetch_yfinance.run,
        'rv': compute_rv.run,
        'deribit': fetch_deribit.run,
        'microstructure': compute_microstructure.run,
        'fred': fetch_fred.run,
        'sentiment': fetch_sentiment.run,
        'fear_greed': fetch_fear_greed.run,
    }

    if args.source:
        if args.source == 'manifest':
            pass
        elif args.source in sources_to_run:
            logger.info(f"Running isolated source: {args.source}")
            sources_to_run[args.source]()
        else:
            logger.error(f"Unknown source: {args.source}")
            return
    else:
        logger.info("Running full pipeline...")
        for name, func in sources_to_run.items():
            logger.info(f"--- Running {name} ---")
            func()

    if not args.source or args.source == 'manifest':
        logger.info("Generating manifest...")
        manifest = ManifestGenerator()
        
        files = [
            ("data/raw/ohlcv/btc_5min.parquet", "Binance", "BTC/USDT", "5min"),
            ("data/raw/ohlcv/eth_5min.parquet", "Binance", "ETH/USDT", "5min"),
            ("data/raw/ohlcv/btc_1h.parquet", "Binance", "BTC/USDT", "1h"),
            ("data/raw/ohlcv/btc_1d.parquet", "Binance", "BTC/USDT", "1d"),
            ("data/raw/ohlcv/eth_1d.parquet", "Binance", "ETH/USDT", "1d"),
            ("data/raw/ohlcv/spx_1d.parquet", "YFinance", "^GSPC", "1d"),
            ("data/raw/ohlcv/nifty_1d.parquet", "YFinance", "^NSEI", "1d"),
            ("data/raw/realized_vol/btc_rv_daily.parquet", "Compute", "BTC/USDT", "1d"),
            ("data/raw/options/btc_deribit_dvol.parquet", "Deribit", "BTC", "1d"),
            ("data/raw/microstructure/btc_obi_daily.parquet", "Compute", "BTC/USDT", "1d"),
            ("data/raw/microstructure/btc_vpin_daily.parquet", "Compute", "BTC/USDT", "1d"),
            ("data/raw/sentiment/newsapi_headlines.parquet", "NewsAPI", "Global", "Event"),
            ("data/raw/sentiment/gdelt_daily.parquet", "GDELT", "Global", "1d"),
            ("data/raw/sentiment/google_trends_weekly.parquet", "Google Trends", "Global", "1w"),
            ("data/raw/macro/fred_series.parquet", "FRED", "Macro", "1d"),
            ("data/raw/macro/fear_greed_index.parquet", "CNN", "SPX", "1d"),
            ("data/raw/macro/crypto_fear_greed.parquet", "Alternative.me", "BTC", "1d"),
            ("data/raw/options/spx_vix_term_structure.parquet", "CBOE", "SPX", "1d")
        ]
        
        for f, s, a, freq in files:
            manifest.add_entry(f, s, a, freq)
            
        manifest.write()

if __name__ == "__main__":
    main()
