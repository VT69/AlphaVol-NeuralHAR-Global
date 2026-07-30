import os
import torch
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeadlineDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts.reset_index(drop=True)
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return str(self.texts.iloc[idx])

def process_sentiment(headlines_file, output_dir, batch_size=64):
    logger.info("1. Loading FinBERT Model and Tokenizer...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}.")

    logger.info(f"2. Loading Extended News Headlines from {headlines_file}...")
    try:
        df_news = pd.read_parquet(headlines_file)
    except FileNotFoundError:
        logger.error(f"Could not find {headlines_file}. Run the extended news fetcher first.")
        return None

    if 'text_finbert' not in df_news.columns:
        df_news['text_finbert'] = df_news.get('title', np.nan)
    else:
        # For sources that didn't populate text_finbert, fall back to title
        if 'title' in df_news.columns:
            df_news['text_finbert'] = df_news['text_finbert'].fillna(df_news['title'])

    # Drop missing texts
    df_news = df_news.dropna(subset=['text_finbert', 'date', 'asset'])
    
    # Sort for time-series integrity
    df_news = df_news.sort_values('date').copy()
    
    dataset = HeadlineDataset(df_news['text_finbert'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"3. Scoring {len(df_news)} headlines in batches of {batch_size}...")
    all_scores = []
    
    with torch.no_grad():
        for batch_num, batch_texts in enumerate(tqdm(dataloader, desc="FinBERT Scoring")):
            inputs = tokenizer(list(batch_texts), padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: 0=Positive, 1=Negative, 2=Neutral
            pos_probs = probs[:, 0].cpu().numpy()
            neg_probs = probs[:, 1].cpu().numpy()
            
            # Calculate Net Sentiment = P(Pos) - P(Neg)
            net_sentiment = pos_probs - neg_probs
            all_scores.extend(net_sentiment)

    df_news['Net_Sentiment'] = all_scores
    
    logger.info("4. Aggregating to Daily Sentiment Feature per Asset...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure date is datetime and set as index
    df_news['date'] = pd.to_datetime(df_news['date'], utc=True)
    df_news.set_index('date', inplace=True)
    
    for asset in df_news['asset'].unique():
        df_asset = df_news[df_news['asset'] == asset]
        
        # Resample to daily mean sentiment
        daily_sentiment = df_asset['Net_Sentiment'].resample('D').mean().fillna(0)
        
        df_daily = pd.DataFrame({'FinBERT_Sentiment': daily_sentiment})
        out_path = os.path.join(output_dir, f'sentiment_daily_{asset.lower()}.parquet')
        df_daily.to_parquet(out_path, engine='pyarrow')
        logger.info(f"✅ {asset}: Saved {len(df_daily)} daily sentiment scores -> {out_path}")

    return df_news

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/sentiment/extended_news_headlines.parquet')
    parser.add_argument('--out_dir', default='data/processed')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    process_sentiment(args.input, args.out_dir, args.batch_size)
