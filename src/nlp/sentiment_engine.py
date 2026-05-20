import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from torch.utils.data import DataLoader, Dataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeadlineDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return str(self.texts.iloc[idx])

def process_sentiment(headlines_file, output_file, batch_size=32):
    logger.info("1. Loading FinBERT Model and Tokenizer...")
    # Using the standard financial BERT model
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move to GPU if available in Colab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}.")

    logger.info("2. Loading GDELT Headlines...")
    try:
        df_news = pd.read_parquet(headlines_file)
    except FileNotFoundError:
        logger.error(f"Could not find {headlines_file}. Run the GDELT ingestion script first.")
        return None

    # Clean missing titles
    df_news = df_news.dropna(subset=['title'])
    
    dataset = HeadlineDataset(df_news['title'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"3. Scoring {len(df_news)} headlines in batches of {batch_size}...")
    all_scores = []
    
    with torch.no_grad():
        for batch_num, batch_texts in enumerate(dataloader):
            # Tokenize the batch
            inputs = tokenizer(list(batch_texts), padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            # Apply softmax to get probabilities [Positive, Negative, Neutral]
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: 0=Positive, 1=Negative, 2=Neutral
            pos_probs = probs[:, 0].cpu().numpy()
            neg_probs = probs[:, 1].cpu().numpy()
            
            # Calculate Net Sentiment = P(Pos) - P(Neg)
            net_sentiment = pos_probs - neg_probs
            all_scores.extend(net_sentiment)
            
            if (batch_num + 1) % 10 == 0:
                logger.info(f"Processed { (batch_num + 1) * batch_size } headlines...")

    # Assign scores back to dataframe
    df_news['Net_Sentiment'] = all_scores
    
    logger.info("4. Aggregating to Daily Sentiment Feature...")
    # Resample to daily frequency and calculate the mean sentiment for that day
    daily_sentiment = df_news['Net_Sentiment'].resample('D').mean().fillna(0)
    
    df_daily = pd.DataFrame({'FinBERT_Sentiment': daily_sentiment})
    df_daily.to_parquet(output_file)
    
    logger.info(f"✅ Daily sentiment successfully saved to {output_file}")
    return df_daily

def merge_alpha_features(har_file, sentiment_file, final_output_file):
    """Merges the baseline HAR features with the NLP sentiment scores."""
    logger.info("Merging HAR Baseline with Alpha Features...")
    
    df_har = pd.read_parquet(har_file)
    df_sent = pd.read_parquet(sentiment_file)
    
    # Left join to ensure we don't lose days where no news was published (Sentiment will be 0)
    df_final = df_har.join(df_sent, how='left')
    df_final['FinBERT_Sentiment'] = df_final['FinBERT_Sentiment'].fillna(0)
    
    df_final.to_parquet(final_output_file)
    logger.info(f"✅ Final Alpha Matrix saved to {final_output_file}")
    return df_final

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Define your paths (using the architecture we established)
    BASE_PATH = '/content/drive/MyDrive/AlphaVol_Data'
    HEADLINES_FILE = os.path.join(BASE_PATH, 'raw', 'gdelt_crypto_headlines.parquet')
    SENTIMENT_FILE = os.path.join(BASE_PATH, 'processed', 'daily_sentiment_scores.parquet')
    
    HAR_FILE = os.path.join(BASE_PATH, 'processed', 'har_features_v1.parquet')
    FINAL_MATRIX_FILE = os.path.join(BASE_PATH, 'processed', 'alpha_features_v1.parquet')
    
    # 1. Process Text to Numbers
    # df_daily_sent = process_sentiment(HEADLINES_FILE, SENTIMENT_FILE)
    
    # 2. Merge into the Final Target Matrix
    # if df_daily_sent is not None:
    #     df_final = merge_alpha_features(HAR_FILE, SENTIMENT_FILE, FINAL_MATRIX_FILE)
    #     print(df_final.tail())
