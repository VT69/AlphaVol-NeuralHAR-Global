import os
import json
import pandas as pd
from datetime import datetime, timezone
from utils.logger import logger

class ManifestGenerator:
    def __init__(self, output_path: str = "data/manifest.json"):
        self.output_path = output_path
        self.entries = []
        
    def add_entry(self, file_path: str, source: str, asset: str, frequency: str, status: str = "OK"):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} not found.")
                
            df = pd.read_parquet(file_path)
            
            if df.empty:
                date_from, date_to = None, None
            else:
                if isinstance(df.index, pd.DatetimeIndex):
                    date_from = df.index.min().isoformat()
                    date_to = df.index.max().isoformat()
                elif 'date' in df.columns:
                    date_from = pd.to_datetime(df['date']).min().isoformat()
                    date_to = pd.to_datetime(df['date']).max().isoformat()
                elif 'timestamp' in df.columns:
                    date_from = pd.to_datetime(df['timestamp']).min().isoformat()
                    date_to = pd.to_datetime(df['timestamp']).max().isoformat()
                elif 'open_time' in df.columns:
                    date_from = pd.to_datetime(df['open_time']).min().isoformat()
                    date_to = pd.to_datetime(df['open_time']).max().isoformat()
                else:
                    date_from, date_to = "Unknown", "Unknown"
            
            missing_pct = (df.isnull().sum() / len(df)).to_dict() if len(df) > 0 else {}
            
            entry = {
                "file": file_path,
                "source": source,
                "asset": asset,
                "frequency": frequency,
                "date_from": date_from,
                "date_to": date_to,
                "rows": len(df),
                "columns": list(df.columns),
                "missing_pct": missing_pct,
                "pull_timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status
            }
        except Exception as e:
            entry = {
                "file": file_path,
                "source": source,
                "asset": asset,
                "frequency": frequency,
                "error": str(e),
                "pull_timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "FAILED"
            }
        self.entries.append(entry)
        
    def write(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.entries, f, indent=4)
        logger.info(f"Manifest written to {self.output_path}")
        self.print_quality_report()

    def print_quality_report(self):
        print("\n" + "="*80)
        print("DATA QUALITY REPORT")
        print("="*80)
        print(f"{'File':<45} | {'Rows':<8} | {'Status':<6} | {'Missing (%)'}")
        print("-" * 80)
        for e in self.entries:
            file_name = os.path.basename(e.get('file', 'Unknown'))
            rows = e.get('rows', 0)
            status = e.get('status', 'FAIL')
            missing = e.get('missing_pct', {})
            max_miss = max([v for v in missing.values() if isinstance(v, (int, float))]) * 100 if missing else 0
            print(f"{file_name:<45} | {rows:<8} | {status:<6} | {max_miss:.2f}%")
        print("="*80 + "\n")
