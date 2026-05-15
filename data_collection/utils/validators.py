import pandas as pd
from utils.logger import logger

def enforce_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if df.empty:
        return df
        
    for col, dtype in schema.items():
        if col not in df.columns and col != 'index':
            continue
            
        try:
            if dtype == 'datetime':
                if col == 'index':
                    df.index = pd.to_datetime(df.index, utc=True)
                else:
                    df[col] = pd.to_datetime(df[col], utc=True)
            elif dtype == 'float':
                if col == 'index':
                    df.index = df.index.astype(float)
                else:
                    df[col] = df[col].astype(float)
            elif dtype == 'int':
                df[col] = df[col].astype(int)
        except Exception as e:
            logger.error(f"Failed to cast {col} to {dtype}: {e}")
            
    return df
