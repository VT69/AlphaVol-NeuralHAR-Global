import os
import time
from utils.logger import logger

def is_cached(filepath: str, max_age_hours: int = 24) -> bool:
    if not os.path.exists(filepath):
        return False
        
    file_age = time.time() - os.path.getmtime(filepath)
    is_fresh = file_age < (max_age_hours * 3600)
    
    if is_fresh:
        logger.info(f"CACHE HIT: {filepath} is fresh ({(file_age/3600):.2f} hours old).")
    else:
        logger.info(f"CACHE EXPIRED: {filepath} needs updating.")
        
    return is_fresh
