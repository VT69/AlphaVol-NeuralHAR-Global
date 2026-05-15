import logging
import os

def setup_logger(name: str) -> logging.Logger:
    log_dir = "data"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

logger = setup_logger("AlphaVol")
