# utils/logger.py
import logging
import sys

def get_logger(name: str, log_to_file=False, log_dir="logs"):
    """
    Returns a configured logger that includes the module/class and method names.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # Avoid adding multiple handlers

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_to_file:
        import os
        from datetime import datetime
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger