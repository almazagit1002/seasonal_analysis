import pandas as pd
import numpy as np
from utils.s3_utils import read_s3_file
from utils.config_reader import ConfigReader
from utils.logger import get_logger

logger = get_logger(__name__, log_to_file=True)


class DataLoader:
    """
    Responsible ONLY for:
    - Loading data from S3
    - Cleaning and preparing time series
    """

    def __init__(self, config_file: str = "config"):
        try:
            logger.info("Initializing DataLoader...")
            self.cfg_reader = ConfigReader(config_file)

            data_cfg = self.cfg_reader.get("data")
            if not data_cfg:
                raise ValueError("Missing 'data' section in config")

            self.bucket = data_cfg.get("bucket")
            self.key = data_cfg.get("bronze_key")
            self.date_column = data_cfg.get("date_column")  # optional but recommended

            logger.info(f"Configured for s3://{self.bucket}/{self.key}")

        except Exception:
            logger.exception("Failed to initialize DataLoader")
            raise

    def load_data(self) -> pd.DataFrame:
        """Loads raw data from S3"""
        try:
            logger.info("Downloading data from S3...")
            df = read_s3_file(bucket=self.bucket, key=self.key)

            if df is None or df.empty:
                raise ValueError("Loaded dataset is empty")

            logger.info(f"Raw data loaded: {len(df)} rows")
            return df

        except Exception:
            logger.exception("Error loading data")
            raise

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares time series data"""
        try:
            logger.info("Preparing data...")

            # 1. Detect or use configured date column
            date_col = self.date_column
            if not date_col:
                date_col = next(
                    (c for c in df.columns if c.lower() in ["date", "timestamp", "time"]),
                    None
                )

            if not date_col:
                datetime_cols = df.select_dtypes(include=["datetime64"]).columns
                if len(datetime_cols) == 0:
                    raise ValueError("No valid date column found")
                date_col = datetime_cols[0]

            logger.info(f"Using date column: {date_col}")

            # 2. Convert + sort
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).set_index(date_col)

            # 3. Enforce daily frequency
            df = df.asfreq("D")

            # 4. Handle missing values
            df = df.ffill()

            # 5. Basic sanity check
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for analysis")

            logger.info(f"Prepared dataset: {len(df)} rows, {len(df.columns)} columns")

            return df

        except Exception:
            logger.exception("Error preparing data")
            raise


