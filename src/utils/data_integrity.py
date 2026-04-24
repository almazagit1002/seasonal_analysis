import pandas as pd
import numpy as np
from s3_utils import read_s3_file 
from config_reader import ConfigReader
from logger import get_logger

logger = get_logger(__name__)

def find_timestamp_column(df):
    """Try to identify the time column automatically."""
    # Check for existing datetime dtypes
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetimetz']).columns
    if len(datetime_cols) > 0:
        return datetime_cols[0]
    
    # Check common names
    common_names = ['timestamp', 'date', 'datetime', 'time', 'opened_at']
    for col in common_names:
        if col in df.columns:
            return col
            
    # Brute force conversion check on first row
    for col in df.columns:
        try:
            pd.to_datetime(df[col].iloc[0])
            return col
        except:
            continue
            
    raise ValueError("Could not automatically find a timestamp column in the dataset.")

def run_integrity_check():
    # 1. Load Config using your ConfigReader
    # Assuming the file is config/config.yaml, pass 'config'
    config_store = ConfigReader("config")
    data_cfg = config_store.get("data")
    
    if not data_cfg:
        raise KeyError("Data configuration not found in YAML.")

    # 2. Download data from S3
    try:
        df = read_s3_file(
            bucket=data_cfg['bucket'],
            key=data_cfg['bronze_key']
        )
    except Exception as e:
        logger.error(f"Failed to fetch data for integrity check: {e}")
        return

    # 3. Identify and Format Time Column
    time_col = find_timestamp_column(df)
    logger.info(f"Analyzing continuity on column: '{time_col}'")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Remove duplicates to avoid frequency confusion
    df = df.drop_duplicates(subset=[time_col])
    
    # 4. Continuity Analysis (Daily)
    first_date = df[time_col].min().floor('D')
    last_date = df[time_col].max().floor('D')
    
    # Create the 'perfect' daily calendar
    ideal_range = pd.date_range(start=first_date, end=last_date, freq='D')
    
    # Compare actual data to ideal calendar
    actual_dates = set(df[time_col].dt.floor('D'))
    ideal_dates = set(ideal_range)
    missing_dates = sorted(list(ideal_dates - actual_dates))
    
    # 5. Reporting
    print("\n" + "="*50)
    print(f"FINANCIAL DATA INTEGRITY REPORT")
    print("="*50)
    print(f"Asset/Key:    {data_cfg['bronze_key']}")
    print(f"Start Date:   {first_date.date()}")
    print(f"End Date:     {last_date.date()}")
    print(f"Total Rows:   {len(df)}")
    print("-" * 50)
    
    if not missing_dates:
        print("RESULT: PASS - No days missing.")
        logger.info("Data continuity check passed.")
    else:
        gap_pct = (len(missing_dates) / len(ideal_range)) * 100
        print(f"❌ RESULT: FAIL - {len(missing_dates)} days missing ({gap_pct:.2f}% leakage)")
        print(f"First 5 missing dates: {[d.strftime('%Y-%m-%d') for d in missing_dates[:5]]}")
        logger.warning(f"Data continuity failed. {len(missing_dates)} missing days detected.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_integrity_check()