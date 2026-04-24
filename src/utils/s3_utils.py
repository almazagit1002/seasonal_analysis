import boto3
import pandas as pd
from io import BytesIO
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def read_s3_file(bucket: str, key: str, file_type: str = "parquet") -> pd.DataFrame:
    """
    Reads a file from S3 and returns a Pandas DataFrame.

    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key (path to file in bucket)
        file_type (str): 'parquet' or 'csv'. Default is 'parquet'

    Returns:
        pd.DataFrame: The loaded dataframe

    Raises:
        ValueError: if file_type is not supported
        boto3.exceptions or pd.errors: if download or parsing fails
    """
    s3 = boto3.client("s3")
    logger.info(f"Reading {file_type.upper()} file from s3://{bucket}/{key}")

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        
        if file_type.lower() == "parquet":
            df = pd.read_parquet(BytesIO(data))
        elif file_type.lower() == "csv":
            df = pd.read_csv(BytesIO(data))
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")
        
        logger.info(f"Loaded {len(df)} rows from s3://{bucket}/{key}")
        return df

    except Exception as e:
        logger.error(f"Failed to read s3://{bucket}/{key}: {e}")
        raise

def upload_df_to_s3(df: pd.DataFrame, bucket: str, s3_key: str, local_file: str = "/tmp/temp.parquet"):
    """
    Upload a pandas DataFrame to S3 as a Parquet file.
    
    Args:
        df (pd.DataFrame): DataFrame to upload.
        bucket (str): S3 bucket name.
        s3_key (str): S3 object key (path in bucket).
        local_file (str): Temporary local file path to save parquet before upload.
    """
    try:
        logger.info(f"Uploading DataFrame to s3://{bucket}/{s3_key}")
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        # Save locally first
        df.to_parquet(local_file, engine='pyarrow', index=False)
        # Upload to S3
        s3 = boto3.client("s3")
        s3.upload_file(local_file, bucket, s3_key)
        logger.info("Upload completed successfully")
    except Exception as e:
        logger.exception(f"Failed to upload DataFrame to S3: s3://{bucket}/{s3_key}")
        raise e
    
def upload_file_to_s3(local_file: str, bucket: str, s3_key: str):
    """
    Upload any local file to S3.

    Args:
        local_file (str): Path to the local file to upload.
        bucket (str): Target S3 bucket name.
        s3_key (str): Target S3 object key (path in bucket).

    Raises:
        FileNotFoundError: if local_file does not exist.
        boto3.exceptions: if upload fails.
    """
    if not os.path.exists(local_file):
        logger.error(f"Local file does not exist: {local_file}")
        raise FileNotFoundError(f"Local file not found: {local_file}")

    try:
        logger.info(f"Uploading file {local_file} to s3://{bucket}/{s3_key}")
        s3 = boto3.client("s3")
        s3.upload_file(local_file, bucket, s3_key)
        logger.info(f"Upload completed successfully: s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.exception(f"Failed to upload file to S3: {local_file} → s3://{bucket}/{s3_key}")
        raise e
    

def download_file_from_s3(bucket: str, s3_key: str, local_file: str):
    """
    Download a file from S3 to local path.
    """
 


    os.makedirs(os.path.dirname(local_file), exist_ok=True)

    try:
        logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_file}")
        s3 = boto3.client("s3")
        s3.download_file(bucket, s3_key, local_file)
        logger.info("Download completed successfully")
    except Exception as e:
        logger.exception(f"Failed to download s3://{bucket}/{s3_key}")
        raise e