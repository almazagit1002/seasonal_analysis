from utils.logger import get_logger
from utils.config_reader import ConfigReader
from utils.s3_utils import upload_df_to_s3
from utils.validators import validate_enriched, validate_profile
from data.data_loader import DataLoader
from analysis.enricher import DataEnricher
from analysis.aggregator import SeasonalAggregator

logger = get_logger(__name__, log_to_file=True)


def main():
    try:
        logger.info("PIPELINE STARTED")

        output_cfg = ConfigReader("config").get("output")
        if not output_cfg:
            raise ValueError("Missing 'output' section in config")
        out_bucket = output_cfg["bucket"]
        silver_key = output_cfg["silver_key"]
        gold_key = output_cfg["gold_key"]

        # 1. Load and prepare raw price data
        loader = DataLoader()
        raw_data = loader.load_data()
        data = loader.prepare_data(raw_data)
        logger.info(f"Data loaded: {len(data)} rows")

        price_col = data.select_dtypes("number").columns[0]
        logger.info(f"Price column: {price_col}")

        # 2. Enrich — stamp every row with cycle + calendar dimensions
        enricher = DataEnricher()
        enriched = enricher.enrich(data, price_col)

        print("\n" + "=" * 60)
        print("ENRICHED DAILY TABLE — tail(10)")
        print("=" * 60)
        print(enriched.tail(10).to_string())

        # 3. Aggregate — build the combined seasonal profile
        aggregator = SeasonalAggregator()
        profile = aggregator.build_profile(enriched)

        print("\n" + "=" * 60)
        print("SEASONAL PROFILE TABLE — tail(10)")
        print("=" * 60)
        print(profile.tail(10).to_string())

        # 4. Validate before upload — abort if data is malformed
        validate_enriched(enriched)
        validate_profile(profile)

        # 5. Upload to S3
        upload_df_to_s3(enriched, out_bucket, silver_key, local_file="/tmp/btc_enriched.parquet")
        upload_df_to_s3(profile,  out_bucket, gold_key,   local_file="/tmp/btc_seasonal_profile.parquet")

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")

    except Exception:
        logger.exception("PIPELINE FAILED")
        raise


if __name__ == "__main__":
    main()
