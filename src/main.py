from utils.logger import get_logger
from utils.config_reader import ConfigReader
from utils.s3_utils import upload_df_to_s3
from utils.validators import (
    validate_enriched, validate_profile,
    validate_phase_table, validate_phase_probability,
)
from data.data_loader import DataLoader
from analysis.enricher import DataEnricher
from analysis.aggregator import SeasonalAggregator
from analysis.cycle_analyzer import CycleAnalyzer
from analysis.phase_probability import PhaseProbabilityEstimator

logger = get_logger(__name__, log_to_file=True)


def main():
    try:
        logger.info("PIPELINE STARTED")

        output_cfg = ConfigReader("config").get("output")
        if not output_cfg:
            raise ValueError("Missing 'output' section in config")
        out_bucket      = output_cfg["bucket"]
        silver_key      = output_cfg["silver_key"]
        gold_key        = output_cfg["gold_key"]
        phase_table_key = output_cfg["phase_table_key"]
        prob_series_key = output_cfg["prob_series_key"]

        # 1. Load and prepare raw price data
        loader   = DataLoader()
        raw_data = loader.load_data()
        data     = loader.prepare_data(raw_data)
        logger.info(f"Data loaded: {len(data)} rows")

        price_col = data.select_dtypes("number").columns[0]
        logger.info(f"Price column: {price_col}")

        # 2. Enrich — calendar dims + days_since_halving (no cycle_phase yet)
        enricher = DataEnricher()
        enriched = enricher.enrich(data, price_col)

        # 3. Compute empirical boundaries and assign cycle IDs
        cycle_analyzer             = CycleAnalyzer()
        df_with_cycles, boundaries = cycle_analyzer.compute_boundaries(enriched, price_col)

        # 4. Stamp empirical cycle_phase using confirmed peak/trough boundaries
        enriched_final = cycle_analyzer.stamp_phases(df_with_cycles, boundaries)

        print("\n" + "=" * 60)
        print("ENRICHED DAILY TABLE — tail(10)")
        print("=" * 60)
        print(enriched_final.tail(10).to_string())

        # 5. Aggregate — seasonal profile (now uses empirically-correct cycle_phase)
        aggregator = SeasonalAggregator()
        profile    = aggregator.build_profile(enriched_final)

        print("\n" + "=" * 60)
        print("SEASONAL PROFILE TABLE — tail(10)")
        print("=" * 60)
        print(profile.tail(10).to_string())

        # 6. Build phase performance table
        phase_table = cycle_analyzer.build_phase_table(enriched_final, boundaries)

        print("\n" + "=" * 60)
        print("PHASE TABLE")
        print("=" * 60)
        print(phase_table.to_string())

        # 7. Phase probability series for the ongoing cycle
        ongoing_cycles = [
            cid for cid, b in boundaries.items() if not b["is_complete"]
        ]
        prob_series = None
        if ongoing_cycles:
            ongoing_label = ongoing_cycles[-1]
            estimator = PhaseProbabilityEstimator()
            estimator.fit(phase_table, boundaries, enriched_final)
            prob_series = estimator.estimate_series(enriched_final, ongoing_label)

            print("\n" + "=" * 60)
            print(f"PHASE PROBABILITY SERIES — cycle {ongoing_label}  tail(10)")
            print("=" * 60)
            print(prob_series.tail(10).to_string())
        else:
            logger.info("No ongoing cycle — skipping phase probability estimation")

        # 8. Validate before upload — abort if data is malformed
        validate_enriched(enriched_final)
        validate_profile(profile)
        validate_phase_table(phase_table)
        if prob_series is not None:
            validate_phase_probability(prob_series)

        # 9. Upload to S3
        upload_df_to_s3(
            enriched_final, out_bucket, silver_key,
            local_file="/tmp/btc_enriched.parquet",
        )
        upload_df_to_s3(
            profile, out_bucket, gold_key,
            local_file="/tmp/btc_seasonal_profile.parquet",
        )
        upload_df_to_s3(
            phase_table, out_bucket, phase_table_key,
            local_file="/tmp/btc_phase_table.parquet",
        )
        if prob_series is not None:
            upload_df_to_s3(
                prob_series, out_bucket, prob_series_key,
                local_file="/tmp/btc_phase_probability.parquet",
            )

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")

    except Exception:
        logger.exception("PIPELINE FAILED")
        raise


if __name__ == "__main__":
    main()
