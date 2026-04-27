from utils.logger import get_logger
from data.data_loader import DataLoader
from analysis.decomposition import Decomposer
from analysis.seasonality import SeasonalityAnalyzer
from analysis.bitcoin_cycle import BitcoinCycleAnalyzer

logger = get_logger(__name__, log_to_file=True)


def main():
    try:
        logger.info("PIPELINE STARTED")

        # =========================================================
        # 1. DATA LOADING
        # =========================================================
        logger.info("Step 1: Loading and preparing data")
        loader = DataLoader()

        raw_data = loader.load_data()
        data = loader.prepare_data(raw_data)

        logger.info(f"Data loaded successfully | rows={len(data)}")

        # =========================================================
        # 2. TIME SERIES DECOMPOSITION
        # =========================================================
        logger.info("Step 2: Running decomposition")

        decomposer = Decomposer()
        decomposition = decomposer.run(data)

        logger.info("Decomposition completed")

        # =========================================================
        # 3. SEASONALITY ANALYSIS
        # =========================================================
        logger.info("Step 3: Running seasonality analysis")

        seasonality = SeasonalityAnalyzer()

        monthly_profile = seasonality.build_monthly_profile(
            decomposition,
            data.index
        )

        seasonal_signal = seasonality.get_current_month_signal(monthly_profile)

        logger.info("Seasonality analysis completed")

        # =========================================================
        # 4. BITCOIN CYCLE ANALYSIS
        # =========================================================
        logger.info("Step 4: Running Bitcoin cycle analysis")

        cycle = BitcoinCycleAnalyzer()

        price_col = data.select_dtypes("number").columns[0]

        cycle_result = cycle.analyze(
            price_df=data,
            price_col=price_col
        )

        logger.info("Bitcoin cycle analysis completed")

        # =========================================================
        # FINAL OUTPUT SUMMARY (structured logging only)
        # =========================================================
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")

        logger.info(f"Seasonality Signal → {seasonal_signal}")
        logger.info(f"Cycle Regime → {cycle_result['cycle_phase']}")
        logger.info(f"Cycle Bias → {cycle_result['market_bias']}")

        #  UPLOAD ALL TO S3 AND ADD NEW ANALYSIS INVESTIGATE WHAT ANALYSIS SHOULD BE DONE 

    except Exception:
        logger.exception("PIPELINE FAILED")
        raise


if __name__ == "__main__":
    main()