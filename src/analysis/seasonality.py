import pandas as pd
from datetime import datetime
from utils.logger import get_logger
from utils.config_reader import ConfigReader

logger = get_logger(__name__, log_to_file=True)


class SeasonalityAnalyzer:
    """
    PURPOSE:
    --------
    Extracts seasonal patterns from a decomposed time series.

    INPUT:
    ------
    decomposition_result:
        Output from statsmodels seasonal_decompose

    index:
        Original datetime index of the dataset

    OUTPUT:
    -------
    - Monthly seasonal edge table
    - Current month seasonal signal

    DESIGN PRINCIPLES:
    ------------------
    - No decomposition logic
    - No data loading
    - No printing (returns structured data only)
    """

    def __init__(self, config_file: str = "config"):
        try:
            self.cfg_reader = ConfigReader(config_file)
            self.signal_cfg = self.cfg_reader.get("signals")

            # Optional thresholds (move from hardcoded logic)
            self.bull_threshold = self.signal_cfg.get("bull_threshold", 2)
            self.bear_threshold = self.signal_cfg.get("bear_threshold", -2)

            logger.info("SeasonalityAnalyzer initialized")

        except Exception:
            logger.exception("Failed to initialize SeasonalityAnalyzer")
            raise

    def build_monthly_profile(self, decomposition_result, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Converts seasonal component into monthly performance edge.

        LOGIC:
        ------
        1. Extract seasonal component
        2. Align it with original index
        3. Group by month
        4. Compute average seasonal factor
        5. Convert to percentage edge
        """

        try:
            logger.info("Building seasonal monthly profile...")

            seasonal_df = pd.DataFrame(
                {"factor": decomposition_result.seasonal},
                index=index
            )

            seasonal_df["month"] = seasonal_df.index.month

            monthly_profile = (
                seasonal_df.groupby("month")["factor"]
                .mean()
                .reset_index()
            )

            month_map = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }

            monthly_profile["month_name"] = monthly_profile["month"].map(month_map)

            # Convert multiplicative factor → percentage edge
            monthly_profile["historical_edge_pct"] = (
                (monthly_profile["factor"] - 1) * 100
            )

            # Sort best → worst months
            monthly_profile = monthly_profile.sort_values(
                "historical_edge_pct",
                ascending=False
            )

            logger.info("Monthly seasonal profile built successfully")

            return monthly_profile[["month_name", "historical_edge_pct"]]

        except Exception:
            logger.exception("Failed to build monthly profile")
            raise

    def get_current_month_signal(self, monthly_profile: pd.DataFrame) -> dict:
        """
        Generates trading signal for the current month.

        OUTPUT:
        -------
        {
            month: str,
            edge: float,
            signal: "BULLISH" | "BEARISH" | "NEUTRAL"
        }
        """

        try:
            current_month = datetime.now().strftime("%b")

            row = monthly_profile[
                monthly_profile["month_name"] == current_month
            ].iloc[0]

            edge = row["historical_edge_pct"]

            if edge > self.bull_threshold:
                signal = "BULLISH"
            elif edge < self.bear_threshold:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            logger.info(f"Seasonal signal generated for {current_month}: {signal}")

            return {
                "month": current_month,
                "edge": float(edge),
                "signal": signal
            }

        except Exception:
            logger.exception("Failed to compute current month signal")
            raise