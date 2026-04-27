from datetime import datetime
import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.config_reader import ConfigReader

logger = get_logger(__name__, log_to_file=True)


class BitcoinCycleAnalyzer:
    """
    BITCOIN 4-YEAR CYCLE ANALYZER

    PURPOSE:
    --------
    Models Bitcoin halving cycle using time + optional market data
    to estimate macro regime.

    This is NOT a simple timer.
    It is a macro market regime estimator.

    OUTPUT:
    -------
    - cycle phase (accumulation / expansion / distribution / contraction)
    - days since halving
    - cycle position (0 → 1 normalized progress)
    - confidence score

    INPUT (optional enhancement later):
    -------------------------------
    - price series (for returns / drawdown / volatility)
    """

    def __init__(self, config_file: str = "config"):
        try:
            logger.info("Initializing BitcoinCycleAnalyzer...")

            self.cfg_reader = ConfigReader(config_file)
            self.config = self.cfg_reader.get("bitcoin_cycle")

            if not self.config:
                raise ValueError("Missing 'bitcoin_cycle' config section")

            # Halving dates (config-driven)
            self.halvings = [
                datetime.fromisoformat(d)
                for d in self.config.get("halvings", [])
            ]

            # Cycle thresholds
            self.bull_days = self.config.get("bull_days", 450)
            self.bear_days = self.config.get("bear_days", 800)

            logger.info("BitcoinCycleAnalyzer ready")

        except Exception:
            logger.exception("Failed to initialize BitcoinCycleAnalyzer")
            raise

    def analyze(self, price_df: pd.DataFrame = None, price_col: str = None) -> dict:
        """
        MAIN ANALYSIS FUNCTION

        Steps:
        ------
        1. Identify last halving
        2. Compute days since halving
        3. Optional: compute market features (if price data provided)
        4. Map to cycle phase
        5. Return structured result
        """

        try:
            now = datetime.now()

            # --- Find last halving ---
            last_halving = max(h for h in self.halvings if h <= now)
            days_since = (now - last_halving).days

            # --- Normalize cycle position (0 → 1 over ~4 years) ---
            cycle_length = 4 * 365
            cycle_position = min(days_since / cycle_length, 1.0)

            # --- Optional market features ---
            volatility = None
            returns = None

            if price_df is not None and price_col:
                series = price_df[price_col]

                returns = series.pct_change().mean() * 100
                volatility = series.pct_change().std() * 100

            # --- Phase classification (time-based core logic) ---
            if days_since <= self.bull_days:
                phase = "EXPANSION"
                bias = "RISK-ON"

            elif self.bull_days < days_since <= self.bear_days:
                phase = "DISTRIBUTION / CORRECTION"
                bias = "RISK-OFF"

            else:
                phase = "ACCUMULATION"
                bias = "EARLY RISK-ON"

            # --- Confidence score (simple heuristic) ---
            confidence = 0.6  # base

            if returns is not None:
                confidence += min(abs(returns) / 10, 0.2)

            if volatility is not None:
                confidence += min(volatility / 20, 0.2)

            confidence = round(min(confidence, 1.0), 2)

            result = {
                "last_halving": last_halving.strftime("%Y-%m-%d"),
                "days_since": days_since,
                "cycle_position": round(cycle_position, 3),
                "cycle_phase": phase,
                "market_bias": bias,
                "returns_%": round(returns, 3) if returns else None,
                "volatility_%": round(volatility, 3) if volatility else None,
                "confidence": confidence
            }

            logger.info(f"Cycle analysis complete: {phase}")

            return result

        except Exception:
            logger.exception("Bitcoin cycle analysis failed")
            raise