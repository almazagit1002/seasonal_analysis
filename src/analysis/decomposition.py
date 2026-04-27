import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from utils.logger import get_logger
from utils.config_reader import ConfigReader

logger = get_logger(__name__, log_to_file=True)


class Decomposer:
    """
    PURPOSE:
    --------
    Runs seasonal decomposition on a prepared time series using
    configuration-defined parameters.

    This class:
    - Reads decomposition settings from config.yaml
    - Performs decomposition (trend / seasonal / residual)
    - Returns raw statsmodels result

    DOES NOT:
    - Load data
    - Plot results
    - Modify DataFrame outside decomposition
    """

    def __init__(self, config_file: str = "config"):
        try:
            logger.info("Initializing Decomposer...")

            self.cfg_reader = ConfigReader(config_file)

            self.decomp_cfg = self.cfg_reader.get("decomposition")
            if not self.decomp_cfg:
                raise ValueError("Missing 'decomposition' section in config")

            self.period = self.decomp_cfg.get("period", 365)
            self.model = self.decomp_cfg.get("model", "multiplicative")

            logger.info(
                f"Decomposer ready (period={self.period}, model={self.model})"
            )

        except Exception:
            logger.exception("Failed to initialize Decomposer")
            raise

    def run(self, df: pd.DataFrame, column_name: str = None):
        """
        Performs seasonal decomposition on the provided time series.

        INPUT:
        ------
        df : pd.DataFrame
            Must already be cleaned and indexed by datetime

        column_name : str
            Column to decompose (auto-detected if None)

        OUTPUT:
        -------
        DecomposeResult (trend, seasonal, residual, observed)
        """

        try:
            if df is None or df.empty:
                raise ValueError("Input DataFrame is empty")

            # Auto-detect numeric column if not provided
            if not column_name:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found for decomposition")
                column_name = numeric_cols[0]

            logger.info(
                f"Running decomposition on '{column_name}' "
                f"(model={self.model}, period={self.period})"
            )

            series = df[column_name]

            # Multiplicative safety check
            if self.model == "multiplicative" and (series <= 0).any():
                logger.warning(
                    "Negative/zero values detected → switching to additive model"
                )
                model = "additive"
            else:
                model = self.model

            result = seasonal_decompose(
                series,
                model=model,
                period=self.period
            )

            logger.info("Decomposition completed successfully")

            return result

        except Exception:
            logger.exception("Decomposition failed")
            raise