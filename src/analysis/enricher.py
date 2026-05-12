import numpy as np
import pandas as pd
from datetime import datetime

from utils.logger import get_logger
from utils.config_reader import ConfigReader

logger = get_logger(__name__, log_to_file=True)

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
_DAY_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}


class DataEnricher:
    """
    Stamps every row of the price DataFrame with halving-cycle timing
    (days_since_halving, cycle_month) and calendar context so that any
    downstream step can slice by any combination of dimensions.

    cycle_phase is NOT assigned here — it is stamped empirically by
    CycleAnalyzer.stamp_phases() after boundary detection.
    """

    def __init__(self, config_file: str = "config"):
        try:
            logger.info("Initializing DataEnricher...")
            cfg       = ConfigReader(config_file)
            cycle_cfg = cfg.get("bitcoin_cycle")
            if not cycle_cfg:
                raise ValueError("Missing 'bitcoin_cycle' section in config")

            halvings_raw = cycle_cfg.get("halvings", [])
            if not halvings_raw:
                raise ValueError("No halvings defined in 'bitcoin_cycle.halvings'")

            self.halvings = sorted(
                datetime.fromisoformat(h["date"]) for h in halvings_raw
            )

            logger.info(f"DataEnricher ready — {len(self.halvings)} halving dates loaded")

        except Exception:
            logger.exception("Failed to initialize DataEnricher")
            raise

    def enrich(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Returns a new DataFrame indexed by date with price, return,
        cycle-timing columns (days_since_halving, cycle_month), and
        calendar dimension columns.

        Rows before the first known halving are included but have NaN
        for all cycle columns.  cycle_phase is absent — see CycleAnalyzer.
        """
        try:
            logger.info(f"Enriching {len(df)} rows...")

            enriched = df[[price_col]].copy()
            enriched.index.name = "date"

            enriched["daily_return_pct"] = enriched[price_col].pct_change() * 100

            last_date     = df.index.max().to_pydatetime().replace(tzinfo=None)
            past_halvings = [h for h in self.halvings if h <= last_date]

            if not past_halvings:
                enriched["days_since_halving"] = np.nan
                enriched["cycle_month"]        = np.nan
            else:
                halving_arr = np.array(past_halvings, dtype="datetime64[D]")
                dates_arr   = df.index.values.astype("datetime64[D]")

                idx   = np.searchsorted(halving_arr, dates_arr, side="right") - 1
                valid = idx >= 0

                safe_idx  = np.maximum(idx, 0)
                raw_days  = (dates_arr - halving_arr[safe_idx]) / np.timedelta64(1, "D")
                days_since = np.where(valid, raw_days, np.nan)

                enriched["days_since_halving"] = days_since
                enriched["cycle_month"] = np.where(
                    valid,
                    np.minimum(days_since // 30 + 1, 48),
                    np.nan,
                )

            enriched["calendar_year"]  = df.index.year
            enriched["calendar_month"] = df.index.month
            enriched["month_name"]     = df.index.month.map(_MONTH_NAMES)
            enriched["day_of_week"]    = df.index.dayofweek
            enriched["day_name"]       = df.index.dayofweek.map(_DAY_NAMES)

            logger.info("Enrichment completed successfully")
            return enriched

        except Exception:
            logger.exception("Enrichment failed")
            raise
