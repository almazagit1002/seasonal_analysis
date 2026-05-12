import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.config_reader import ConfigReader
from analysis.phase_detector import assign_cycle_id, compute_cycle_boundaries

logger = get_logger(__name__, log_to_file=True)

_PHASE_ORDER = ["EXPANSION", "DISTRIBUTION", "ACCUMULATION"]


def _empirical_phase(days: float, b: dict) -> str:
    if days <= b["peak_day"]:
        return "EXPANSION"
    if days <= b["trough_day"]:
        return "DISTRIBUTION"
    return "ACCUMULATION"


def _phase_status(phase: str, b: dict) -> str:
    if b.get("is_complete"):
        return "complete"
    cur = b["max_day"]
    if phase == "EXPANSION":
        return "complete" if cur > b["peak_day"] else "ongoing"
    if phase == "DISTRIBUTION":
        if b.get("trough_confirmed", True) and cur > b["trough_day"]:
            return "complete"
        return "ongoing"
    return "ongoing"


def _wealth_multiplier(series: pd.Series) -> float:
    r = series.dropna().values
    return float(np.prod(1 + r / 100))


class CycleAnalyzer:
    """
    Builds empirical BTC halving-cycle phase boundaries and performance stats.

    All parameters (halving dates, detection thresholds) are read from config —
    nothing is hardcoded.  Adding a new halving to config is the only change
    required to support future cycles.

    Typical call sequence in the pipeline:
        df_with_cycles, boundaries = analyzer.compute_boundaries(enriched, price_col)
        enriched_final = analyzer.stamp_phases(df_with_cycles, boundaries)
        phase_table    = analyzer.build_phase_table(enriched_final, boundaries)
    """

    def __init__(self, config_file: str = "config") -> None:
        try:
            logger.info("Initializing CycleAnalyzer...")
            cfg       = ConfigReader(config_file)
            cycle_cfg = cfg.get("bitcoin_cycle")
            if not cycle_cfg:
                raise ValueError("Missing 'bitcoin_cycle' section in config")

            self._halvings_cfg = cycle_cfg.get("halvings", [])
            self._peak_cfg     = cycle_cfg.get("peak_detection", {})
            self._trough_cfg   = cycle_cfg.get("trough_detection", {})

            if not self._halvings_cfg:
                raise ValueError("No halvings defined in 'bitcoin_cycle.halvings'")
            for key in ("drawdown_threshold", "confirmation_days", "fallback_cap_days"):
                if key not in self._peak_cfg:
                    raise ValueError(f"Missing 'peak_detection.{key}' in config")
            for key in ("recovery_pct", "confirmation_days"):
                if key not in self._trough_cfg:
                    raise ValueError(f"Missing 'trough_detection.{key}' in config")

            self._halving_labels = [h["label"] for h in self._halvings_cfg]

            logger.info(
                f"CycleAnalyzer ready — {len(self._halvings_cfg)} halvings in config  "
                f"labels: {self._halving_labels}"
            )

        except Exception:
            logger.exception("Failed to initialize CycleAnalyzer")
            raise

    def compute_boundaries(
        self, enriched: pd.DataFrame, price_col: str
    ) -> tuple[pd.DataFrame, dict]:
        """
        Assign cycle IDs to every row and compute empirical phase boundaries.

        Parameters
        ----------
        enriched  : DataFrame from DataEnricher.enrich()
        price_col : name of the closing-price column in enriched

        Returns
        -------
        (df_with_cycle_id, boundaries)
          df_with_cycle_id : enriched copy with 'cycle_id' column added and
                             price column renamed to 'price'
          boundaries       : {cycle_label: {peak_day, peak_price, trough_day,
                              trough_price, max_day, is_complete, trough_confirmed}}
        """
        try:
            logger.info("Computing cycle boundaries...")

            df = enriched.copy()
            if price_col != "price":
                df = df.rename(columns={price_col: "price"})

            df         = assign_cycle_id(df, self._halvings_cfg)
            boundaries = compute_cycle_boundaries(
                df, self._halvings_cfg, self._peak_cfg, self._trough_cfg
            )

            logger.info(f"Boundaries computed for: {list(boundaries.keys())}")
            return df, boundaries

        except Exception:
            logger.exception("Failed to compute cycle boundaries")
            raise

    def stamp_phases(self, df: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
        """
        Stamp 'cycle_phase' onto every row using empirical boundaries.

        df must already have 'cycle_id' and 'days_since_halving' columns
        (i.e. the first return value of compute_boundaries).

        For the ongoing cycle: if the trough is not yet confirmed, all
        post-peak rows are kept as DISTRIBUTION rather than ACCUMULATION.
        """
        try:
            logger.info("Stamping empirical cycle phases...")
            df = df.copy()

            def _get_phase(row):
                cid = row["cycle_id"]
                if cid is None or cid not in boundaries:
                    return None
                if pd.isna(row["days_since_halving"]):
                    return None
                return _empirical_phase(row["days_since_halving"], boundaries[cid])

            df["cycle_phase"] = df.apply(_get_phase, axis=1)

            for cid, b in boundaries.items():
                if b.get("is_complete") or b.get("trough_confirmed", True):
                    continue
                mask = (df["cycle_id"] == cid) & (df["cycle_phase"] == "ACCUMULATION")
                if mask.any():
                    logger.info(
                        f"[{cid}] trough not confirmed — "
                        f"reclassifying {mask.sum()} ACCUMULATION rows as DISTRIBUTION"
                    )
                    df.loc[mask, "cycle_phase"] = "DISTRIBUTION"

            stamped = df["cycle_phase"].notna().sum()
            logger.info(f"Phase stamping completed — {stamped} rows assigned a phase")
            return df

        except Exception:
            logger.exception("Failed to stamp cycle phases")
            raise

    def build_phase_table(self, df: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
        """
        Compute per-(cycle × empirical phase) performance statistics.

        df must have 'cycle_id', 'cycle_phase', and 'daily_return_pct' columns
        (i.e. the output of stamp_phases).

        Returns
        -------
        DataFrame with columns: cycle_id, cycle_phase, duration_days,
        avg_daily_%, total_return_%, wealth_mult, status
        """
        try:
            logger.info("Building phase table...")

            valid = df[
                df["cycle_id"].notna() &
                df["cycle_phase"].notna() &
                df["daily_return_pct"].notna()
            ].copy()

            grp    = valid.groupby(["cycle_id", "cycle_phase"])["daily_return_pct"]
            wealth = grp.apply(_wealth_multiplier)

            stats = pd.DataFrame({
                "duration_days":  grp.count(),
                "avg_daily_%":    grp.mean().round(4),
                "total_return_%": ((wealth - 1) * 100).round(1),
                "wealth_mult":    wealth.round(2),
            }).reset_index()

            stats["status"] = stats.apply(
                lambda r: _phase_status(
                    r["cycle_phase"],
                    boundaries.get(r["cycle_id"], {}),
                ),
                axis=1,
            )

            stats["cycle_id"] = pd.Categorical(
                stats["cycle_id"], categories=self._halving_labels, ordered=True
            )
            stats["cycle_phase"] = pd.Categorical(
                stats["cycle_phase"], categories=_PHASE_ORDER, ordered=True
            )
            stats = stats.sort_values(["cycle_id", "cycle_phase"]).reset_index(drop=True)

            logger.info(
                f"Phase table built: {len(stats)} rows across "
                f"{stats['cycle_id'].nunique()} cycles"
            )
            return stats

        except Exception:
            logger.exception("Failed to build phase table")
            raise
