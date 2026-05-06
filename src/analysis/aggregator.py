import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__, log_to_file=True)

_GROUP_COLS = [
    "cycle_month",
    "cycle_phase",
    "calendar_month",
    "month_name",
    "day_of_week",
    "day_name",
]


class SeasonalAggregator:
    """
    Aggregates the enriched daily DataFrame into a seasonal profile table.

    Each row in the output represents one unique combination of
    (cycle_month × calendar_month × day_of_week) and summarises
    the historical return distribution for that slot.
    """

    def build_profile(self, enriched: pd.DataFrame) -> pd.DataFrame:
        """
        INPUT:  enriched DataFrame from DataEnricher.enrich()
        OUTPUT: aggregated profile with one row per dimension combination
        """
        try:
            logger.info("Building seasonal profile...")

            # Drop rows with no cycle context (pre-first-halving) or no return
            df = enriched.dropna(subset=["days_since_halving", "daily_return_pct"]).copy()

            # Cast cycle_month to int now that NaNs are gone
            df["cycle_month"] = df["cycle_month"].astype(int)

            base = (
                df.groupby(_GROUP_COLS)["daily_return_pct"]
                .agg(
                    avg_return_pct="mean",
                    median_return_pct="median",
                    volatility_pct="std",
                    sample_size="count",
                )
                .reset_index()
            )

            wins = (
                df[df["daily_return_pct"] > 0]
                .groupby(_GROUP_COLS)["daily_return_pct"]
                .count()
                .reset_index(name="win_count")
            )

            profile = base.merge(wins, on=_GROUP_COLS, how="left")
            profile["win_count"] = profile["win_count"].fillna(0)
            profile["win_rate_pct"] = (
                (profile["win_count"] / profile["sample_size"]) * 100
            ).round(1)
            profile.drop(columns=["win_count"], inplace=True)

            for col in ["avg_return_pct", "median_return_pct", "volatility_pct"]:
                profile[col] = profile[col].round(4)

            profile = profile.sort_values(
                ["cycle_month", "calendar_month", "day_of_week"]
            ).reset_index(drop=True)

            logger.info(
                f"Profile built: {len(profile)} rows "
                f"({profile['cycle_month'].nunique()} cycle months, "
                f"{profile['sample_size'].min()}–{profile['sample_size'].max()} obs/cell)"
            )
            return profile

        except Exception:
            logger.exception("Failed to build seasonal profile")
            raise
