import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_PHASES = {"EXPANSION", "DISTRIBUTION", "ACCUMULATION"}

_ENRICHED_COLS = {
    "price", "daily_return_pct", "days_since_halving", "cycle_month",
    "cycle_phase", "calendar_year", "calendar_month", "month_name",
    "day_of_week", "day_name",
}

_PROFILE_COLS = {
    "cycle_month", "cycle_phase", "calendar_month", "month_name",
    "day_of_week", "day_name", "avg_return_pct", "median_return_pct",
    "volatility_pct", "sample_size", "win_rate_pct",
}


def validate_enriched(df: pd.DataFrame) -> None:
    missing = _ENRICHED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Enriched table missing columns: {missing}")

    if df.index.duplicated().any():
        raise ValueError("Enriched table has duplicate dates in index")

    valid = df.dropna(subset=["days_since_halving"])

    if (valid["days_since_halving"] < 0).any():
        raise ValueError("days_since_halving has negative values")

    if (~valid["cycle_month"].between(1, 48)).any():
        bad = valid.loc[~valid["cycle_month"].between(1, 48), "cycle_month"].unique()
        raise ValueError(f"cycle_month out of range [1,48]: {bad}")

    if (~valid["cycle_phase"].isin(_VALID_PHASES)).any():
        bad = valid.loc[~valid["cycle_phase"].isin(_VALID_PHASES), "cycle_phase"].unique()
        raise ValueError(f"cycle_phase has unexpected values: {bad}")

    if (~df["day_of_week"].between(0, 6)).any():
        raise ValueError("day_of_week out of range [0,6]")

    if (~df["calendar_month"].between(1, 12)).any():
        raise ValueError("calendar_month out of range [1,12]")

    logger.info("Enriched table passed validation")


def validate_profile(df: pd.DataFrame) -> None:
    missing = _PROFILE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Profile table missing columns: {missing}")

    if (df["sample_size"] < 1).any():
        raise ValueError(f"sample_size < 1 in {(df['sample_size'] < 1).sum()} rows")

    if (~df["win_rate_pct"].between(0, 100)).any():
        raise ValueError("win_rate_pct out of range [0,100]")

    if (~df["cycle_month"].between(1, 48)).any():
        bad = df.loc[~df["cycle_month"].between(1, 48), "cycle_month"].unique()
        raise ValueError(f"cycle_month out of range [1,48]: {bad}")

    if (~df["cycle_phase"].isin(_VALID_PHASES)).any():
        bad = df.loc[~df["cycle_phase"].isin(_VALID_PHASES), "cycle_phase"].unique()
        raise ValueError(f"cycle_phase has unexpected values: {bad}")

    if df.duplicated(subset=["cycle_month", "cycle_phase", "calendar_month", "day_of_week"]).any():
        n = df.duplicated(subset=["cycle_month", "cycle_phase", "calendar_month", "day_of_week"]).sum()
        raise ValueError(f"Profile table has {n} duplicate (cycle_month, cycle_phase, calendar_month, day_of_week) rows")

    logger.info("Profile table passed validation")
