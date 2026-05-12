import re
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_PHASES   = {"EXPANSION", "DISTRIBUTION", "ACCUMULATION"}
_VALID_STATUSES = {"complete", "ongoing"}
_YEAR_PATTERN   = re.compile(r"^\d{4}$")

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


_PHASE_TABLE_COLS = {
    "cycle_id", "cycle_phase", "duration_days",
    "avg_daily_%", "total_return_%", "wealth_mult", "status",
}


def validate_phase_table(df: pd.DataFrame) -> None:
    missing = _PHASE_TABLE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Phase table missing columns: {missing}")

    if df.duplicated(subset=["cycle_id", "cycle_phase"]).any():
        n = df.duplicated(subset=["cycle_id", "cycle_phase"]).sum()
        raise ValueError(f"Phase table has {n} duplicate (cycle_id, cycle_phase) rows")

    bad_ids = [
        cid for cid in df["cycle_id"].astype(str).unique()
        if not _YEAR_PATTERN.match(cid)
    ]
    if bad_ids:
        raise ValueError(f"Phase table has non-year cycle_id values: {bad_ids}")

    bad_phases = set(df["cycle_phase"].astype(str).unique()) - _VALID_PHASES
    if bad_phases:
        raise ValueError(f"Phase table has unexpected cycle_phase values: {bad_phases}")

    if (df["duration_days"] < 1).any():
        raise ValueError("Phase table has duration_days < 1")

    bad_status = set(df["status"].unique()) - _VALID_STATUSES
    if bad_status:
        raise ValueError(f"Phase table has unexpected status values: {bad_status}")

    logger.info("Phase table passed validation")


_PROB_REQUIRED_COLS = {
    "cycle_day",
    "p_expansion_prior", "p_distribution_prior", "p_accumulation_prior",
    "dominant_phase_prior",
    "drawdown_from_ath",
    "p_expansion_posterior", "p_distribution_posterior", "p_accumulation_posterior",
    "dominant_phase_posterior", "signals_agree",
}
_PROB_PRIOR_COLS = [
    "p_expansion_prior", "p_distribution_prior", "p_accumulation_prior"
]
_PROB_POST_COLS = [
    "p_expansion_posterior", "p_distribution_posterior", "p_accumulation_posterior"
]


def validate_phase_probability(df: pd.DataFrame) -> None:
    missing = _PROB_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Phase probability table missing columns: {missing}")

    if (df["cycle_day"] < 0).any():
        raise ValueError("Phase probability table has negative cycle_day values")

    for col in _PROB_PRIOR_COLS:
        if not df[col].between(0.0, 1.0).all():
            raise ValueError(f"{col} has values outside [0, 1]")

    prior_sum = df[_PROB_PRIOR_COLS].sum(axis=1)
    if not prior_sum.between(0.999, 1.001).all():
        bad = prior_sum[~prior_sum.between(0.999, 1.001)]
        raise ValueError(
            f"Prior probabilities do not sum to 1 in {len(bad)} rows "
            f"(range [{bad.min():.4f}, {bad.max():.4f}])"
        )

    # Posterior columns are optional (NaN when drawdown stats unavailable)
    post_present = df[_PROB_POST_COLS].notna().all(axis=1)
    if post_present.any():
        for col in _PROB_POST_COLS:
            subset = df.loc[post_present, col]
            if not subset.between(0.0, 1.0).all():
                raise ValueError(f"{col} has values outside [0, 1]")

        post_sum = df.loc[post_present, _PROB_POST_COLS].sum(axis=1)
        if not post_sum.between(0.999, 1.001).all():
            bad = post_sum[~post_sum.between(0.999, 1.001)]
            raise ValueError(
                f"Posterior probabilities do not sum to 1 in {len(bad)} rows"
            )

    bad_dominant = (
        df["dominant_phase_prior"]
        .dropna()
        .pipe(lambda s: s[~s.isin(_VALID_PHASES)])
    )
    if not bad_dominant.empty:
        raise ValueError(
            f"dominant_phase_prior has unexpected values: {bad_dominant.unique()}"
        )

    logger.info("Phase probability table passed validation")
