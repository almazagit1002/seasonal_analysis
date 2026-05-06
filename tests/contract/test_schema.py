import pytest
import numpy as np
import pandas as pd

from utils.validators import validate_enriched, validate_profile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _valid_enriched(n=3):
    idx = pd.date_range("2024-04-21", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "price":              [80_000.0] * n,
        "daily_return_pct":   [0.5]       * n,
        "days_since_halving": [1.0, 2.0, 3.0][:n],
        "cycle_month":        [1.0]        * n,
        "cycle_phase":        ["EXPANSION"] * n,
        "calendar_year":      [2024]        * n,
        "calendar_month":     [4]           * n,
        "month_name":         ["Apr"]       * n,
        "day_of_week":        [0, 1, 2][:n],
        "day_name":           ["Monday", "Tuesday", "Wednesday"][:n],
    }, index=idx)


def _valid_profile(n=3):
    return pd.DataFrame({
        "cycle_month":        [1, 2, 3][:n],
        "cycle_phase":        ["EXPANSION"] * n,
        "calendar_month":     [4]           * n,
        "month_name":         ["Apr"]       * n,
        "day_of_week":        [0, 1, 2][:n],
        "day_name":           ["Monday", "Tuesday", "Wednesday"][:n],
        "avg_return_pct":     [0.5]         * n,
        "median_return_pct":  [0.4]         * n,
        "volatility_pct":     [1.2]         * n,
        "sample_size":        [3]           * n,
        "win_rate_pct":       [60.0]        * n,
    })


# ─────────────────────────────────────────────────────────────────────────────
# validate_enriched — happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_enriched_passes_valid_data():
    validate_enriched(_valid_enriched())   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# validate_enriched — failure cases
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_enriched_raises_on_missing_column():
    df = _valid_enriched().drop(columns=["cycle_phase"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_enriched(df)


def test_validate_enriched_raises_on_negative_days_since_halving():
    df = _valid_enriched()
    df["days_since_halving"] = [-1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="negative"):
        validate_enriched(df)


def test_validate_enriched_raises_on_cycle_month_zero():
    df = _valid_enriched()
    df["cycle_month"] = [0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="cycle_month out of range"):
        validate_enriched(df)


def test_validate_enriched_raises_on_cycle_month_above_48():
    df = _valid_enriched()
    df["cycle_month"] = [49.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="cycle_month out of range"):
        validate_enriched(df)


def test_validate_enriched_raises_on_invalid_cycle_phase():
    df = _valid_enriched()
    df["cycle_phase"] = ["BULLISH", "EXPANSION", "EXPANSION"]
    with pytest.raises(ValueError, match="cycle_phase has unexpected values"):
        validate_enriched(df)


def test_validate_enriched_raises_on_duplicate_index():
    df = _valid_enriched()
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_enriched(df)


def test_validate_enriched_ignores_nan_cycle_columns():
    # Pre-halving rows have NaN — validator must not flag them
    df = _valid_enriched()
    df.loc[df.index[0], "days_since_halving"] = np.nan
    df.loc[df.index[0], "cycle_month"]        = np.nan
    df.loc[df.index[0], "cycle_phase"]        = None
    validate_enriched(df)   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# validate_profile — happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_profile_passes_valid_data():
    validate_profile(_valid_profile())   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# validate_profile — failure cases
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_profile_raises_on_missing_column():
    df = _valid_profile().drop(columns=["win_rate_pct"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_profile(df)


def test_validate_profile_raises_on_sample_size_zero():
    df = _valid_profile()
    df.loc[0, "sample_size"] = 0
    with pytest.raises(ValueError, match="sample_size < 1"):
        validate_profile(df)


def test_validate_profile_raises_on_win_rate_above_100():
    df = _valid_profile()
    df.loc[0, "win_rate_pct"] = 101.0
    with pytest.raises(ValueError, match="win_rate_pct out of range"):
        validate_profile(df)


def test_validate_profile_raises_on_win_rate_negative():
    df = _valid_profile()
    df.loc[0, "win_rate_pct"] = -1.0
    with pytest.raises(ValueError, match="win_rate_pct out of range"):
        validate_profile(df)


def test_validate_profile_raises_on_duplicate_key():
    df = _valid_profile()
    # Same cycle_month + cycle_phase + calendar_month + day_of_week → duplicate
    df = pd.concat([df, df.iloc[[0]]]).reset_index(drop=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_profile(df)


def test_validate_profile_raises_on_invalid_cycle_phase():
    df = _valid_profile()
    df.loc[0, "cycle_phase"] = "SIDEWAYS"
    with pytest.raises(ValueError, match="cycle_phase has unexpected values"):
        validate_profile(df)
