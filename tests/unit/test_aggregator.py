import pytest
import numpy as np
import pandas as pd

from analysis.aggregator import SeasonalAggregator


def _make_enriched(rows):
    """
    rows: list of (return_pct, days_since, cycle_month, cycle_phase, cal_month, dow)
    Calendar label columns are derived automatically.
    """
    month_map = {5: "May", 6: "Jun"}
    day_map   = {0: "Monday", 1: "Tuesday", 2: "Wednesday"}

    data = {
        "price":              [100.0] * len(rows),
        "daily_return_pct":   [r[0] for r in rows],
        "days_since_halving": [r[1] for r in rows],
        "cycle_month":        [r[2] for r in rows],
        "cycle_phase":        [r[3] for r in rows],
        "calendar_year":      [2024]  * len(rows),
        "calendar_month":     [r[4] for r in rows],
        "month_name":         [month_map.get(r[4], "Jan") for r in rows],
        "day_of_week":        [r[5] for r in rows],
        "day_name":           [day_map.get(r[5], "Thursday") for r in rows],
    }
    idx = pd.date_range("2024-04-21", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def base_enriched():
    """
    3 clean groups + 2 pre-halving rows (NaN) that must be dropped.

    Group A  (cycle_month=1, cal_month=5, dow=0):  returns [2.0, -2.0]
    Group B  (cycle_month=1, cal_month=5, dow=1):  returns [3.0,  1.0]
    Group C  (cycle_month=1, cal_month=5, dow=2):  returns [-1.0, -3.0]
    Pre-halv (NaN days_since):                     dropped silently
    """
    rows = [
        # return,  days_since, c_month, phase,          cal_month, dow
        ( 2.0,     1.0,        1.0,     "EXPANSION",    5,         0),
        (-2.0,     2.0,        1.0,     "EXPANSION",    5,         0),
        ( 3.0,     3.0,        1.0,     "EXPANSION",    5,         1),
        ( 1.0,     4.0,        1.0,     "EXPANSION",    5,         1),
        (-1.0,     5.0,        1.0,     "EXPANSION",    5,         2),
        (-3.0,     6.0,        1.0,     "EXPANSION",    5,         2),
        (np.nan,   np.nan,     np.nan,  None,           1,         0),  # pre-halving
        (np.nan,   np.nan,     np.nan,  None,           1,         1),  # pre-halving
    ]
    return _make_enriched(rows)


# ── pre-halving rows are excluded ─────────────────────────────────────────────

def test_pre_halving_rows_dropped(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    assert profile["sample_size"].sum() == 6   # 6 valid rows, 2 NaN dropped


# ── sample_size is accurate ───────────────────────────────────────────────────

def test_sample_size(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    assert (profile["sample_size"] == 2).all()


# ── win_rate: all negative → 0, all positive → 100, mixed → 50 ───────────────

def test_win_rate_all_negative(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    row = profile[(profile["day_of_week"] == 2)]   # Group C: [-1, -3]
    assert row["win_rate_pct"].iloc[0] == 0.0


def test_win_rate_all_positive(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    row = profile[(profile["day_of_week"] == 1)]   # Group B: [3, 1]
    assert row["win_rate_pct"].iloc[0] == 100.0


def test_win_rate_mixed(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    row = profile[(profile["day_of_week"] == 0)]   # Group A: [2, -2]
    assert row["win_rate_pct"].iloc[0] == 50.0


# ── aggregation math ──────────────────────────────────────────────────────────

def test_avg_return(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    row = profile[profile["day_of_week"] == 1]     # Group B: [3, 1] → avg=2.0
    assert row["avg_return_pct"].iloc[0] == pytest.approx(2.0, abs=1e-3)


def test_median_return(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    row = profile[profile["day_of_week"] == 2]     # Group C: [-1, -3] → median=-2.0
    assert row["median_return_pct"].iloc[0] == pytest.approx(-2.0, abs=1e-3)


def test_avg_return_mixed_group_is_zero(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    row = profile[profile["day_of_week"] == 0]     # Group A: [2, -2] → avg=0.0
    assert row["avg_return_pct"].iloc[0] == pytest.approx(0.0, abs=1e-3)


# ── no duplicate dimension combinations ──────────────────────────────────────

def test_no_duplicate_keys(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    dupes = profile.duplicated(subset=["cycle_month", "calendar_month", "day_of_week"])
    assert not dupes.any()


# ── win_rate is never NaN even when win_count merge produces NaN ──────────────

def test_win_rate_not_nan_when_all_negative(base_enriched):
    profile = SeasonalAggregator().build_profile(base_enriched)
    assert not profile["win_rate_pct"].isna().any()
