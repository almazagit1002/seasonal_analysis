import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from analysis.enricher import DataEnricher

_HALVINGS_CFG = [
    {"label": "2012", "date": "2012-11-28", "tolerance_days": 0},
    {"label": "2016", "date": "2016-07-09", "tolerance_days": 0},
    {"label": "2020", "date": "2020-05-11", "tolerance_days": 0},
    {"label": "2024", "date": "2024-04-20", "tolerance_days": 0},
    {"label": "2028", "date": "2028-03-01", "tolerance_days": 21},
]

_HALVINGS_CFG_NO_FUTURE = [
    {"label": "2012", "date": "2012-11-28", "tolerance_days": 0},
    {"label": "2016", "date": "2016-07-09", "tolerance_days": 0},
    {"label": "2020", "date": "2020-05-11", "tolerance_days": 0},
]


@pytest.fixture
def enricher():
    mock_cfg = MagicMock()
    mock_cfg.get.return_value = {"halvings": _HALVINGS_CFG}
    with patch("analysis.enricher.ConfigReader", return_value=mock_cfg):
        yield DataEnricher()


@pytest.fixture
def cap_enricher():
    """No future halvings — allows dates far from 2020 to stay in that cycle."""
    mock_cfg = MagicMock()
    mock_cfg.get.return_value = {"halvings": _HALVINGS_CFG_NO_FUTURE}
    with patch("analysis.enricher.ConfigReader", return_value=mock_cfg):
        yield DataEnricher()


def _df(dates, prices=None):
    if prices is None:
        prices = [float(10_000 + i * 100) for i in range(len(dates))]
    return pd.DataFrame({"price": prices}, index=pd.DatetimeIndex(dates, tz="UTC"))


# ── days_since_halving ────────────────────────────────────────────────────────

def test_days_since_halving_on_halving_day(enricher):
    df = _df(["2024-04-20"])
    result = enricher.enrich(df, "price")
    assert result["days_since_halving"].iloc[0] == 0.0


def test_days_since_halving_one_day_after(enricher):
    df = _df(["2024-04-21"])
    result = enricher.enrich(df, "price")
    assert result["days_since_halving"].iloc[0] == 1.0


def test_days_since_halving_thirty_days_after(enricher):
    df = _df(["2024-05-20"])
    result = enricher.enrich(df, "price")
    assert result["days_since_halving"].iloc[0] == 30.0


def test_days_since_halving_is_nan_before_first_halving(enricher):
    df = _df(["2011-01-01"])
    result = enricher.enrich(df, "price")
    assert pd.isna(result["days_since_halving"].iloc[0])


# ── cycle_phase absent from enrich output ─────────────────────────────────────

def test_enrich_does_not_produce_cycle_phase(enricher):
    """cycle_phase is stamped later by CycleAnalyzer, not by DataEnricher."""
    df = _df(["2024-05-01"])
    result = enricher.enrich(df, "price")
    assert "cycle_phase" not in result.columns


# ── cycle_month ───────────────────────────────────────────────────────────────

_HALVING = pd.Timestamp("2024-04-20", tz="UTC")


@pytest.mark.parametrize("offset, expected_month", [
    (0,    1),
    (29,   1),
    (30,   2),
    (1409, 47),
    (1410, 48),
])
def test_cycle_month_boundaries(enricher, offset, expected_month):
    date = _HALVING + pd.Timedelta(days=offset)
    df = _df([date])
    result = enricher.enrich(df, "price")
    assert result["cycle_month"].iloc[0] == expected_month


_HALVING_2020 = pd.Timestamp("2020-05-11", tz="UTC")


@pytest.mark.parametrize("offset, expected_month", [
    (1440, 48),
    (1500, 48),
])
def test_cycle_month_cap(cap_enricher, offset, expected_month):
    date = _HALVING_2020 + pd.Timedelta(days=offset)
    df = _df([date])
    result = cap_enricher.enrich(df, "price")
    assert result["cycle_month"].iloc[0] == expected_month


def test_cycle_month_is_nan_before_first_halving(enricher):
    df = _df(["2011-01-01"])
    result = enricher.enrich(df, "price")
    assert pd.isna(result["cycle_month"].iloc[0])


# ── calendar columns ──────────────────────────────────────────────────────────

def test_calendar_columns(enricher):
    df = _df(["2024-04-22"])
    result = enricher.enrich(df, "price")
    assert result["calendar_year"].iloc[0]  == 2024
    assert result["calendar_month"].iloc[0] == 4
    assert result["month_name"].iloc[0]     == "Apr"
    assert result["day_of_week"].iloc[0]    == 0
    assert result["day_name"].iloc[0]       == "Monday"


# ── daily_return_pct ──────────────────────────────────────────────────────────

def test_daily_return_pct(enricher):
    df = _df(["2024-04-21", "2024-04-22"], prices=[100.0, 110.0])
    result = enricher.enrich(df, "price")
    assert pd.isna(result["daily_return_pct"].iloc[0])
    assert result["daily_return_pct"].iloc[1] == pytest.approx(10.0)


# ── timezone safety ───────────────────────────────────────────────────────────

def test_timezone_aware_index_does_not_raise(enricher):
    df = _df(["2024-04-21"])
    enricher.enrich(df, "price")


# ── future halving not used as reference ─────────────────────────────────────

def test_future_halving_excluded(enricher):
    df = _df(["2027-01-01"])
    result = enricher.enrich(df, "price")
    expected_days = (pd.Timestamp("2027-01-01") - pd.Timestamp("2024-04-20")).days
    assert result["days_since_halving"].iloc[0] == float(expected_days)
