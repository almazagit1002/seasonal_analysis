"""
Contract / ground-truth tests for compute_cycle_boundaries.

These tests assert that the computed phase boundaries match known
historical BTC milestones.  They require live S3 access to load the
enriched price table — they are automatically skipped when credentials
are unavailable.

Known reference points used as assertions
------------------------------------------
Cycle 2016
  ATH  : ~$19,891  on Dec 17 2017  ≈ day 526 post-halving
  Trough: ~$3,122  on Dec 15 2018  ≈ day 890 post-halving

Cycle 2020
  ATH  : ~$68,789  on Nov 10 2021  ≈ day 548 post-halving
         (NOT the Mar 2024 pre-halving ATH of ~$73 750 at day ~1404)
  Trough: ~$15,599 on Nov 21 2022  ≈ day 1108 post-halving
"""
import pytest
import pandas as pd

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def enriched_df():
    """Load enriched table from S3; skip this module if unavailable."""
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from utils.s3_utils import read_s3_file
        return read_s3_file("btc-seasonal-analysis-faadab22", "btc_enriched.parquet")
    except Exception as exc:
        pytest.skip(f"S3 data unavailable ({exc})")


@pytest.fixture(scope="module")
def boundaries(enriched_df):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from analysis.phase_detector import assign_cycle_id, compute_cycle_boundaries
    from utils.config_reader import ConfigReader
    cfg       = ConfigReader("config")
    cycle_cfg = cfg.get("bitcoin_cycle")
    df = assign_cycle_id(enriched_df, cycle_cfg["halvings"])
    return compute_cycle_boundaries(
        df,
        cycle_cfg["halvings"],
        cycle_cfg["peak_detection"],
        cycle_cfg["trough_detection"],
    )


# ── 2016 cycle — the clean reference cycle ───────────────────────────────────

class TestCycle2016:
    """
    2016 is the cleanest cycle: single unambiguous ATH, clear bear market.
    If these fail the algorithm is broken at a fundamental level.
    """

    def test_peak_day_within_range(self, boundaries):
        """ATH was Dec 17, 2017 ≈ day 526.  Allow ±45 days."""
        b = boundaries["2016"]
        assert 480 <= b["peak_day"] <= 570, (
            f"2016 peak day {b['peak_day']} not in expected range [480, 570] "
            f"(historical: ~526)"
        )

    def test_peak_price_in_range(self, boundaries):
        """ATH was ~$19,891.  Allow ±20 %."""
        b = boundaries["2016"]
        assert 15_000 <= b["peak_price"] <= 24_000, (
            f"2016 peak price ${b['peak_price']:,.0f} out of range "
            f"(historical: ~$19,891)"
        )

    def test_trough_day_within_range(self, boundaries):
        """Cycle low was Dec 15, 2018 ≈ day 890.  Allow ±50 days."""
        b = boundaries["2016"]
        assert 840 <= b["trough_day"] <= 940, (
            f"2016 trough day {b['trough_day']} not in expected range [840, 940] "
            f"(historical: ~890)"
        )

    def test_trough_price_in_range(self, boundaries):
        """Cycle low was ~$3,122.  Allow ±40 % (data quality variation)."""
        b = boundaries["2016"]
        assert 1_800 <= b["trough_price"] <= 4_400, (
            f"2016 trough price ${b['trough_price']:,.0f} out of range "
            f"(historical: ~$3,122)"
        )

    def test_distribution_duration(self, boundaries):
        """DISTRIBUTION ran ~364 days (Jul 2018 – Dec 2018).  Minimum 200 days."""
        b = boundaries["2016"]
        dur = b["trough_day"] - b["peak_day"]
        assert dur >= 200, (
            f"2016 DISTRIBUTION only {dur} days — "
            f"expected at least 200 (historical: ~364)"
        )


# ── 2020 cycle — the double-top / late-ATH problem ───────────────────────────

class TestCycle2020:
    """
    2020 cycle has two traps for a naive price.idxmax() approach:
      1. Double top (Apr 2021 $64K and Nov 2021 $69K)
      2. A new pre-halving ATH in Mar 2024 (~$73 750, day ~1404)
    Both must be handled correctly.
    """

    def test_peak_day_is_not_late_cycle_ath(self, boundaries):
        """
        The Mar 2024 pre-halving ATH was at day ~1404.
        The true bull market top (Nov 2021) was at day ~548.
        peak_day must be < 700.
        """
        b = boundaries["2020"]
        assert b["peak_day"] < 700, (
            f"2020 peak day {b['peak_day']} ≥ 700 — "
            f"algorithm picked the late-cycle ATH (day ~1404) instead of "
            f"the Nov 2021 bull top (~day 548)"
        )

    def test_peak_day_within_range(self, boundaries):
        """Nov 2021 ATH ≈ day 548.  Allow ±60 days for the double-top region."""
        b = boundaries["2020"]
        assert 330 <= b["peak_day"] <= 650, (
            f"2020 peak day {b['peak_day']} not in [330, 650] "
            f"(historical: ~548)"
        )

    def test_peak_price_in_range(self, boundaries):
        """ATH was ~$68,789.  Allow ±25 %."""
        b = boundaries["2020"]
        assert 45_000 <= b["peak_price"] <= 90_000, (
            f"2020 peak price ${b['peak_price']:,.0f} out of range "
            f"(historical: ~$68,789)"
        )

    def test_trough_day_within_range(self, boundaries):
        """
        FTX-crash low was ~Nov 21 2022 ≈ day 924 from the May 2020 halving.
        Allow ±60 days.
        """
        b = boundaries["2020"]
        assert 860 <= b["trough_day"] <= 990, (
            f"2020 trough day {b['trough_day']} not in [860, 990] "
            f"(historical: ~924, Nov 2022 FTX low)"
        )

    def test_distribution_duration_plausible(self, boundaries):
        """Bear market from Nov 2021 to Nov 2022 was ~365 days.  Minimum 200."""
        b = boundaries["2020"]
        dur = b["trough_day"] - b["peak_day"]
        assert dur >= 200, (
            f"2020 DISTRIBUTION only {dur} days — "
            f"expected at least 200 (historical: ~560)"
        )


# ── cross-cycle invariants ─────────────────────────────────────────────────────

class TestCrossCycleInvariants:
    """These must hold for every cycle regardless of exact day counts."""

    @pytest.mark.parametrize("cycle_id", ["2012", "2016", "2020", "2024"])
    def test_peak_before_trough(self, boundaries, cycle_id):
        if cycle_id not in boundaries:
            pytest.skip(f"cycle {cycle_id} not in boundaries")
        b = boundaries[cycle_id]
        assert b["peak_day"] < b["trough_day"], (
            f"[{cycle_id}] peak_day {b['peak_day']} >= trough_day {b['trough_day']}"
        )

    @pytest.mark.parametrize("cycle_id", ["2012", "2016", "2020", "2024"])
    def test_peak_price_gt_trough_price(self, boundaries, cycle_id):
        if cycle_id not in boundaries:
            pytest.skip(f"cycle {cycle_id} not in boundaries")
        b = boundaries[cycle_id]
        assert b["peak_price"] > b["trough_price"], (
            f"[{cycle_id}] peak ${b['peak_price']:,.0f} <= "
            f"trough ${b['trough_price']:,.0f}"
        )

    @pytest.mark.parametrize("cycle_id", ["2016", "2020"])
    def test_distribution_at_least_60_days(self, boundaries, cycle_id):
        """
        A bear market shorter than 60 days is a data or algorithm error.
        2012 excluded (truncated dataset).
        """
        b = boundaries[cycle_id]
        dur = b["trough_day"] - b["peak_day"]
        assert dur >= 60, (
            f"[{cycle_id}] DISTRIBUTION = {dur} days — "
            f"implausibly short (minimum expected: 60)"
        )

    @pytest.mark.parametrize("cycle_id", ["2016", "2020"])
    def test_accumulation_at_least_90_days(self, boundaries, cycle_id):
        """
        An accumulation phase shorter than 90 days is implausible.
        2012 excluded (truncated dataset).
        """
        b = boundaries[cycle_id]
        dur = b["max_day"] - b["trough_day"]
        assert dur >= 90, (
            f"[{cycle_id}] ACCUMULATION = {dur} days — "
            f"implausibly short (minimum expected: 90)"
        )

    @pytest.mark.parametrize("cycle_id", ["2016", "2020"])
    def test_expansion_at_least_100_days(self, boundaries, cycle_id):
        """Bull runs shorter than 100 days from a halving are implausible."""
        b = boundaries[cycle_id]
        assert b["peak_day"] >= 100, (
            f"[{cycle_id}] EXPANSION = {b['peak_day']} days — "
            f"implausibly short (minimum expected: 100)"
        )
