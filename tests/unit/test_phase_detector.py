"""
Unit tests for analysis.phase_detector

All tests use synthetic price arrays — no S3 or external data required.
Each test targets one specific algorithmic behaviour.

Ground-truth tests against real BTC history live in
tests/contract/test_cycle_boundaries.py.
"""
import numpy as np
import pytest

from analysis.phase_detector import find_confirmed_peak, find_trough_after_peak


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ramp(start: float, end: float, n: int) -> np.ndarray:
    """Linearly interpolated price array."""
    return np.linspace(start, end, n)


def _flat(price: float, n: int) -> np.ndarray:
    return np.full(n, price)


def _build(segments: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (prices, days) from a list of (start, end, n_days) segments.
    Days are zero-indexed and continuous across segments.
    """
    price_parts = [_ramp(s, e, n) for s, e, n in segments]
    prices = np.concatenate(price_parts)
    days   = np.arange(len(prices), dtype=float)
    return prices, days


# ─── find_confirmed_peak ──────────────────────────────────────────────────────

class TestFindConfirmedPeak:

    # ── basic correctness ─────────────────────────────────────────────────────

    def test_clean_single_peak_is_detected(self):
        """
        Price rises to 100, drops 70 % and stays there.
        Peak must be identified near day 99 (top of the ramp).
        """
        prices, days = _build([
            (10, 100, 100),   # days   0-99: rise
            (100, 30, 50),    # days 100-149: drop (70 %)
            (30,  30, 150),   # days 150-299: flat low
        ])
        peak_day, peak_price = find_confirmed_peak(
            prices, days,
            drawdown_threshold=0.40,
            confirmation_days=60,
            cycle_id="clean_peak",
        )
        assert peak_price == pytest.approx(100, rel=0.05)
        assert 90 <= peak_day <= 110

    def test_peak_day_is_int(self):
        prices, days = _build([(10, 100, 100), (100, 20, 300)])
        peak_day, _ = find_confirmed_peak(prices, days, cycle_id="type_check")
        assert isinstance(peak_day, int)

    def test_peak_price_gt_first_price(self):
        prices, days = _build([(50, 500, 300), (500, 100, 200), (100, 100, 100)])
        _, peak_price = find_confirmed_peak(prices, days, cycle_id="price_gt_first")
        assert peak_price > prices[0]

    # ── double top: must return the SECOND (higher) peak ─────────────────────

    def test_double_top_returns_second_higher_peak(self):
        """
        First top 64, drops 56 %, recovers and rises to 69.
        69 then confirmed by sustained 70 % drawdown.
        Algorithm must lock in the SECOND peak at 69, not the first at 64.
        """
        prices, days = _build([
            (10,  64, 100),   # rise to first top (64) at day ~99
            (64,  28,  40),   # drop 56 %
            (28,  69,  80),   # recovery + new ATH at day ~219
            (69,  20,  60),   # confirmed drop 71 %
            (20,  20, 150),   # sustained low
        ])
        peak_day, peak_price = find_confirmed_peak(
            prices, days,
            drawdown_threshold=0.40,
            confirmation_days=60,
            cycle_id="double_top",
        )
        assert peak_price == pytest.approx(69, rel=0.05), (
            f"Expected second peak ~69, got {peak_price:.2f}"
        )
        assert peak_day >= 170, (
            f"Peak day {peak_day} is too early — first top was not rejected"
        )

    def test_early_candidate_rejected_then_same_peak_confirmed_later(self):
        """
        Price drops from peak, recovers within the window (early candidate rejected),
        then enters a sustained bear market.  The algorithm must reject the early
        confirmation but still lock in the SAME peak (day ~99) once the sustained
        decline is underway.

        This verifies that a window-based rejection does not corrupt the state:
        running_max stays at 100, and the confirmation eventually fires correctly
        when the final crash has no recovery.
        """
        prices, days = _build([
            (10,  100, 100),  # rise to 100 at day ~99
            (100,  50,  30),  # quick 50 % drop — window will detect recovery → rejected
            (50,  100,  60),  # full recovery (price back to 100, same ATH)
            (100,  20, 100),  # final sustained crash — no recovery → confirmed
            (20,   20, 200),  # sustained low
        ])
        peak_day, peak_price = find_confirmed_peak(
            prices, days,
            drawdown_threshold=0.40,
            confirmation_days=80,
            cycle_id="reject_then_confirm",
        )
        # Peak is day 99 ($100) — confirmed after the second, sustained crash
        assert peak_price == pytest.approx(100, rel=0.05), (
            f"Expected peak price ~100, got {peak_price:.2f}"
        )
        assert peak_day <= 105, (
            f"Expected peak near day 99, got {peak_day}"
        )

    # ── late-cycle new ATH triggers fallback ─────────────────────────────────

    def test_late_cycle_ath_triggers_fallback_to_correct_peak(self):
        """
        Simulates the 2020 BTC cycle:
          - Bull run to 69 at day ~550, confirmed drawdown to 20
          - Then a late-cycle new ATH to 73 at day ~1400 (just before next halving)
          - The 73-day ATH invalidates the 69 confirmation
          - Fallback (first 800 days) must find 69 at day ~550, not 73

        This is the exact failure mode of the old price.idxmax() approach.
        """
        prices, days = _build([
            (10,   69, 550),   # days   0-549: bull run to 69
            (69,   20, 200),   # days 550-749: confirmed crash
            (20,   20, 300),   # days 750-1049: flat bottom
            (20,   73, 200),   # days 1050-1249: late-cycle run to new ATH
            (73,   70,  10),   # days 1250-1259: small move (cycle ends)
        ])
        peak_day, peak_price = find_confirmed_peak(
            prices, days,
            drawdown_threshold=0.40,
            confirmation_days=120,
            cycle_id="late_ath_fallback",
        )
        assert peak_price == pytest.approx(69, rel=0.05), (
            f"Fallback returned wrong price {peak_price:.2f} — "
            f"should have found 69 in the first 800 days"
        )
        assert peak_day <= 600, (
            f"Peak day {peak_day} is too late — late-cycle ATH was not handled"
        )

    # ── insufficient forward data → fallback ─────────────────────────────────

    def test_insufficient_forward_data_activates_fallback(self):
        """
        Price drops 50 % but cycle ends 10 days later.
        No confirmation window available → fallback to highest in first 800 days.
        """
        prices, days = _build([
            (10, 100, 100),   # rise to 100
            (100, 40,  10),   # drop 60 % — but only 10 forward days, can't confirm
        ])
        peak_day, peak_price = find_confirmed_peak(
            prices, days,
            drawdown_threshold=0.40,
            confirmation_days=90,
            cycle_id="no_forward_data",
        )
        # Fallback: max in first 800 days = 100 at day 99
        assert peak_price == pytest.approx(100, rel=0.05)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            find_confirmed_peak(np.array([]), np.array([]), cycle_id="empty")

    # ── monotone price (no drawdown) → fallback ───────────────────────────────

    def test_monotone_increasing_uses_fallback(self):
        """
        Price only rises — no drawdown ever fires.
        Fallback: highest in first 800 days.
        """
        prices = np.linspace(100, 500, 400)
        days   = np.arange(400, dtype=float)
        peak_day, peak_price = find_confirmed_peak(
            prices, days,
            drawdown_threshold=0.40,
            confirmation_days=60,
            cycle_id="monotone",
        )
        # Fallback covers all 400 days (< 800), so it finds the true max at day 399
        assert peak_day <= 399
        assert peak_price == pytest.approx(500, rel=0.05)


# ─── find_trough_after_peak ───────────────────────────────────────────────────

class TestFindTroughAfterPeak:

    def test_trough_is_strictly_after_peak(self):
        prices, days = _build([(10, 100, 100), (100, 5, 200)])
        trough_day, _ = find_trough_after_peak(prices, days, peak_day=99, cycle_id="after_peak")
        assert trough_day > 99

    def test_trough_price_below_peak_price(self):
        prices, days = _build([(10, 100, 100), (100, 5, 200)])
        _, trough_price = find_trough_after_peak(prices, days, peak_day=99, cycle_id="price_below")
        assert trough_price < 100

    def test_trough_is_lowest_post_peak_value(self):
        """Trough must be the MINIMUM, not just any post-peak value."""
        # Post-peak: drops to 5, bounces to 30, drops to 3, ends at 10
        prices = np.concatenate([
            _ramp(10, 100, 100),
            np.array([80, 60, 40, 20, 5, 15, 30, 10, 3, 8, 10]),
        ])
        days = np.arange(len(prices), dtype=float)
        _, trough_price = find_trough_after_peak(prices, days, peak_day=99, cycle_id="min_check")
        assert trough_price == pytest.approx(3, abs=0.1)

    def test_trough_day_is_int(self):
        prices, days = _build([(10, 100, 100), (100, 20, 200)])
        trough_day, _ = find_trough_after_peak(prices, days, peak_day=99, cycle_id="type_check")
        assert isinstance(trough_day, int)

    def test_no_data_after_peak_returns_peak_day(self):
        """When the cycle ends at the peak, trough falls back to the peak day."""
        prices = _ramp(10, 100, 100)
        days   = np.arange(100, dtype=float)
        trough_day, _ = find_trough_after_peak(prices, days, peak_day=99, cycle_id="no_post")
        assert trough_day == 99


# ─── combined invariants (peak + trough together) ────────────────────────────

class TestPeakTroughInvariants:

    @pytest.fixture
    def simple_cycle(self):
        """Rise → crash → flat → small recovery."""
        prices, days = _build([
            (100, 10_000, 400),   # EXPANSION
            (10_000,  3_000, 300), # DISTRIBUTION
            (3_000,   5_000, 400), # ACCUMULATION
        ])
        return prices, days

    def test_peak_before_trough(self, simple_cycle):
        prices, days = simple_cycle
        peak_day, _ = find_confirmed_peak(prices, days, cycle_id="invariant")
        trough_day, _ = find_trough_after_peak(prices, days, peak_day=peak_day, cycle_id="invariant")
        assert peak_day < trough_day

    def test_peak_price_gt_trough_price(self, simple_cycle):
        prices, days = simple_cycle
        peak_day, peak_price = find_confirmed_peak(prices, days, cycle_id="inv_price")
        _, trough_price = find_trough_after_peak(prices, days, peak_day=peak_day, cycle_id="inv_price")
        assert peak_price > trough_price

    def test_phase_durations_are_positive(self, simple_cycle):
        prices, days = simple_cycle
        peak_day, _ = find_confirmed_peak(prices, days, cycle_id="inv_dur")
        trough_day, _ = find_trough_after_peak(prices, days, peak_day=peak_day, cycle_id="inv_dur")
        max_day = int(days[-1])
        assert peak_day   > 0,         "EXPANSION must be > 0 days"
        assert trough_day > peak_day,  "DISTRIBUTION must be > 0 days"
        assert max_day    > trough_day,"ACCUMULATION must be > 0 days"
