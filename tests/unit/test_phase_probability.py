"""
Unit tests for analysis.phase_probability

Tests cover the pure functions and class behaviour without S3 or real data.
The PhaseProbabilityEstimator is pre-fitted by directly setting its internal
state so tests don't depend on a real ConfigReader or full enriched table.
"""
import numpy as np
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from analysis.phase_probability import (
    PhaseProbabilityEstimator,
    _weighted_mean_sigma,
    _time_prior,
    _price_posterior,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

_HALVINGS_CFG = [
    {"label": "2016", "date": "2016-07-09", "tolerance_days": 0, "data_quality_weight": 1.0},
    {"label": "2020", "date": "2020-05-11", "tolerance_days": 0, "data_quality_weight": 1.0},
    {"label": "2024", "date": "2024-04-20", "tolerance_days": 0, "data_quality_weight": 1.0},
]

_DRAWDOWN_STATS = {
    "EXPANSION":    {"mu": 0.05, "sigma": 0.08},
    "DISTRIBUTION": {"mu": 0.65, "sigma": 0.12},
    "ACCUMULATION": {"mu": 0.38, "sigma": 0.10},
}

# Representative values derived from 2016+2020 cycles
_MU_PEAK      = 537.0
_SIGMA_PEAK   = 11.0
_MU_TROUGH    = 902.0   # 537 + 365
_SIGMA_TROUGH = 11.18   # sqrt(11² + 2²)


@pytest.fixture
def estimator():
    """Pre-fitted estimator with internal state set directly."""
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, default=None: {
        "bitcoin_cycle": {"halvings": _HALVINGS_CFG},
        "estimation":    {"min_cycles_required": 2, "prior_only_threshold": 0.05},
    }.get(key, default)

    with patch("analysis.phase_probability.ConfigReader", return_value=mock_cfg):
        est = PhaseProbabilityEstimator()

    est._fitted          = True
    est._n_complete      = 2
    est._duration_stats  = {
        "EXPANSION":    {"mu": _MU_PEAK,        "sigma": _SIGMA_PEAK, "n": 2},
        "DISTRIBUTION": {"mu": 365.0,            "sigma": 2.0,        "n": 2},
        "ACCUMULATION": {"mu": 519.0,            "sigma": 10.0,       "n": 2},
    }
    est._boundary_stats  = {
        "mu_peak":      _MU_PEAK,
        "sigma_peak":   _SIGMA_PEAK,
        "mu_trough":    _MU_TROUGH,
        "sigma_trough": _SIGMA_TROUGH,
    }
    est._drawdown_stats  = _DRAWDOWN_STATS
    return est


# ── _weighted_mean_sigma ──────────────────────────────────────────────────────

class TestWeightedMeanSigma:

    def test_equal_weights_match_numpy(self):
        vals = np.array([100.0, 200.0, 300.0])
        wts  = np.ones(3)
        mu, sigma = _weighted_mean_sigma(vals, wts)
        assert mu    == pytest.approx(np.mean(vals),   rel=1e-6)
        assert sigma == pytest.approx(np.std(vals),    rel=1e-6)

    def test_zero_weight_observation_excluded(self):
        """An observation with weight 0 should not affect the result."""
        vals_full    = np.array([10.0, 500.0, 510.0])
        wts_full     = np.array([1.0,  1.0,   1.0])
        vals_partial = np.array([500.0, 510.0])
        wts_partial  = np.ones(2)

        mu_z, _ = _weighted_mean_sigma(np.array([10.0, 500.0, 510.0]), np.array([0.0, 1.0, 1.0]))
        mu_p, _ = _weighted_mean_sigma(vals_partial, wts_partial)
        assert mu_z == pytest.approx(mu_p, rel=1e-6)

    def test_half_weight_pulls_mean_toward_high_weight_obs(self):
        """Low-weight obs should have less pull on the mean."""
        vals = np.array([100.0, 500.0])
        wts_equal = np.array([1.0, 1.0])
        wts_biased = np.array([0.25, 1.0])
        mu_eq, _ = _weighted_mean_sigma(vals, wts_equal)
        mu_bi, _ = _weighted_mean_sigma(vals, wts_biased)
        assert mu_bi > mu_eq  # biased toward 500

    def test_single_observation_sigma_is_zero(self):
        mu, sigma = _weighted_mean_sigma(np.array([42.0]), np.array([1.0]))
        assert mu    == pytest.approx(42.0)
        assert sigma == pytest.approx(0.0, abs=1e-10)

    def test_returns_floats(self):
        mu, sigma = _weighted_mean_sigma(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        assert isinstance(mu, float)
        assert isinstance(sigma, float)


# ── _time_prior ───────────────────────────────────────────────────────────────

class TestTimePrior:

    def test_sums_to_one(self):
        for day in [0, 200, 537, 700, 902, 1200]:
            p = _time_prior(day, _MU_PEAK, _SIGMA_PEAK, _MU_TROUGH, _SIGMA_TROUGH)
            assert sum(p) == pytest.approx(1.0, abs=1e-9)

    def test_early_day_dominated_by_expansion(self):
        """Day 100 — well before expected peak at ~537."""
        p = _time_prior(100, _MU_PEAK, _SIGMA_PEAK, _MU_TROUGH, _SIGMA_TROUGH)
        assert p[0] > 0.99   # EXPANSION dominates

    def test_mid_day_dominated_by_distribution(self):
        """Day 720 — past peak (~537) but before trough (~902)."""
        p = _time_prior(720, _MU_PEAK, _SIGMA_PEAK, _MU_TROUGH, _SIGMA_TROUGH)
        assert p[1] > 0.50   # DISTRIBUTION dominates

    def test_late_day_dominated_by_accumulation(self):
        """Day 1100 — well past expected trough at ~902."""
        p = _time_prior(1100, _MU_PEAK, _SIGMA_PEAK, _MU_TROUGH, _SIGMA_TROUGH)
        assert p[2] > 0.99   # ACCUMULATION dominates

    def test_all_probabilities_non_negative(self):
        for day in range(0, 1400, 50):
            p = _time_prior(day, _MU_PEAK, _SIGMA_PEAK, _MU_TROUGH, _SIGMA_TROUGH)
            assert all(v >= 0 for v in p)

    def test_monotone_expansion_probability_decreasing(self):
        """P(EXPANSION) should decrease monotonically as days increase."""
        days = list(range(0, 900, 30))
        p_exp = [
            _time_prior(d, _MU_PEAK, _SIGMA_PEAK, _MU_TROUGH, _SIGMA_TROUGH)[0]
            for d in days
        ]
        assert all(p_exp[i] >= p_exp[i + 1] for i in range(len(p_exp) - 1))

    def test_monotone_accumulation_probability_increasing(self):
        """P(ACCUMULATION) should increase monotonically as days increase."""
        days = list(range(500, 1400, 30))
        p_accum = [
            _time_prior(d, _MU_PEAK, _SIGMA_PEAK, _MU_TROUGH, _SIGMA_TROUGH)[2]
            for d in days
        ]
        assert all(p_accum[i] <= p_accum[i + 1] for i in range(len(p_accum) - 1))


# ── _price_posterior ──────────────────────────────────────────────────────────

class TestPricePosterior:

    def test_sums_to_one(self):
        prior = [0.1, 0.7, 0.2]
        post  = _price_posterior(prior, 0.60, _DRAWDOWN_STATS, prior_only_dd=0.05)
        assert sum(post) == pytest.approx(1.0, abs=1e-9)

    def test_high_drawdown_boosts_distribution(self):
        """70% drawdown is squarely in DISTRIBUTION territory."""
        prior = [1/3, 1/3, 1/3]
        post  = _price_posterior(prior, 0.70, _DRAWDOWN_STATS, prior_only_dd=0.05)
        assert post[1] > post[0]
        assert post[1] > post[2]

    def test_low_drawdown_returns_prior_unchanged(self):
        """Near-ATH (drawdown < threshold) should not change the prior."""
        prior = [0.8, 0.15, 0.05]
        post  = _price_posterior(prior, 0.02, _DRAWDOWN_STATS, prior_only_dd=0.05)
        assert post == pytest.approx(prior, abs=1e-9)

    def test_all_non_negative(self):
        prior = [0.2, 0.5, 0.3]
        post  = _price_posterior(prior, 0.50, _DRAWDOWN_STATS, prior_only_dd=0.05)
        assert all(v >= 0 for v in post)


# ── PhaseProbabilityEstimator.estimate ────────────────────────────────────────

class TestEstimate:

    def test_prior_sums_to_one(self, estimator):
        for day in [100, 400, 700, 1000]:
            r = estimator.estimate(day)
            total = (r["p_expansion_prior"] + r["p_distribution_prior"] +
                     r["p_accumulation_prior"])
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_posterior_sums_to_one_when_drawdown_provided(self, estimator):
        r = estimator.estimate(700, current_drawdown=0.65)
        total = (r["p_expansion_posterior"] + r["p_distribution_posterior"] +
                 r["p_accumulation_posterior"])
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_no_drawdown_leaves_posterior_nan(self, estimator):
        r = estimator.estimate(500)
        assert np.isnan(r["p_expansion_posterior"])
        assert np.isnan(r["p_distribution_posterior"])
        assert np.isnan(r["p_accumulation_posterior"])

    def test_dominant_phase_prior_early_is_expansion(self, estimator):
        r = estimator.estimate(50)
        assert r["dominant_phase_prior"] == "EXPANSION"

    def test_dominant_phase_prior_late_is_accumulation(self, estimator):
        r = estimator.estimate(1100)
        assert r["dominant_phase_prior"] == "ACCUMULATION"

    def test_signals_agree_when_consistent(self, estimator):
        """Day 700 with large drawdown → both signals say DISTRIBUTION."""
        r = estimator.estimate(700, current_drawdown=0.70)
        assert r["signals_agree"] is True

    def test_signals_disagree_when_inconsistent(self, estimator):
        """
        Day 530 is 0.6σ before the expected peak (μ=537, σ=11) so the time
        prior still says EXPANSION (~74%).  An 80% drawdown is 9σ above the
        typical EXPANSION drawdown mean (5%) and squarely in DISTRIBUTION
        territory, so the price likelihood flips the posterior to DISTRIBUTION.
        This is the intended "signals disagree" scenario.
        """
        r = estimator.estimate(530, current_drawdown=0.80)
        assert r["dominant_phase_prior"]     == "EXPANSION"
        assert r["dominant_phase_posterior"] == "DISTRIBUTION"
        assert r["signals_agree"] is False

    def test_raises_if_not_fitted(self):
        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "bitcoin_cycle": {"halvings": _HALVINGS_CFG},
            "estimation":    {"min_cycles_required": 2, "prior_only_threshold": 0.05},
        }.get(key, default)
        with patch("analysis.phase_probability.ConfigReader", return_value=mock_cfg):
            est = PhaseProbabilityEstimator()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            est.estimate(500)


# ── PhaseProbabilityEstimator.estimate_series ─────────────────────────────────

class TestEstimateSeries:

    @pytest.fixture
    def synthetic_enriched(self):
        """Minimal enriched DataFrame for the ongoing cycle (2024)."""
        n    = 100
        days = np.arange(n, dtype=float)
        # Price rises for 60 days then drops — simulates EXPANSION then DISTRIBUTION
        prices = np.concatenate([
            np.linspace(30_000, 60_000, 60),
            np.linspace(60_000, 25_000, 40),
        ])
        idx = pd.date_range("2024-04-20", periods=n, freq="D", tz="UTC")
        return pd.DataFrame({
            "price":             prices,
            "days_since_halving": days,
            "cycle_id":          "2024",
            "cycle_phase":       ["EXPANSION"] * 60 + ["DISTRIBUTION"] * 40,
            "daily_return_pct":  np.random.randn(n) * 0.5,
        }, index=idx)

    def test_returns_dataframe(self, estimator, synthetic_enriched):
        result = estimator.estimate_series(synthetic_enriched, "2024")
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_input(self, estimator, synthetic_enriched):
        result = estimator.estimate_series(synthetic_enriched, "2024")
        assert len(result) == len(synthetic_enriched)

    def test_prior_columns_present(self, estimator, synthetic_enriched):
        result = estimator.estimate_series(synthetic_enriched, "2024")
        for col in ["p_expansion_prior", "p_distribution_prior", "p_accumulation_prior"]:
            assert col in result.columns

    def test_prior_sums_to_one_every_row(self, estimator, synthetic_enriched):
        result = estimator.estimate_series(synthetic_enriched, "2024")
        totals = result[["p_expansion_prior", "p_distribution_prior",
                         "p_accumulation_prior"]].sum(axis=1)
        assert (totals.between(0.999, 1.001)).all()

    def test_posterior_sums_to_one_every_row(self, estimator, synthetic_enriched):
        result = estimator.estimate_series(synthetic_enriched, "2024")
        post_cols = ["p_expansion_posterior", "p_distribution_posterior",
                     "p_accumulation_posterior"]
        if result[post_cols].notna().all(axis=None):
            totals = result[post_cols].sum(axis=1)
            assert (totals.between(0.999, 1.001)).all()

    def test_drawdown_non_negative(self, estimator, synthetic_enriched):
        result = estimator.estimate_series(synthetic_enriched, "2024")
        assert (result["drawdown_from_ath"] >= 0).all()

    def test_raises_for_unknown_cycle(self, estimator, synthetic_enriched):
        with pytest.raises(ValueError, match="No rows found"):
            estimator.estimate_series(synthetic_enriched, "1999")
