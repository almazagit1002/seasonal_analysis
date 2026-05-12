"""
Phase probability estimator for the ongoing BTC halving cycle.

Produces per-day probabilities of being in EXPANSION / DISTRIBUTION /
ACCUMULATION using two complementary signals:

  1. Time prior  — Gaussian CDFs fitted to weighted phase-duration statistics
                   across completed historical cycles.  Each cycle's weight is
                   controlled by data_quality_weight in config, allowing early
                   cycles with lower data confidence to contribute without
                   dominating the estimate.

  2. Price likelihood — distribution of drawdown-from-ATH observed historically
                        within each phase, combined with the time prior via
                        Naïve Bayes to produce a posterior estimate.

Both signals are reported separately so the caller can see when they agree
and when they diverge (divergence is itself analytically useful).

Uncertainty note: with N=3–4 completed cycles the fitted σ values are
indicative, not statistically tight.  Treat probabilities as informed
ranges rather than precise point estimates.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.logger import get_logger
from utils.config_reader import ConfigReader

logger = get_logger(__name__, log_to_file=True)

_PHASES     = ["EXPANSION", "DISTRIBUTION", "ACCUMULATION"]
_MIN_SIGMA  = 5.0    # floor on phase-duration σ to prevent degenerate CDFs
_MIN_DD_SIG = 0.05   # floor on drawdown σ


class PhaseProbabilityEstimator:
    """
    Fits phase-duration and drawdown distributions from completed cycles and
    produces per-day phase probabilities for the ongoing cycle.

    Typical call sequence:
        est = PhaseProbabilityEstimator()
        est.fit(phase_table, boundaries, enriched)
        prob_table = est.estimate_series(enriched, ongoing_label)
    """

    def __init__(self, config_file: str = "config") -> None:
        try:
            logger.info("Initializing PhaseProbabilityEstimator...")
            cfg       = ConfigReader(config_file)
            cycle_cfg = cfg.get("bitcoin_cycle")
            est_cfg   = cfg.get("estimation") or {}

            if not cycle_cfg:
                raise ValueError("Missing 'bitcoin_cycle' section in config")

            self._halvings_cfg        = cycle_cfg.get("halvings", [])
            self._min_cycles_required = int(est_cfg.get("min_cycles_required", 2))
            self._prior_only_dd       = float(est_cfg.get("prior_only_threshold", 0.05))

            self._weights: dict[str, float] = {
                h["label"]: float(h.get("data_quality_weight", 1.0))
                for h in self._halvings_cfg
            }

            # populated by fit()
            self._duration_stats: dict  = {}
            self._boundary_stats: dict  = {}
            self._drawdown_stats: dict  = {}
            self._n_complete: int       = 0
            self._fitted: bool          = False

            logger.info(
                f"PhaseProbabilityEstimator ready — "
                f"min_cycles={self._min_cycles_required}  "
                f"weights: { {k: v for k, v in self._weights.items()} }"
            )
        except Exception:
            logger.exception("Failed to initialize PhaseProbabilityEstimator")
            raise

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        phase_table: pd.DataFrame,
        boundaries:  dict,
        enriched:    pd.DataFrame,
        price_col:   str = "price",
    ) -> "PhaseProbabilityEstimator":
        """
        Fit duration and drawdown distributions from completed cycles.

        Parameters
        ----------
        phase_table : from CycleAnalyzer.build_phase_table()
        boundaries  : from CycleAnalyzer.compute_boundaries()
        enriched    : enriched DataFrame with cycle_id, cycle_phase, price columns
        price_col   : name of the closing-price column (default 'price')
        """
        try:
            logger.info("Fitting phase probability distributions...")

            complete_cycles = (
                phase_table[phase_table["status"] == "complete"]["cycle_id"]
                .astype(str).unique().tolist()
            )
            self._n_complete = len(complete_cycles)

            if self._n_complete < self._min_cycles_required:
                raise ValueError(
                    f"Only {self._n_complete} complete cycle(s) available — "
                    f"need at least {self._min_cycles_required} to fit"
                )

            logger.info(f"  complete cycles used for fitting: {complete_cycles}")

            self._duration_stats = self._fit_duration_stats(boundaries, complete_cycles)
            self._boundary_stats = self._compute_boundary_stats()
            self._drawdown_stats = self._fit_drawdown_stats(
                enriched, boundaries, complete_cycles, price_col
            )

            self._fitted = True
            logger.info("Fitting completed successfully")
            return self

        except Exception:
            logger.exception("Failed to fit PhaseProbabilityEstimator")
            raise

    def _fit_duration_stats(self, boundaries: dict, complete_cycles: list) -> dict:
        """
        Weighted mean and σ for each phase duration across completed cycles.

        EXPANSION duration  = peak_day
        DISTRIBUTION duration = trough_day - peak_day
        ACCUMULATION duration = max_day   - trough_day

        Each cycle is weighted by data_quality_weight from config.
        """
        obs: dict = {p: {"values": [], "weights": []} for p in _PHASES}

        for cid in complete_cycles:
            if cid not in boundaries:
                logger.warning(f"  [{cid}] missing from boundaries — skipped")
                continue
            b = boundaries[cid]
            w = self._weights.get(cid, 1.0)

            obs["EXPANSION"]["values"].append(float(b["peak_day"]))
            obs["EXPANSION"]["weights"].append(w)

            obs["DISTRIBUTION"]["values"].append(float(b["trough_day"] - b["peak_day"]))
            obs["DISTRIBUTION"]["weights"].append(w)

            obs["ACCUMULATION"]["values"].append(float(b["max_day"] - b["trough_day"]))
            obs["ACCUMULATION"]["weights"].append(w)

        stats = {}
        for phase in _PHASES:
            vals = np.array(obs[phase]["values"], dtype=float)
            wts  = np.array(obs[phase]["weights"], dtype=float)
            mu, sigma = _weighted_mean_sigma(vals, wts)
            sigma = max(sigma, _MIN_SIGMA)
            stats[phase] = {"mu": mu, "sigma": sigma, "n": len(vals)}
            logger.info(
                f"  {phase:14s}: n={len(vals)}  "
                f"μ={mu:6.1f}d  σ={sigma:5.1f}d  "
                f"range=[{vals.min():.0f}, {vals.max():.0f}]  "
                f"weights={wts.tolist()}"
            )

        return stats

    def _compute_boundary_stats(self) -> dict:
        """
        Derive absolute boundary-day distributions from phase duration stats.

        EXPANSION  ends at  peak_day  ~ N(μ_peak, σ_peak)
        DISTRIBUTION ends at trough_day ~ N(μ_trough, σ_trough)
          where μ_trough = μ_peak + μ_dist
          and   σ_trough = sqrt(σ_peak² + σ_dist²)   (independent error propagation)
        """
        exp  = self._duration_stats["EXPANSION"]
        dist = self._duration_stats["DISTRIBUTION"]

        mu_peak      = exp["mu"]
        sigma_peak   = exp["sigma"]
        mu_trough    = mu_peak + dist["mu"]
        sigma_trough = float(np.sqrt(sigma_peak ** 2 + dist["sigma"] ** 2))

        logger.info(f"  EXPANSION ends:    μ={mu_peak:.1f}d  σ={sigma_peak:.1f}d")
        logger.info(f"  DISTRIBUTION ends: μ={mu_trough:.1f}d  σ={sigma_trough:.1f}d")

        return {
            "mu_peak":      mu_peak,
            "sigma_peak":   sigma_peak,
            "mu_trough":    mu_trough,
            "sigma_trough": sigma_trough,
        }

    def _fit_drawdown_stats(
        self,
        enriched:        pd.DataFrame,
        boundaries:      dict,
        complete_cycles: list,
        price_col:       str,
    ) -> dict:
        """
        For each phase, compute the distribution of drawdown-from-ATH across
        all days of all completed cycles.

        drawdown = (cycle_peak_price - current_price) / cycle_peak_price
        Clipped to [0, 1] — rows above the cycle ATH (during EXPANSION) get 0.
        """
        records = []

        for cid in complete_cycles:
            if cid not in boundaries:
                continue
            b      = boundaries[cid]
            w      = self._weights.get(cid, 1.0)
            subset = enriched[
                (enriched["cycle_id"] == cid) &
                enriched["cycle_phase"].notna() &
                enriched[price_col].notna()
            ].copy()

            if subset.empty:
                logger.warning(f"  [{cid}] no rows for drawdown fitting")
                continue

            peak_price         = float(b["peak_price"])
            subset["drawdown"] = ((peak_price - subset[price_col]) / peak_price).clip(0.0, 1.0)
            subset["w"]        = w
            records.append(subset[["cycle_phase", "drawdown", "w"]])

        if not records:
            logger.warning("No drawdown data — price likelihood will be disabled")
            return {}

        all_data = pd.concat(records, ignore_index=True)
        stats    = {}

        for phase in _PHASES:
            mask = all_data["cycle_phase"] == phase
            n    = mask.sum()
            if n < 10:
                logger.warning(f"  [{phase}] only {n} drawdown obs — skipped")
                continue
            vals = all_data.loc[mask, "drawdown"].values
            wts  = all_data.loc[mask, "w"].values
            mu, sigma = _weighted_mean_sigma(vals, wts)
            sigma = max(sigma, _MIN_DD_SIG)
            stats[phase] = {"mu": mu, "sigma": sigma}
            logger.info(
                f"  drawdown {phase:14s}: "
                f"μ={mu:.2%}  σ={sigma:.2%}  n={n}"
            )

        return stats

    # ── single-point estimate ─────────────────────────────────────────────────

    def estimate(
        self,
        current_day:      float,
        current_drawdown: float | None = None,
    ) -> dict:
        """
        Phase probabilities for a single day of the ongoing cycle.

        Parameters
        ----------
        current_day      : days_since_halving
        current_drawdown : fraction below cycle ATH  (0 = at ATH, 0.7 = 70% below)
                           If None, only the time prior is returned.

        Returns
        -------
        dict — always contains prior probabilities; posterior columns are NaN
        when drawdown stats are unavailable or no drawdown is supplied.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate()")

        b = self._boundary_stats
        prior = _time_prior(
            current_day,
            b["mu_peak"], b["sigma_peak"],
            b["mu_trough"], b["sigma_trough"],
        )

        result: dict = {
            "cycle_day":              current_day,
            "p_expansion_prior":      round(prior[0], 4),
            "p_distribution_prior":   round(prior[1], 4),
            "p_accumulation_prior":   round(prior[2], 4),
            "dominant_phase_prior":   _PHASES[int(np.argmax(prior))],
            "drawdown_from_ath":      np.nan,
            "p_expansion_posterior":  np.nan,
            "p_distribution_posterior": np.nan,
            "p_accumulation_posterior": np.nan,
            "dominant_phase_posterior": None,
            "signals_agree":          None,
        }

        if current_drawdown is not None and len(self._drawdown_stats) == len(_PHASES):
            posterior = _price_posterior(
                prior, current_drawdown,
                self._drawdown_stats, self._prior_only_dd,
            )
            result.update({
                "drawdown_from_ath":         round(current_drawdown, 4),
                "p_expansion_posterior":     round(posterior[0], 4),
                "p_distribution_posterior":  round(posterior[1], 4),
                "p_accumulation_posterior":  round(posterior[2], 4),
                "dominant_phase_posterior":  _PHASES[int(np.argmax(posterior))],
                "signals_agree":             (
                    _PHASES[int(np.argmax(prior))] == _PHASES[int(np.argmax(posterior))]
                ),
            })

        return result

    # ── series estimate ───────────────────────────────────────────────────────

    def estimate_series(
        self,
        enriched:      pd.DataFrame,
        ongoing_label: str,
        price_col:     str = "price",
    ) -> pd.DataFrame:
        """
        Phase probability estimates for every day of the ongoing cycle.

        Parameters
        ----------
        enriched      : enriched DataFrame with cycle_id, days_since_halving, price
        ongoing_label : cycle label string (e.g. '2024')
        price_col     : name of the closing-price column

        Returns
        -------
        DataFrame indexed by date, one row per day, with columns:
          cycle_day, p_*_prior, dominant_phase_prior,
          drawdown_from_ath,
          p_*_posterior, dominant_phase_posterior, signals_agree
        """
        try:
            if not self._fitted:
                raise RuntimeError("Call fit() before estimate_series()")

            logger.info(
                f"Estimating phase probability series for cycle {ongoing_label}..."
            )

            subset = (
                enriched[enriched["cycle_id"] == ongoing_label]
                .sort_values("days_since_halving")
                .copy()
            )

            if subset.empty:
                raise ValueError(
                    f"No rows found for cycle '{ongoing_label}' in enriched data"
                )

            subset["running_ath"] = subset[price_col].cummax()
            subset["drawdown_from_ath"] = (
                (subset["running_ath"] - subset[price_col]) / subset["running_ath"]
            ).clip(lower=0.0)

            days      = subset["days_since_halving"].values.astype(float)
            drawdowns = subset["drawdown_from_ath"].values.astype(float)

            b = self._boundary_stats

            # ── vectorised time prior ─────────────────────────────────────────
            p_exp   = 1.0 - norm.cdf(days, b["mu_peak"],   b["sigma_peak"])
            p_accum = norm.cdf(days,        b["mu_trough"], b["sigma_trough"])
            p_dist  = np.maximum(0.0, 1.0 - p_exp - p_accum)
            totals  = p_exp + p_dist + p_accum
            p_exp  /= totals
            p_dist /= totals
            p_accum /= totals

            dominant_prior = np.where(
                p_exp >= np.maximum(p_dist, p_accum),
                "EXPANSION",
                np.where(p_dist >= p_accum, "DISTRIBUTION", "ACCUMULATION"),
            )

            result = pd.DataFrame(
                {
                    "cycle_day":              days,
                    "p_expansion_prior":      np.round(p_exp, 4),
                    "p_distribution_prior":   np.round(p_dist, 4),
                    "p_accumulation_prior":   np.round(p_accum, 4),
                    "dominant_phase_prior":   dominant_prior,
                    "drawdown_from_ath":      np.round(drawdowns, 4),
                },
                index=subset.index,
            )

            # ── vectorised posterior ──────────────────────────────────────────
            if len(self._drawdown_stats) == len(_PHASES):
                ds       = self._drawdown_stats
                l_exp    = norm.pdf(drawdowns, ds["EXPANSION"]["mu"],    ds["EXPANSION"]["sigma"])
                l_dist   = norm.pdf(drawdowns, ds["DISTRIBUTION"]["mu"], ds["DISTRIBUTION"]["sigma"])
                l_accum  = norm.pdf(drawdowns, ds["ACCUMULATION"]["mu"], ds["ACCUMULATION"]["sigma"])

                # Near-ATH rows: set all likelihoods equal → posterior = prior
                near_ath = drawdowns < self._prior_only_dd
                l_exp[near_ath]   = 1.0
                l_dist[near_ath]  = 1.0
                l_accum[near_ath] = 1.0

                u_exp   = p_exp   * l_exp
                u_dist  = p_dist  * l_dist
                u_accum = p_accum * l_accum
                u_total = u_exp + u_dist + u_accum

                safe    = np.where(u_total > 0, u_total, 1.0)
                post_exp   = np.where(u_total > 0, u_exp   / safe, p_exp)
                post_dist  = np.where(u_total > 0, u_dist  / safe, p_dist)
                post_accum = np.where(u_total > 0, u_accum / safe, p_accum)

                dominant_post = np.where(
                    post_exp >= np.maximum(post_dist, post_accum),
                    "EXPANSION",
                    np.where(post_dist >= post_accum, "DISTRIBUTION", "ACCUMULATION"),
                )

                result["p_expansion_posterior"]    = np.round(post_exp,   4)
                result["p_distribution_posterior"] = np.round(post_dist,  4)
                result["p_accumulation_posterior"] = np.round(post_accum, 4)
                result["dominant_phase_posterior"] = dominant_post
                result["signals_agree"]            = dominant_prior == dominant_post
            else:
                logger.warning(
                    "Drawdown stats incomplete — posterior columns set to NaN"
                )
                result["p_expansion_posterior"]    = np.nan
                result["p_distribution_posterior"] = np.nan
                result["p_accumulation_posterior"] = np.nan
                result["dominant_phase_posterior"] = None
                result["signals_agree"]            = None

            result.index.name = "date"
            logger.info(
                f"Probability series built: {len(result)} rows  "
                f"cycle {ongoing_label}"
            )
            return result

        except Exception:
            logger.exception("Failed to build probability series")
            raise


# ── module-level pure functions (testable without class state) ────────────────

def _weighted_mean_sigma(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Population-weighted mean and standard deviation."""
    total = float(weights.sum())
    mu    = float(np.dot(weights, values) / total)
    var   = float(np.dot(weights, (values - mu) ** 2) / total)
    return mu, float(np.sqrt(var))


def _time_prior(
    day:          float,
    mu_peak:      float,
    sigma_peak:   float,
    mu_trough:    float,
    sigma_trough: float,
) -> list[float]:
    """
    Return [p_expansion, p_distribution, p_accumulation] from duration CDFs.
    The three values sum to 1.
    """
    p_exp   = float(1.0 - norm.cdf(day, mu_peak,   sigma_peak))
    p_accum = float(norm.cdf(day,        mu_trough, sigma_trough))
    p_dist  = float(max(0.0, 1.0 - p_exp - p_accum))
    total   = p_exp + p_dist + p_accum
    return [p_exp / total, p_dist / total, p_accum / total]


def _price_posterior(
    prior:            list[float],
    current_drawdown: float,
    drawdown_stats:   dict,
    prior_only_dd:    float,
) -> list[float]:
    """
    Update time prior with drawdown likelihood via Naïve Bayes.
    Near-ATH rows (drawdown < prior_only_dd) skip the update.
    """
    if current_drawdown < prior_only_dd:
        return prior

    likelihoods = [
        float(norm.pdf(current_drawdown,
                       drawdown_stats[p]["mu"],
                       drawdown_stats[p]["sigma"]))
        if p in drawdown_stats else 1.0
        for p in _PHASES
    ]

    unnorm = [p * l for p, l in zip(prior, likelihoods)]
    total  = sum(unnorm)
    if total == 0:
        return prior
    return [u / total for u in unnorm]
