# Survival Analysis for Phase Transition Timing

**Status**: To implement  
**Project**: `seasonal_analysis`  
**Extends**: `src/analysis/phase_probability.py` — replaces the Gaussian CDF duration model  
**Priority**: High — highest ROI for small N, methodologically correct for time-to-event data

---

## Why the current model falls short

The `PhaseProbabilityEstimator` fits a Gaussian to the 3–4 observed phase durations and uses its CDF to compute P(still in EXPANSION at day N). This has two structural problems:

1. **Ignores censoring.** The 2024 EXPANSION lasted 536 days and is confirmed complete. A Gaussian fit treats this as a noisy observation of a duration drawn from N(μ, σ). Survival analysis treats it as an *exact* observation: EXPANSION ended at exactly day 536. This is categorically more informative.

2. **Assumes constant hazard within the Gaussian shape.** The Gaussian implies a bell-shaped hazard — transitions are most likely near the mean and less likely far from it. The Weibull distribution allows the hazard to *increase with time* (shape parameter k > 1), which is economically intuitive: the longer you stay in EXPANSION without a peak, the more overextended the market becomes and the more likely a peak is.

---

## Core concepts

**Survival function** `S(t)`: probability of still being in the current phase at day t.  
`S(t) = P(phase duration > t)`

**Hazard function** `h(t)`: instantaneous rate of transitioning out of the current phase at day t, given survival to day t.  
`h(t) = f(t) / S(t)` where f(t) is the phase duration density.

**Conditional transition probability**: given we're at day N without a transition yet, probability of transitioning in the next k days.  
`P(transition in [N, N+k] | survived to N) = 1 - S(N+k) / S(N)`

This is the most actionable output: not "P(EXPANSION) = 74%" but "given we're at day 530, the probability EXPANSION ends in the next 30 days is 41%."

---

## Weibull distribution

The Weibull is parameterised by shape `k` and scale `λ`:

```
S(t) = exp(-(t/λ)^k)
h(t) = (k/λ) × (t/λ)^(k-1)
```

- `k = 1`: constant hazard (exponential distribution — memoryless)
- `k > 1`: hazard increases with time (the longer we wait, the more likely a transition)
- `k < 1`: hazard decreases with time

For BTC EXPANSION: expect `k > 1` (bull markets become increasingly unstable the longer they run). For DISTRIBUTION: also expect `k > 1` (capitulation becomes more likely the longer the bear market extends). Fitting will confirm or contradict this.

---

## Kaplan-Meier (non-parametric alternative)

With only 3–4 complete cycles, parametric assumptions are fragile. KM is the honest fallback:

- No distributional assumption
- Exact observations are used as-is (no Gaussian smoothing)
- Produces a step-function survival curve
- Confidence intervals via Greenwood's formula

With N=4, KM gives very wide confidence intervals — but it's not lying about that uncertainty. The Gaussian and Weibull look more precise than they are. Use KM as the primary estimator and Weibull as the secondary model to smooth interpolation between steps.

---

## Implementation plan

### New dependency
```
pip install lifelines
```
`lifelines` provides KM estimator, Weibull fitter, and confidence intervals in scikit-learn style.

### New class: `SurvivalPhaseDurationModel`

**Location**: `src/analysis/phase_probability.py` (add alongside existing class, or split into `src/analysis/survival_duration.py`)

```python
from lifelines import KaplanMeierFitter, WeibullFitter

class SurvivalPhaseDurationModel:
    """
    Fits Kaplan-Meier and Weibull survival models to phase durations
    from completed cycles.  Replaces the Gaussian CDF in PhaseProbabilityEstimator.

    Handles exact observations only (no censoring needed for completed cycles;
    the ongoing cycle is excluded from duration fitting since its phase is
    not yet complete).
    """

    def fit(self, durations: list[float], weights: list[float], phase: str) -> None:
        """
        durations : list of observed phase durations from completed cycles
        weights   : data_quality_weight per cycle (from config)
        phase     : "EXPANSION" | "DISTRIBUTION" | "ACCUMULATION"
        """
        # KM fit (non-parametric, exact observations, no censoring)
        self.km_ = KaplanMeierFitter()
        self.km_.fit(durations=durations, weights=weights, label=phase)

        # Weibull fit (parametric, for smooth hazard curves)
        self.wb_ = WeibullFitter()
        self.wb_.fit(durations=durations, weights=weights)

        self.phase_ = phase

    def survival_prob(self, t: float) -> float:
        """P(phase duration > t) — probability of still being in this phase at day t."""
        return float(self.km_.survival_function_at_times([t]).iloc[0])

    def conditional_transition_prob(self, current_day: float, horizon_days: int = 30) -> float:
        """
        Given we've been in this phase for current_day days, probability of
        transitioning within the next horizon_days days.
        """
        s_now  = self.survival_prob(current_day)
        s_then = self.survival_prob(current_day + horizon_days)
        if s_now == 0:
            return 1.0
        return float(1.0 - s_then / s_now)

    def hazard_at(self, t: float) -> float:
        """Instantaneous transition rate at day t (from Weibull fit)."""
        return float(self.wb_.hazard_at_times([t]).iloc[0])
```

### Changes to `PhaseProbabilityEstimator.fit()`

Replace the `_fit_duration_stats` → `_compute_boundary_stats` chain with:

```python
def _fit_survival_models(self, boundaries, complete_cycles):
    """Fit one SurvivalPhaseDurationModel per phase."""
    obs = {phase: {"durations": [], "weights": []} for phase in _PHASES}

    for cid in complete_cycles:
        b = boundaries[cid]
        w = self._weights.get(cid, 1.0)
        obs["EXPANSION"]["durations"].append(b["peak_day"])
        obs["EXPANSION"]["weights"].append(w)
        obs["DISTRIBUTION"]["durations"].append(b["trough_day"] - b["peak_day"])
        obs["DISTRIBUTION"]["weights"].append(w)
        obs["ACCUMULATION"]["durations"].append(b["max_day"] - b["trough_day"])
        obs["ACCUMULATION"]["weights"].append(w)

    self._survival_models = {}
    for phase in _PHASES:
        m = SurvivalPhaseDurationModel()
        m.fit(obs[phase]["durations"], obs[phase]["weights"], phase)
        self._survival_models[phase] = m
```

### Changes to `_time_prior()` and `estimate_series()`

Replace the Gaussian CDF call:
```python
# OLD
p_exp = 1.0 - norm.cdf(day, mu_peak, sigma_peak)

# NEW
p_exp = self._survival_models["EXPANSION"].survival_prob(day)
```

For the absolute trough boundary (DISTRIBUTION end), use a convolution approach:
- The trough day = expansion_duration + distribution_duration
- With survival functions: P(still in DISTRIBUTION at absolute day N) requires knowing the expansion end day
- Since expansion end is confirmed for the ongoing cycle (536 days for 2024), this simplifies:
  `P(still in DISTRIBUTION at day N) = DISTRIBUTION_survival.survival_prob(N - peak_day)`

This is cleaner than the Gaussian error propagation currently used.

### New output columns in `estimate_series()`

```python
# For each day in the ongoing cycle:
result["transition_prob_30d"]  # P(current phase ends in next 30 days)
result["transition_prob_60d"]  # P(current phase ends in next 60 days)
result["hazard_today"]         # instantaneous hazard rate (Weibull)
result["km_survival_lower"]    # KM 95% CI lower bound on S(t)
result["km_survival_upper"]    # KM 95% CI upper bound on S(t)
```

The `km_survival_lower/upper` columns are particularly valuable — they show the honest uncertainty range around the survival estimate given N=3–4.

---

## Config changes

No new config keys needed. The `data_quality_weight` per halving already controls the weighted KM fit. Add to `estimation` block:

```yaml
estimation:
  survival_horizon_days: [30, 60, 90]   # horizons for conditional transition probability output
```

---

## Tests to write

| Test | What it verifies |
|---|---|
| `test_km_survival_at_zero_is_one` | S(0) = 1 for all phases |
| `test_survival_monotone_decreasing` | S(t) ≥ S(t+1) for all t |
| `test_conditional_prob_between_zero_and_one` | output in [0, 1] |
| `test_conditional_prob_increases_with_horizon` | 60d prob ≥ 30d prob |
| `test_conditional_prob_at_very_late_day_is_one` | near 100% after observed maximum |
| `test_weibull_hazard_positive` | h(t) > 0 for all t > 0 |
| `test_weighted_obs_changes_fit` | lower-weight obs has less pull on KM curve |
| `test_prior_with_survival_sums_to_one` | three-phase prior sums to 1 |

---

## Expected output (illustrative)

For the 2024 cycle at day 750 (DISTRIBUTION ongoing):

```
phase_prior
  EXPANSION:    0.0%   (confirmed ended at day 536)
  DISTRIBUTION: 67.0%
  ACCUMULATION: 33.0%

conditional_transition_prob (current phase → next)
  next 30 days:  18.4%  [KM CI: 8.1% – 35.2%]
  next 60 days:  32.7%  [KM CI: 17.3% – 52.1%]
  next 90 days:  44.1%  [KM CI: 26.8% – 63.4%]

hazard_today: 0.0024  (0.24% daily probability of phase transition)
```

The confidence intervals are wide — that's correct and honest with N=3.

---

## Limitations

- KM with N=4 produces very wide confidence intervals. This is not a bug — it reflects genuine uncertainty. Do not suppress or narrow them artificially.
- Weibull assumes a specific parametric form. Validate the fit against the KM curve using log-rank test or AIC comparison with exponential distribution.
- The convolution for the DISTRIBUTION boundary (onset depends on EXPANSION end) is simplified when the expansion end is confirmed. For a genuinely ongoing EXPANSION, the two uncertainties stack and the KM approach needs adjustment.
