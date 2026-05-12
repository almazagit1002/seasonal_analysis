# Hidden Markov Model on Daily Returns

**Status**: To implement  
**Project**: `seasonal_analysis`  
**Extends**: `src/analysis/phase_probability.py` — adds a parallel estimation path using all daily return data  
**Priority**: High — uses ~3,500 daily observations instead of 3 boundary events; highest data leverage

---

## Why HMM is the most data-efficient technique here

Every other technique in this roadmap uses the 3–4 observed phase boundary dates as its primary training signal. The HMM uses **every single daily return observation** across all completed cycles — roughly 3,500 data points. This is the fundamental reason it's the highest-ROI technique.

The current model asks: "given that EXPANSION historically ends around day 537 ± 11, what phase are we probably in?" The HMM asks: "given that today's return is -3.2% and the 30-day volatility is 4.1%, what hidden state (phase) are we most likely in?" These are complementary questions, and the HMM answer is informed by much richer evidence.

---

## How a Gaussian HMM works

A Gaussian HMM assumes:
1. There is an unobserved (hidden) state `s_t` ∈ {EXPANSION, DISTRIBUTION, ACCUMULATION}
2. The observed daily return `r_t` is drawn from `N(μ_s, σ_s)` — a Gaussian whose parameters depend on the current state
3. The state transitions according to a Markov chain with transition matrix `A` where `A[i][j] = P(next state is j | current state is i)`
4. The initial state probabilities are `π`

**Training (Baum-Welch / EM)**: Given a sequence of returns from completed cycles (with known state labels for supervision), the algorithm estimates `μ_s, σ_s` per state, and the transition matrix `A`.

**Decoding (Viterbi)**: Given a new return sequence (the ongoing cycle), find the most likely hidden state sequence. This gives a deterministic labelling.

**Filtering (Forward algorithm)**: At each day, compute the posterior probability distribution over states `P(s_t | r_1, ..., r_t)`. This is what feeds the probability estimator.

---

## Supervised vs unsupervised training

### Unsupervised (standard HMM)
Train on the full return sequence without providing state labels. The algorithm discovers clusters of similar return behaviour. With 3 states it will find something, but the discovered states may not align with EXPANSION/DISTRIBUTION/ACCUMULATION.

**Problem**: with 4 cycles, the algorithm might find a "crash day" state, a "grind up" state, and a "sideways" state — not the economically meaningful phases.

### Supervised initialisation + constrained transitions
Better approach for our use case:

1. **Initialise emission parameters** from known phase labels: compute `μ_s` and `σ_s` for each phase using the historical phase assignments from `enriched_final`. This gives the algorithm a strong starting point.

2. **Constrain the transition matrix** to forward-only transitions: EXPANSION can only go to DISTRIBUTION, DISTRIBUTION can only go to ACCUMULATION, ACCUMULATION can only go to EXPANSION (next cycle). No backwards transitions allowed.

3. **Run EM refinement** (a few iterations) to fine-tune the emission parameters while keeping the transition structure.

This gives a model that is economically interpretable and trained on actual phase labels.

---

## What the emission distributions look like (expected)

From historical data:

| Phase | Expected μ (daily return) | Expected σ (daily volatility) |
|---|---|---|
| EXPANSION | +0.4% to +0.7% | 3–5% |
| DISTRIBUTION | -0.3% to -0.4% | 4–6% |
| ACCUMULATION | +0.2% to +0.3% | 2–4% |

These are rough priors. The actual fitted values may differ and should be logged for inspection.

---

## Multi-variate extension

A univariate HMM on daily returns misses information. A multivariate Gaussian HMM on a feature vector is more discriminative:

```python
features = [
    "daily_return_pct",          # mean return signal
    "rolling_30d_volatility",    # volatility regime
    "rolling_7d_return",         # short-term momentum
]
```

The 3-dimensional Gaussian emission has a full covariance matrix per state. With ~3,500 training observations across 3 phases, this is well-conditioned.

---

## Implementation plan

### New dependency
```
pip install hmmlearn
```

### New class: `HMMPhaseDetector`

**Location**: `src/analysis/hmm_detector.py`

```python
from hmmlearn import hmm
import numpy as np

class HMMPhaseDetector:
    """
    Gaussian HMM fitted on daily returns (and optionally rolling volatility)
    from completed BTC halving cycles.

    Produces per-day posterior phase probabilities for the ongoing cycle
    using the forward algorithm — the sequential equivalent of the
    Naïve Bayes price likelihood in PhaseProbabilityEstimator.

    Training uses supervised emission initialisation from known phase labels,
    with a constrained forward-only transition matrix.
    """

    _PHASE_IDX = {"EXPANSION": 0, "DISTRIBUTION": 1, "ACCUMULATION": 2}
    _IDX_PHASE = {0: "EXPANSION", 1: "DISTRIBUTION", 2: "ACCUMULATION"}

    def __init__(self, n_iter: int = 50, features: list[str] = None):
        self.n_iter    = n_iter
        self.features  = features or ["daily_return_pct", "rolling_30d_vol"]
        self.model_    = None
        self._fitted   = False

    def fit(self, enriched: pd.DataFrame, complete_cycles: list[str]) -> "HMMPhaseDetector":
        """
        Parameters
        ----------
        enriched        : enriched DataFrame with cycle_id, cycle_phase, feature columns
        complete_cycles : list of cycle labels to use for training
        """
        # Build feature matrix from completed cycles only
        train_data = enriched[
            enriched["cycle_id"].isin(complete_cycles) &
            enriched["cycle_phase"].notna()
        ].sort_index().copy()

        train_data = self._add_features(train_data)
        train_data = train_data.dropna(subset=self.features)

        X      = train_data[self.features].values
        labels = train_data["cycle_phase"].map(self._PHASE_IDX).values

        # Initialise emission parameters from supervised labels
        n_features = len(self.features)
        means_init = np.array([
            X[labels == i].mean(axis=0) for i in range(3)
        ])
        covars_init = np.array([
            np.diag(X[labels == i].var(axis=0)) + np.eye(n_features) * 1e-4
            for i in range(3)
        ])

        # Constrained forward-only transition matrix
        # EXPANSION→DISTRIBUTION, DISTRIBUTION→ACCUMULATION, ACCUMULATION→EXPANSION
        transmat_init = np.array([
            [0.99, 0.01, 0.00],   # from EXPANSION
            [0.00, 0.99, 0.01],   # from DISTRIBUTION
            [0.01, 0.00, 0.99],   # from ACCUMULATION (→next cycle)
        ])

        self.model_ = hmm.GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=self.n_iter,
            init_params="",     # disable automatic initialisation
            params="mc",        # train only means and covariances, not transition matrix
        )
        self.model_.startprob_ = np.array([0.80, 0.15, 0.05])
        self.model_.transmat_  = transmat_init
        self.model_.means_     = means_init
        self.model_.covars_    = covars_init

        # Provide sequence lengths (one per cycle) so EM treats them as independent
        lengths = [
            (labels == 0).sum() + (labels == 1).sum() + (labels == 2).sum()
        ]
        # NOTE: proper multi-sequence fit requires splitting by cycle and passing lengths
        # See implementation note below.
        self.model_.fit(X, lengths=lengths)
        self._fitted = True
        return self

    def predict_proba(self, enriched: pd.DataFrame, ongoing_label: str) -> pd.DataFrame:
        """
        Run the forward algorithm on the ongoing cycle to get posterior
        phase probabilities for each day.

        Returns DataFrame with p_expansion_hmm, p_distribution_hmm,
        p_accumulation_hmm, dominant_phase_hmm columns.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        subset = enriched[enriched["cycle_id"] == ongoing_label].sort_index().copy()
        subset = self._add_features(subset)
        subset = subset.dropna(subset=self.features)

        X = subset[self.features].values

        # Forward algorithm: returns log-likelihood and posterior state probabilities
        _, posteriors = self.model_.score_samples(X)  # posteriors shape: (T, 3)

        result = pd.DataFrame({
            "p_expansion_hmm":    posteriors[:, self._PHASE_IDX["EXPANSION"]],
            "p_distribution_hmm": posteriors[:, self._PHASE_IDX["DISTRIBUTION"]],
            "p_accumulation_hmm": posteriors[:, self._PHASE_IDX["ACCUMULATION"]],
        }, index=subset.index)

        result["dominant_phase_hmm"] = result[
            ["p_expansion_hmm", "p_distribution_hmm", "p_accumulation_hmm"]
        ].idxmax(axis=1).str.replace("p_|_hmm", "", regex=True).str.upper()

        return result

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "rolling_30d_vol" not in df.columns:
            df["rolling_30d_vol"] = (
                df["daily_return_pct"].rolling(30, min_periods=10).std()
            )
        return df
```

### Multi-sequence fitting (implementation note)

`hmmlearn.GaussianHMM.fit()` accepts a concatenated array `X` and a `lengths` list telling the model where each independent sequence begins and ends. For 3 complete cycles:

```python
sequences = []
lengths   = []
for cid in complete_cycles:
    cycle_data = train_data[train_data["cycle_id"] == cid].sort_index()
    cycle_data = self._add_features(cycle_data).dropna(subset=self.features)
    sequences.append(cycle_data[self.features].values)
    lengths.append(len(cycle_data))

X = np.concatenate(sequences)
self.model_.fit(X, lengths=lengths)
```

This is critical. Without `lengths`, the model incorrectly treats the last day of the 2016 cycle and the first day of the 2020 cycle as sequential, which introduces false transition signals at cycle boundaries.

### Integration with `PhaseProbabilityEstimator`

The HMM produces a separate probability vector `[p_exp_hmm, p_dist_hmm, p_accum_hmm]` for each day. Combine with the existing time prior and price posterior using ensemble averaging:

```python
# In estimate_series():
hmm_probs = self._hmm_detector.predict_proba(enriched, ongoing_label)

# Weighted ensemble: time_prior × α + hmm_posterior × β + price_posterior × γ
# where α + β + γ = 1
# Initial weights: α=0.33, β=0.33, γ=0.34 — tune on held-out cycles
result["p_expansion_ensemble"]    = (
    α * result["p_expansion_prior"] +
    β * hmm_probs["p_expansion_hmm"] +
    γ * result["p_expansion_posterior"]
)
# ... same for distribution and accumulation
# normalise to sum to 1
```

The ensemble weights (`α`, `β`, `γ`) should live in config under `estimation.ensemble_weights`.

### New output columns

Added to the probability series table:

```
p_expansion_hmm           float   HMM posterior for EXPANSION
p_distribution_hmm        float   HMM posterior for DISTRIBUTION
p_accumulation_hmm        float   HMM posterior for ACCUMULATION
dominant_phase_hmm        str     most likely phase per HMM
p_expansion_ensemble      float   weighted ensemble of all signals
p_distribution_ensemble   float
p_accumulation_ensemble   float
dominant_phase_ensemble   str
hmm_signals_agree         bool    does HMM agree with time prior?
```

---

## Config changes

```yaml
estimation:
  hmm_features:
    - "daily_return_pct"
    - "rolling_30d_vol"
  hmm_n_iter: 50
  hmm_transition_self_prob: 0.99    # probability of staying in current phase each day
  ensemble_weights:
    time_prior:      0.33
    price_posterior: 0.33
    hmm_posterior:   0.34
```

---

## Pipeline placement

```
Step 4: stamp_phases()          → enriched_final (with cycle_id, cycle_phase)
Step 6: build_phase_table()     → phase_table
Step 7: PhaseProbabilityEstimator.fit()   ← pass enriched_final for drawdown stats
        HMMPhaseDetector.fit()            ← pass enriched_final for HMM training
        estimator.estimate_series()       ← time prior + price posterior
        hmm_detector.predict_proba()      ← HMM posterior
        merge → combined prob_series
Step 8: validate_phase_probability()
Step 9: upload btc_phase_probability.parquet
```

The `HMMPhaseDetector` trains independently from `PhaseProbabilityEstimator` and the two outputs are merged before upload.

---

## Tests to write

| Test | What it verifies |
|---|---|
| `test_hmm_posteriors_sum_to_one` | p_exp + p_dist + p_accum = 1 every row |
| `test_hmm_early_cycle_mostly_expansion` | first 30 days predominantly EXPANSION |
| `test_hmm_posteriors_non_negative` | all values ≥ 0 |
| `test_hmm_raises_if_not_fitted` | RuntimeError before fit() |
| `test_multi_sequence_lengths_used` | fit with lengths vs. without lengths differs |
| `test_add_features_creates_vol_column` | rolling vol column appears |
| `test_predict_proba_index_matches_input` | output index aligns with ongoing cycle rows |
| `test_ensemble_sums_to_one` | combined ensemble probabilities sum to 1 |
| `test_dominant_phase_hmm_matches_argmax` | dominant_phase_hmm is the column with highest probability |

---

## Limitations and caveats

1. **Stationarity assumption**: the HMM assumes the emission distributions are stable across cycles. If BTC return distributions shift over time (they do — 2012 had 10× the daily volatility of 2024), the 2012 observations will distort the 2016/2020/2024 estimates. Weight by `data_quality_weight` in the training data, or exclude 2012 from HMM training.

2. **Non-stationarity of the series**: HMMs assume the returns are stationary within each state. BTC EXPANSION has a trend — returns are positive throughout but the magnitude changes. Consider using return *residuals* (after removing a smoothed trend) instead of raw returns.

3. **Transition matrix constraints**: `hmmlearn` does not natively support constrained transition matrices. The `params="mc"` trick (training only means and covariances) freezes the transition matrix at its initialised value. This is the right approach given the economic constraint (no backwards transitions), but it means EM cannot refine the transition probabilities.

4. **N=3 training sequences**: the HMM is trained on 3 independent sequences. This is low but manageable because each sequence is ~1,400 days long (~4,200 total daily observations). The emission parameters will be reasonably estimated; the transition probabilities are more uncertain.
