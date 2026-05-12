# Additional Price Features + Linear Discriminant Analysis

**Status**: To implement  
**Project**: `seasonal_analysis`  
**Extends**: `src/analysis/phase_probability.py` — replaces the Naïve Bayes price likelihood  
**Priority**: Medium-High — low complexity, meaningful improvement over Naïve Bayes, no new dependencies

---

## Why Naïve Bayes is the wrong tool here

The current `_price_posterior()` function applies Naïve Bayes: it multiplies the time prior by independent Gaussian likelihoods for each feature (currently only one feature: drawdown from ATH). Naïve Bayes assumes features are **conditionally independent given the phase**. They are not.

Example of the problem: low drawdown (near ATH) and low volatility are both individually consistent with EXPANSION. But they are also *correlated* — low drawdown implies we're near the ATH which implies recent days were trending up which implies low volatility. Naïve Bayes treats these as two independent signals boosting EXPANSION probability. LDA models their joint distribution and gives the correct, non-inflated update.

More concretely: with 5 features and Naïve Bayes, a day where all 5 features point to EXPANSION gives a likelihood product so large it swamps the prior almost regardless. With LDA and a realistic covariance structure, the 5 features collectively provide one appropriately-sized update.

---

## Features to add

All of these are computable from the existing `enriched_final` DataFrame, which already has `price`, `daily_return_pct`, `days_since_halving`, and `cycle_phase`:

### Feature 1: Drawdown from cycle ATH (already exists)
```python
drawdown_from_ath = (running_ath - price) / running_ath
```
Existing signal. Keep it.

### Feature 2: 30-day rolling volatility
```python
vol_30d = daily_return_pct.rolling(30, min_periods=10).std()
```
Phase signal: EXPANSION moderate (~3–4%), DISTRIBUTION high (~5–7%), ACCUMULATION low (~2–3%).

### Feature 3: 7-day momentum (short-term trend)
```python
momentum_7d = daily_return_pct.rolling(7).mean()
```
Phase signal: EXPANSION positive, DISTRIBUTION negative, ACCUMULATION near-zero with occasional spikes.

### Feature 4: Recovery from provisional trough
```python
running_trough   = price.cummin()  # after peak day
recovery_pct     = (price - running_trough) / running_trough
```
Only meaningful once a provisional peak is confirmed. Signal: positive and growing in ACCUMULATION, near-zero or declining in DISTRIBUTION.

### Feature 5: Drawdown velocity (rate of decline from ATH)
```python
drawdown_7d_change = drawdown_from_ath.diff(7)
```
Positive means drawdown is accelerating (price falling faster). Signal: spikes in early DISTRIBUTION, stabilises in ACCUMULATION.

### Feature 6: Normalised days since last ATH
```python
days_since_ath = (price.expanding().idxmax() - price.index).dt.days.abs()
days_since_ath_norm = days_since_ath / days_since_ath.max()
```
Proxy for "how stale is the rally." Signal: small in EXPANSION (ATH is recent), growing through DISTRIBUTION and ACCUMULATION.

---

## Linear Discriminant Analysis

LDA finds the linear combinations of features that best separate the three phase classes. Given feature vector `x = [f1, f2, ..., f6]`:

```
P(phase | x) ∝ P(x | phase) × P(phase)
```

Where `P(x | phase)` is modelled as a multivariate Gaussian with the **same covariance matrix** across all classes (homoscedastic assumption), estimated from training data.

**Why not QDA (Quadratic)?** QDA allows different covariance matrices per class. With ~3,500 training samples and 6 features, QDA is estimable but may overfit. LDA with a shared covariance is more regularised and appropriate for our sample size.

**Why not logistic regression?** Logistic regression is discriminative — it models P(phase | features) directly without assumptions about the feature distribution. This is slightly more principled but gives nearly identical results to LDA when features are approximately Gaussian. LDA is simpler to implement and explain.

---

## Implementation plan

### No new dependencies
`sklearn.discriminant_analysis.LinearDiscriminantAnalysis` is already available in scikit-learn (which `pandas` environments typically have).

### New class: `LDAPhaseClassifier`

**Location**: `src/analysis/phase_probability.py` (alongside `PhaseProbabilityEstimator`) or in a new `src/analysis/lda_classifier.py`

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class LDAPhaseClassifier:
    """
    Trains a Linear Discriminant Analysis model on phase-labelled daily
    observations from completed cycles.

    Replaces the Naïve Bayes price likelihood in PhaseProbabilityEstimator
    with a properly-correlated multivariate Gaussian classifier.
    """

    FEATURES = [
        "drawdown_from_ath",
        "vol_30d",
        "momentum_7d",
        "recovery_from_trough",
        "drawdown_velocity_7d",
        "days_since_ath_norm",
    ]

    def __init__(self):
        self._lda     = LinearDiscriminantAnalysis(solver="svd")
        self._scaler  = StandardScaler()
        self._fitted  = False
        self._classes = ["EXPANSION", "DISTRIBUTION", "ACCUMULATION"]

    def fit(
        self,
        enriched:        pd.DataFrame,
        boundaries:      dict,
        complete_cycles: list[str],
        weights:         dict[str, float],
    ) -> "LDAPhaseClassifier":
        """
        Parameters
        ----------
        enriched        : enriched DataFrame with cycle_id, cycle_phase, price columns
        boundaries      : boundaries dict (for peak_price, trough_day)
        complete_cycles : list of cycle labels used for training
        weights         : data_quality_weight per cycle label
        """
        train = enriched[
            enriched["cycle_id"].isin(complete_cycles) &
            enriched["cycle_phase"].notna()
        ].copy()

        train = self._build_features(train, boundaries)
        train = train.dropna(subset=self.FEATURES)

        X      = self._scaler.fit_transform(train[self.FEATURES].values)
        y      = train["cycle_phase"].values
        w      = train["cycle_id"].map(weights).values

        self._lda.fit(X, y, sample_weight=w)
        self._fitted = True
        return self

    def predict_proba(
        self,
        enriched:      pd.DataFrame,
        ongoing_label: str,
        boundaries:    dict,
    ) -> pd.DataFrame:
        """
        Returns per-day P(EXPANSION), P(DISTRIBUTION), P(ACCUMULATION)
        for the ongoing cycle.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        subset = enriched[enriched["cycle_id"] == ongoing_label].copy()
        subset = self._build_features(subset, boundaries)

        # Keep only rows where all features are available
        valid_mask = subset[self.FEATURES].notna().all(axis=1)
        subset_valid = subset[valid_mask]

        X      = self._scaler.transform(subset_valid[self.FEATURES].values)
        probs  = self._lda.predict_proba(X)    # shape (N, 3), columns = self._lda.classes_

        # Align columns to canonical order
        class_order = list(self._lda.classes_)
        col_map     = {cls: i for i, cls in enumerate(class_order)}

        result = pd.DataFrame(index=subset_valid.index)
        result["p_expansion_lda"]    = probs[:, col_map["EXPANSION"]]
        result["p_distribution_lda"] = probs[:, col_map["DISTRIBUTION"]]
        result["p_accumulation_lda"] = probs[:, col_map["ACCUMULATION"]]
        result["dominant_phase_lda"] = self._lda.predict(X)

        # Reindex to full ongoing cycle (NaN for early rows missing rolling features)
        full_result = pd.DataFrame(index=subset.index)
        return full_result.join(result)

    @staticmethod
    def _build_features(df: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
        df = df.copy()

        # Drawdown from cycle ATH (running max within cycle, per cycle)
        df["running_ath"] = df.groupby("cycle_id")["price"].cummax()
        df["drawdown_from_ath"] = (
            (df["running_ath"] - df["price"]) / df["running_ath"]
        ).clip(lower=0.0)

        # 30-day rolling volatility
        df["vol_30d"] = df.groupby("cycle_id")["daily_return_pct"].transform(
            lambda s: s.rolling(30, min_periods=10).std()
        )

        # 7-day momentum
        df["momentum_7d"] = df.groupby("cycle_id")["daily_return_pct"].transform(
            lambda s: s.rolling(7, min_periods=5).mean()
        )

        # Recovery from provisional trough (relative to confirmed peak)
        def _recovery(group):
            cid = group["cycle_id"].iloc[0]
            if cid not in boundaries:
                return pd.Series(np.nan, index=group.index)
            peak_day = boundaries[cid]["peak_day"]
            post_peak = group["days_since_halving"] > peak_day
            trough_so_far = group.loc[post_peak, "price"].cummin()
            recovery = (group.loc[post_peak, "price"] - trough_so_far) / trough_so_far
            return recovery.reindex(group.index)

        df["recovery_from_trough"] = df.groupby("cycle_id", group_keys=False).apply(_recovery)

        # Drawdown velocity (7-day change in drawdown)
        df["drawdown_velocity_7d"] = df.groupby("cycle_id")["drawdown_from_ath"].transform(
            lambda s: s.diff(7)
        )

        # Days since last ATH (normalised)
        def _days_since_ath(group):
            ath_idx  = group["price"].expanding().apply(lambda x: x.argmax(), raw=True).astype(int)
            day_nums = np.arange(len(group))
            days_ago = day_nums - ath_idx.values
            max_val  = days_ago.max() if days_ago.max() > 0 else 1
            return pd.Series(days_ago / max_val, index=group.index)

        df["days_since_ath_norm"] = df.groupby("cycle_id", group_keys=False).apply(_days_since_ath)

        return df
```

### Integration with `PhaseProbabilityEstimator`

Replace `_price_posterior()` (which uses Naïve Bayes on drawdown alone) with `LDAPhaseClassifier.predict_proba()`:

```python
# In PhaseProbabilityEstimator.fit():
self._lda_classifier = LDAPhaseClassifier()
self._lda_classifier.fit(enriched, boundaries, complete_cycles, self._weights)

# In estimate_series():
lda_probs = self._lda_classifier.predict_proba(enriched, ongoing_label, boundaries)

# Combine: time prior × LDA posterior via Bayes (replace current Naïve Bayes block)
# LDA already returns proper P(phase | features), so simply use as likelihood:
u_exp   = p_exp_prior   * lda_probs["p_expansion_lda"]
u_dist  = p_dist_prior  * lda_probs["p_distribution_lda"]
u_accum = p_accum_prior * lda_probs["p_accumulation_lda"]
# normalise to sum to 1
```

### New output columns

```
p_expansion_lda        float   LDA posterior for EXPANSION
p_distribution_lda     float   LDA posterior for DISTRIBUTION
p_accumulation_lda     float   LDA posterior for ACCUMULATION
dominant_phase_lda     str     LDA most likely phase
```

Plus the ensemble columns if combined with HMM (see `02_hmm_daily_returns.md`).

---

## Config changes

```yaml
estimation:
  lda_features:
    - "drawdown_from_ath"
    - "vol_30d"
    - "momentum_7d"
    - "recovery_from_trough"
    - "drawdown_velocity_7d"
    - "days_since_ath_norm"
  lda_min_history_days: 30    # min days in cycle before LDA output is trusted
```

The `lda_min_history_days` is important: rolling features like `vol_30d` are NaN for the first 30 days. The LDA should only be used once enough history exists; before that, fall back to the time prior alone.

---

## Tests to write

| Test | What it verifies |
|---|---|
| `test_lda_proba_sums_to_one` | three class probs sum to 1 for every row |
| `test_lda_proba_non_negative` | all values ≥ 0 |
| `test_lda_raises_if_not_fitted` | RuntimeError before fit() |
| `test_build_features_creates_all_columns` | all 6 feature columns present after `_build_features` |
| `test_vol_30d_non_negative` | volatility is always ≥ 0 |
| `test_drawdown_from_ath_between_zero_and_one` | clipped correctly |
| `test_recovery_from_trough_starts_after_peak` | NaN before peak_day |
| `test_lda_output_index_matches_input` | result index aligns with ongoing cycle |
| `test_lda_weighted_fit_differs_from_unweighted` | weights actually change the fit |
| `test_nan_rows_excluded_from_training` | fit doesn't crash with early NaN rows |

---

## Limitations and caveats

1. **LDA assumes homoscedastic Gaussians.** The volatility of returns in EXPANSION and DISTRIBUTION may be quite different. QDA is the principled fix but requires more data per class. A practical middle ground: use `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` which applies Ledoit-Wolf shrinkage to the covariance estimate.

2. **Early cycle rows have NaN features.** The rolling features (vol_30d, momentum_7d) are unavailable for the first 30 days. Fall back to time prior only for those rows and document this behaviour in the output (a `lda_available` boolean column is helpful).

3. **Feature leakage from `recovery_from_trough`.** This feature requires knowing the peak_day (from `boundaries`), which was itself estimated from price data. There is mild circularity — the phase labels used for LDA training were derived from a process that already used price signals. This is acceptable but should be documented.

4. **Class imbalance.** If EXPANSION lasts ~537 days and ACCUMULATION lasts ~519 days but we only have ~365 days of DISTRIBUTION data, the LDA training set is mildly imbalanced. Use `class_weight='balanced'` equivalent — pass `sample_weight` that inversely scales with class frequency.

5. **Feature scaling is required.** LDA is sensitive to scale; the `StandardScaler` in the implementation handles this. Always fit the scaler on training data and transform both training and test data — never fit on the test (ongoing cycle) data.
