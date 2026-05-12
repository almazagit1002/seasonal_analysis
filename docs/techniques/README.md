# Phase Probability Estimator — Improvement Roadmap

Techniques to improve `PhaseProbabilityEstimator` beyond the current Gaussian CDF + Naïve Bayes baseline.

## In this project (`seasonal_analysis`)

| # | File | Technique | Status | Priority | Data needed |
|---|---|---|---|---|---|
| 01 | `01_survival_analysis.md` | Survival analysis (Weibull / Kaplan-Meier) — replaces Gaussian duration model | To implement | High | Existing phase boundaries |
| 02 | `02_hmm_daily_returns.md` | Hidden Markov Model on daily returns — parallel estimation path from all daily observations | To implement | High | Existing enriched table |
| 03 | `03_price_features_lda.md` | Additional price features + LDA — replaces Naïve Bayes price likelihood | To implement | Medium-High | Existing enriched table |

## Separate project

| # | File | Project | Status | Data needed |
|---|---|---|---|---|
| 04 | `04_onchain_metrics_project.md` | `btc_cycle_intelligence` — on-chain metrics (MVRV, SOPR, exchange flows) combined with `seasonal_analysis` outputs | Nice to have | Glassnode API |

## Implementation order

```
Current:  Gaussian CDF + Naïve Bayes (single drawdown feature)
    ↓
Step 1:   Add survival analysis (01) — better duration model, honest uncertainty
    ↓
Step 2:   Add LDA + price features (03) — better price likelihood, replaces Naïve Bayes
    ↓
Step 3:   Add HMM (02) — independent estimation path, ensemble with above
    ↓
Future:   On-chain metrics project (04) — separate repo, consumes this project's outputs
```

Each step is independently deployable — the pipeline and output schema extend without breaking existing consumers.
