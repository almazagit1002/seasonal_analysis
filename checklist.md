# Production Readiness Checklist

Full roadmap from current state to production-ready pipeline.
Items are ordered by dependency within each section. Blocking items are marked **[BLOCKING]**.

---

## 0 · Project health — fix first

These are existing defects that block everything else.

- [ ] **[BLOCKING]** Fix `requirements.txt` — `scipy` is used in `phase_probability.py` but not listed; pipeline cannot install clean
- [ ] **[BLOCKING]** Fix `setup.py` — `python_requires=">=3.9"` but code uses `tuple[int, float]` union syntax (3.10+) and `float | None` (3.10+); set to `>=3.10`
- [ ] **[BLOCKING]** Consolidate log directories — logs write to both `logs/` (when run from root) and `src/logs/` (when invoked from src/); pick one path in `logger.py` and enforce it
- [ ] Add `.gitignore` — `logs/`, `src/logs/`, `*.parquet`, `.env`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `venv/` are all untracked or should be excluded
- [ ] Remove tracked log files from git history (`git rm -r --cached logs/ src/logs/`)

---

## 1 · Seasonal analysis — apply a better approach

**Assessment**: the current seasonal profile is not production-credible as-is.

The `SeasonalAggregator` computes simple groupby averages with no uncertainty quantification. With N=3 complete cycles, a seasonal pattern appearing in all three cells is statistically indistinguishable from chance in most cases. Publishing `avg_return_pct = 0.73%` without a confidence interval or p-value is misleading — a downstream consumer cannot know whether that edge is real.

The answer is not to replace the current approach but to layer significance on top of it. The groupby profile stays; what's added is rigour.

### 1a · Statistical significance layer **[HIGH PRIORITY]**
- [ ] Add `p_value` column to the seasonal profile — two-sided t-test per cell (H₀: avg_return_pct = 0)
- [ ] Add `ci_lower_95` and `ci_upper_95` columns — 95% confidence interval on `avg_return_pct` using t-distribution (small-N appropriate)
- [ ] Add `is_significant` boolean column — `p_value < 0.05` AND `sample_size >= 3`
- [ ] Add `effect_size` (Cohen's d) column — practical significance beyond statistical
- [ ] Update `validate_profile` to check these new columns
- [ ] Update `test_schema.py` with the new columns
- [ ] Document in README: "rows with `is_significant = False` should not be used for trading decisions"

### 1b · Bootstrap confidence intervals
- [ ] Replace t-distribution CIs with bootstrap resampling (10,000 iterations per cell) — does not assume normality, more honest for fat-tailed returns
- [ ] Add `bootstrap_ci_lower` and `bootstrap_ci_upper` columns alongside the t-test CIs
- [ ] Store bootstrap CIs as the primary CIs; keep t-test CIs for comparison

### 1c · Recency-weighted profile
- [ ] Apply `data_quality_weight` (already in config per halving) to the `SeasonalAggregator` groupby — 2012 cycle contributes less to `avg_return_pct` since its market dynamics differ from 2016+
- [ ] Add `weighted_avg_return_pct` column alongside the current equal-weight `avg_return_pct`
- [ ] Make the weighting method config-driven (toggle between equal-weight and weighted)

### 1d · Walk-forward backtest **[CRITICAL FOR CREDIBILITY]**
- [ ] Create `src/analysis/backtest.py` — `SeasonalBacktester` class
- [ ] For each complete cycle N: train the seasonal profile on all cycles < N, generate predicted returns for cycle N, compute prediction error
- [ ] Output metrics: out-of-sample R², MAE, directional accuracy (predicted sign matches actual sign), information coefficient (Spearman correlation of predicted vs actual returns)
- [ ] If directional accuracy < 55% across the walk-forward, the seasonal profile has no predictive value — document this honestly regardless of outcome
- [ ] Upload backtest results as `btc_seasonal_backtest.parquet` to S3

### 1e · Spectral confirmation (optional but valuable)
- [ ] Add `src/analysis/spectral.py` — apply FFT to the full return series, produce a periodogram
- [ ] Identify dominant frequencies: confirm or deny that 4-year, annual, and weekly cycles are genuine spectral peaks vs artefacts
- [ ] If no spectral peak at the halving frequency, the entire seasonal analysis premise should be flagged
- [ ] Save periodogram as a chart to S3 alongside data outputs

---

## 2 · Phase probability — implement the three new techniques

All three live in or alongside `src/analysis/phase_probability.py`. Full specs in `docs/techniques/`.

### 2a · Survival analysis (Weibull + Kaplan-Meier)
See `docs/techniques/01_survival_analysis.md`

- [ ] Add `lifelines` to `requirements.txt`
- [ ] Create `SurvivalPhaseDurationModel` class in `src/analysis/phase_probability.py`
- [ ] Implement `fit(durations, weights, phase)` using `KaplanMeierFitter` and `WeibullFitter`
- [ ] Implement `survival_prob(t)` — P(phase duration > t)
- [ ] Implement `conditional_transition_prob(current_day, horizon_days)` — given we've been in this phase for N days, probability of transitioning within horizon_days
- [ ] Replace Gaussian CDF calls in `PhaseProbabilityEstimator._time_prior()` with survival model calls
- [ ] Add `transition_prob_30d`, `transition_prob_60d`, `transition_prob_90d` columns to `estimate_series()` output
- [ ] Add `km_survival_lower`, `km_survival_upper` (95% CI) columns — honest uncertainty bounds
- [ ] Add `survival_horizon_days: [30, 60, 90]` to `estimation` config block
- [ ] Write tests: S(0)=1, monotone decreasing, conditional prob in [0,1], conditional prob increases with horizon, wide CIs with N=3 (verify, not suppress)

### 2b · HMM on daily returns
See `docs/techniques/02_hmm_daily_returns.md`

- [ ] Add `hmmlearn` to `requirements.txt`
- [ ] Create `src/analysis/hmm_detector.py` — `HMMPhaseDetector` class
- [ ] Implement `fit(enriched, complete_cycles)` — supervised emission initialisation from phase labels, forward-only constrained transition matrix, multi-sequence fitting with `lengths` parameter
- [ ] Implement `predict_proba(enriched, ongoing_label)` — forward algorithm posterior per day
- [ ] Add `rolling_30d_vol` feature computation as a static helper
- [ ] Add ensemble combination: `p_*_ensemble = α×prior + β×hmm + γ×lda` (weights from config)
- [ ] Add `hmm_features`, `hmm_n_iter`, `hmm_transition_self_prob`, `ensemble_weights` to config
- [ ] Add new output columns: `p_*_hmm`, `dominant_phase_hmm`, `p_*_ensemble`, `dominant_phase_ensemble`, `hmm_signals_agree`
- [ ] Update `validate_phase_probability` for new columns
- [ ] Write tests: posteriors sum to 1, non-negative, early cycle = expansion-dominant, multi-sequence lengths tested, raises if not fitted

### 2c · Additional price features + LDA
See `docs/techniques/03_price_features_lda.md`

- [ ] Add `scikit-learn` to `requirements.txt`
- [ ] Create `src/analysis/lda_classifier.py` — `LDAPhaseClassifier` class
- [ ] Implement `_build_features(df, boundaries)` — compute all 6 features: drawdown_from_ath, vol_30d, momentum_7d, recovery_from_trough, drawdown_velocity_7d, days_since_ath_norm
- [ ] Implement `fit(enriched, boundaries, complete_cycles, weights)` — StandardScaler + LinearDiscriminantAnalysis with sample_weight
- [ ] Implement `predict_proba(enriched, ongoing_label, boundaries)` — returns per-day LDA posteriors with NaN for early rows (< 30 days rolling history)
- [ ] Replace Naïve Bayes `_price_posterior()` in `PhaseProbabilityEstimator` with LDA classifier
- [ ] Add `lda_features` and `lda_min_history_days` to config
- [ ] Add new output columns: `p_*_lda`, `dominant_phase_lda`
- [ ] Write tests: proba sums to 1, non-negative, all 6 features created by `_build_features`, NaN handling for early rows, weighted fit differs from unweighted

---

## 3 · Code quality

- [ ] Add type annotations to all public method signatures across `src/` (most are missing return types and parameter annotations)
- [ ] Add `mypy` to dev dependencies and configure `pyproject.toml` with `[tool.mypy]` — enforce `strict = true` at minimum for new files
- [ ] Add `ruff` (replaces flake8 + isort + pyupgrade) to dev dependencies — configure in `pyproject.toml`
- [ ] Add `black` formatter to dev dependencies — configure line-length 100 in `pyproject.toml`
- [ ] Add `pre-commit` — configure `.pre-commit-config.yaml` with ruff, black, mypy hooks
- [ ] Migrate from `setup.py` to `pyproject.toml` (PEP 517/518 standard) — keep `setup.py` only if needed for editable install compatibility
- [ ] Pin dependency versions in `requirements.txt` (e.g. `pandas==2.2.3`) — unpinned deps break reproducibility
- [ ] Create `requirements-dev.txt` properly: `lifelines`, `hmmlearn`, `scikit-learn`, `scipy`, `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `pre-commit`, `matplotlib`
- [ ] Fix the `_phase_status` unused `cycle_id` parameter (Pylance flagged) in `cycle_analyzer.py:21`

---

## 4 · Documentation

- [ ] **[BLOCKING]** Rewrite `README.md` — current version is completely outdated (describes old fixed-threshold phase system, missing empirical detection, phase probability, phase table; row counts and column lists are wrong)
- [ ] Update `CLAUDE.md` — add new modules (`phase_probability.py`, `cycle_analyzer.py`, `lda_classifier.py`, `hmm_detector.py`), new pipeline steps, new config keys
- [ ] Add architecture diagram to README — Mermaid flowchart of the full pipeline (data flow from S3 → enricher → cycle_analyzer → aggregator → probability estimator → S3)
- [ ] Add S3 output table to README — list all 5 parquet files with their row count, column count, and purpose
- [ ] Create `CHANGELOG.md` — document what changed and when; start from the empirical phase detection refactor
- [ ] Create `CONTRIBUTING.md` — how to add a new halving to config, how to run tests, how to add a new analysis module
- [ ] Add docstring to every public method that is missing one — priority: `DataLoader`, `s3_utils`, `config_reader`
- [ ] Add data dictionary doc — one table per output parquet, every column, its type, range, and meaning

---

## 5 · Testing

### 5a · Coverage and quality
- [ ] Add `pytest-cov` and configure minimum coverage threshold (target: 80% for `src/analysis/`, 90% for `src/utils/`)
- [ ] Add coverage report to CI — fail if below threshold
- [ ] Add `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]` — set `testpaths`, `addopts = "--cov=src --cov-report=term-missing"`, timeout per test

### 5b · Missing unit tests
- [ ] `test_cycle_analyzer.py` — `CycleAnalyzer.compute_boundaries`, `stamp_phases`, `build_phase_table` are not unit tested (contract test exists but requires S3)
- [ ] `test_validators.py` for `validate_phase_table` and `validate_phase_probability` — currently only `validate_enriched` and `validate_profile` are tested in `test_schema.py`
- [ ] `test_config_reader.py` — `ConfigReader` is entirely untested
- [ ] `test_s3_utils.py` — mock boto3, test `read_s3_file` and `upload_df_to_s3` error paths
- [ ] `test_aggregator_with_phases.py` — aggregator with empirically-labeled phases (current tests predate empirical phasing)

### 5c · Integration tests
- [ ] Create `tests/integration/test_pipeline_e2e.py` — mock S3 with `moto` library, run full `main()`, assert all 5 parquet files are written to the mock bucket with correct schemas
- [ ] Create `tests/integration/test_probability_estimator_fit.py` — fit `PhaseProbabilityEstimator` on synthetic multi-cycle data, assert fitted parameters are sensible
- [ ] Add `moto` and `pytest-mock` to dev dependencies

### 5d · Regression tests
- [ ] After the walk-forward backtest is implemented, add a regression test that asserts directional accuracy does not drop below a documented baseline between pipeline versions

---

## 6 · Packaging and dependencies

- [ ] Create `pyproject.toml` as the single source of build metadata (migrate from `setup.py`)
- [ ] Pin all runtime dependencies with exact versions; document which versions were tested
- [ ] Add `scipy` to `requirements.txt` immediately (blocking — see section 0)
- [ ] Add `lifelines>=0.27`, `hmmlearn>=0.3`, `scikit-learn>=1.4` to `requirements.txt` when their respective modules are implemented
- [ ] Create `Makefile` with standard targets:
  ```makefile
  make install        # pip install -e .[dev]
  make test           # pytest tests/unit tests/contract
  make test-all       # pytest tests/ (includes integration, requires AWS)
  make lint           # ruff check src/ tests/
  make format         # black src/ tests/
  make typecheck      # mypy src/
  make run            # python src/main.py
  make docker-build   # docker build -t btc-seasonal-analysis .
  make docker-run     # docker run with env vars
  ```

---

## 7 · Docker

- [ ] Create `Dockerfile`:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY src/ ./src/
  COPY config/ ./config/
  COPY conftest.py .
  ENV PYTHONPATH=/app/src
  CMD ["python", "src/main.py"]
  ```
- [ ] Create `.dockerignore` — exclude `venv/`, `logs/`, `tests/`, `exploration/`, `.git/`, `*.pyc`
- [ ] Create `docker-compose.yml` for local development:
  ```yaml
  services:
    pipeline:
      build: .
      environment:
        - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
        - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
        - AWS_DEFAULT_REGION=us-east-1
      volumes:
        - ./logs:/app/logs   # persist logs to host
    localstack:              # mock S3 for local testing without real AWS
      image: localstack/localstack
      ports: ["4566:4566"]
      environment:
        - SERVICES=s3
  ```
- [ ] Create `.env.example` — document every required environment variable:
  ```
  AWS_ACCESS_KEY_ID=
  AWS_SECRET_ACCESS_KEY=
  AWS_DEFAULT_REGION=us-east-1
  # Optional: override log level
  LOG_LEVEL=INFO
  ```
- [ ] Test Docker build and run locally before pushing to CI
- [ ] Push Docker image to AWS ECR (create ECR repository) — tag with git commit SHA and `latest`

---

## 8 · CI/CD pipeline (GitHub Actions)

Create `.github/workflows/` with three workflows:

### 8a · `ci.yml` — runs on every PR and push to main
```yaml
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps: [checkout, setup-python, install-dev, ruff check, black --check, mypy]
  unit-tests:
    runs-on: ubuntu-latest
    steps: [checkout, setup-python, install, pytest tests/unit tests/contract --cov]
  docker-build:
    runs-on: ubuntu-latest
    steps: [checkout, docker build (no push)]
```

### 8b · `release.yml` — runs on tag push (`v*.*.*`)
```yaml
on: push (tags: v*.*.*)
jobs:
  build-and-push:
    steps: [checkout, configure AWS credentials, login to ECR, docker build, docker push (tagged + latest)]
```

### 8c · `scheduled-run.yml` — daily pipeline execution
```yaml
on:
  schedule: [{cron: "0 6 * * *"}]   # 06:00 UTC daily
  workflow_dispatch:                  # allow manual trigger
jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - configure AWS credentials (OIDC — no long-lived secrets)
      - pull Docker image from ECR
      - docker run (pipeline)
      - notify on failure (SNS or Slack webhook)
```

### 8d · Secrets to configure in GitHub repository settings
- `AWS_ROLE_ARN` — IAM role for OIDC (preferred over access keys)
- `ECR_REGISTRY` — ECR registry URL
- `SLACK_WEBHOOK_URL` — for failure notifications

---

## 9 · Orchestration

The scheduled GitHub Actions workflow handles simple daily scheduling. For more robust orchestration:

- [ ] Create IAM role for pipeline execution — `btc-seasonal-analysis-pipeline` with:
  - S3 read on `crypto-historical-price-faadab22`
  - S3 read/write on `btc-seasonal-analysis-faadab22`
  - CloudWatch Logs write
  - No other permissions
- [ ] Configure OIDC trust between GitHub Actions and the IAM role — eliminates long-lived AWS credentials from GitHub secrets
- [ ] **Optional: AWS EventBridge + ECS (if pipeline needs to run independently of GitHub)**
  - Create EventBridge rule: daily cron at 06:00 UTC → trigger ECS task
  - Create ECS task definition pointing at ECR image
  - ECS Fargate (serverless — no EC2 to manage)
  - This is more robust than GitHub Actions for long-running pipeline (no 6-hour job limit)
- [ ] **Optional: AWS Step Functions** — if pipeline needs retry logic, parallel steps, or conditional branching
  - Define state machine: Load → Enrich → Boundaries → StampPhases → Aggregate → PhaseTable → Probability → Validate → Upload
  - Each step is a Lambda or ECS task
  - Built-in retry with exponential backoff, error catching, notifications

---

## 10 · Monitoring and alerting

### 10a · Pipeline execution metrics
- [ ] Emit CloudWatch custom metrics from `main.py` after each run:
  - `PipelineSuccess` (0/1)
  - `PipelineDurationSeconds`
  - `RowsProcessed` (enriched table row count)
  - `S3UploadCount` (number of files uploaded)
  - `EnrichedRowCount`, `ProfileRowCount`, `PhaseTableRowCount`, `ProbabilityRowCount`
- [ ] Create CloudWatch dashboard: pipeline success rate, duration trend, output sizes over time

### 10b · Alerting
- [ ] **Pipeline failure alert**: CloudWatch alarm on `PipelineSuccess = 0` → SNS → email
- [ ] **Data staleness alert**: CloudWatch alarm if `btc_enriched.parquet` S3 LastModified > 36 hours ago (pipeline may have silently not run) → SNS → email
- [ ] **Schema validation failure alert**: log a `VALIDATION_FAILED` metric in `validate_*` functions; alarm on this metric
- [ ] **Row count anomaly alert**: alarm if `EnrichedRowCount` changes by more than ±1% between runs (unexpected data loss or duplication)
- [ ] **Phase detection anomaly alert**: log a `PHASE_DETECTION_ANOMALY` metric if the ongoing cycle's peak_day or trough_day changes by more than 30 days from the previous run (would indicate a detection instability)
- [ ] Create SNS topic `btc-seasonal-alerts`, subscribe team email

### 10c · Data quality monitoring
- [ ] Add a daily data quality check as a separate lightweight pipeline step:
  - Source data freshness: is `bronze/btc.parquet` up to date? (last row date within 2 days of today)
  - Expected row count range per output table
  - Distribution drift: compare `avg_return_pct` distribution for the current cycle against historical — alert if mean shifts > 2σ from prior runs
- [ ] Log all quality check results to CloudWatch; alert on failures

---

## 11 · Security

- [ ] **Remove all hardcoded AWS credentials** — confirm no credentials are in any file (grep for `AWS_ACCESS_KEY_ID`, `AKIA`)
- [ ] Switch to IAM role + OIDC — no long-lived access keys anywhere
- [ ] Enable S3 bucket versioning on both buckets — allows recovery from accidental overwrites
- [ ] Enable S3 server-side encryption (SSE-S3 or SSE-KMS) on both buckets
- [ ] Add S3 bucket policy denying public access (block public ACLs)
- [ ] Add `dependabot.yml` to auto-PR dependency updates for security patches
- [ ] Run `pip-audit` or `safety check` in CI to catch vulnerable dependencies
- [ ] Rotate any existing long-lived AWS access keys after OIDC is set up

---

## Summary: implementation order

```
Phase 1 — Fix defects (section 0)          1–2 days
Phase 2 — Seasonal significance (1a, 1b)   3–4 days
Phase 3 — Walk-forward backtest (1d)       2–3 days
Phase 4 — Code quality + docs (3, 4)       3–4 days
Phase 5 — Testing gaps (5)                 3–4 days
Phase 6 — Survival analysis (2a)           2–3 days
Phase 7 — Docker + CI/CD (7, 8)            3–4 days
Phase 8 — LDA + HMM (2b, 2c)              4–5 days
Phase 9 — Orchestration + monitoring (9, 10) 3–4 days
Phase 10 — Security hardening (11)         1–2 days

Total estimated: ~25–35 working days
```

**The highest-leverage items that make the biggest difference:**
1. `requirements.txt` fix (30 minutes — blocking everything)
2. Statistical significance on seasonal profile (makes outputs trustworthy)
3. Walk-forward backtest (determines if the pipeline has any predictive value at all)
4. Docker + GitHub Actions CI (production deployment and daily execution)
5. Alerting on pipeline failure (you need to know when it breaks)

Items 1–5 alone make the project production-credible. The probability estimator techniques (survival, HMM, LDA) are research improvements on top of a working base.
