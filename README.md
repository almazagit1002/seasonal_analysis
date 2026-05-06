# BTC Seasonal Analysis Pipeline

## Setup
```bash
source venv/bin/activate
pip install -e .
```

Run the pipeline:
```bash
python src/main.py
```

Run tests:
```bash
pytest tests/
```

---

## What the pipeline does

The pipeline reads daily BTC price history from S3, enriches every row with cycle and calendar context, aggregates it into a seasonal profile, validates both outputs, and uploads them as Parquet files.

### Step 1 — Enrich (`btc_enriched.parquet`)

Every calendar day in the price history gets stamped with:

| Column | Description |
|---|---|
| `price` | Daily closing price |
| `daily_return_pct` | Day-over-day % return |
| `days_since_halving` | Calendar days elapsed since the most recent BTC halving |
| `cycle_month` | 30-day bucket within the 4-year cycle (1–48) |
| `cycle_phase` | Named macro regime: EXPANSION (days 0–450), DISTRIBUTION (451–800), ACCUMULATION (801+) |
| `calendar_year` | Calendar year |
| `calendar_month` | Calendar month number (1–12) |
| `month_name` | Calendar month name (Jan–Dec) |
| `day_of_week` | Weekday number (0=Monday, 6=Sunday) |
| `day_name` | Weekday name |

Rows before the first halving (November 2012) have NaN for all cycle columns. Future halvings are excluded from the reference point calculation.

Phase thresholds and halving dates are config-driven (`config/config.yaml`), not hardcoded.

### Step 2 — Aggregate (`btc_seasonal_profile.parquet`)

The enriched table is grouped by `(cycle_month, cycle_phase, calendar_month, day_of_week)` to produce a lookup table of historical seasonal tendencies.

Each row answers: *"When BTC was at this point in its 4-year cycle, on this month and weekday — what happened?"*

| Column | Description |
|---|---|
| `avg_return_pct` | Average daily return across all matching observations |
| `median_return_pct` | Median daily return (less sensitive to outliers) |
| `volatility_pct` | Standard deviation of daily returns |
| `win_rate_pct` | % of days with a positive return |
| `sample_size` | Number of historical observations (typically 1–6, one per completed cycle) |

The profile has **1,811 rows** drawn from 3 completed halving cycles (2012, 2016, 2020). Rows with `sample_size < 3` should be treated as anecdotal.

### Step 3 — Validate & Upload

Before uploading, both DataFrames are validated against schema contracts (`src/utils/validators.py`). If any check fails the pipeline aborts — nothing malformed lands in S3.

```
s3://btc-seasonal-analysis-faadab22/
├── btc_enriched.parquet          ← 4,757 rows × 11 columns
└── btc_seasonal_profile.parquet  ← 1,811 rows × 11 columns
```

---

## Future analysis

### Statistical rigour
- **P-values and confidence intervals** — the current seasonal edges are raw averages with no significance testing. With only 3 cycles, most cells fail a t-test. Adding p-values would let consumers filter out noise from real signal.
- **Bootstrap resampling** — resample within each cycle to build empirical confidence intervals around `avg_return_pct` without assuming normality.
- **Seasonality strength tests** — formal statistical tests (Canova-Hansen, HEGY) to confirm whether the observed seasonal patterns are statistically present in the series at all before building models on top of them.

### Classical time series models

- **SARIMA (Seasonal ARIMA)** — extends ARIMA with an explicit seasonal component `(P, D, Q, s)`. Useful for modelling autocorrelation in returns at known seasonal frequencies (weekly `s=7`, annual `s=365`). Can generate point forecasts with confidence intervals. Main limitation for BTC: assumes linear relationships and stationary variance.
- **STL decomposition (LOESS)** — more robust alternative to classical multiplicative decomposition. Handles outliers and allows the seasonal component to evolve over time, which matters for an asset whose behaviour changed dramatically across cycles. Produces cleaner trend/seasonal/residual separation than statsmodels `seasonal_decompose`.
- **GARCH / EGARCH** — BTC returns exhibit strong volatility clustering (large moves follow large moves). A GARCH(1,1) model captures the conditional variance structure that the current pipeline ignores entirely. EGARCH adds asymmetry — negative shocks increase volatility more than positive ones of the same size. Critical for any risk-adjusted view of seasonal edges.

### Modern decomposition and frequency-domain analysis

- **Fourier series decomposition** — represents the seasonal component as a sum of sine and cosine waves at different frequencies. More flexible than fixed-period decomposition: you can model annual, halving-cycle, and weekly seasonality simultaneously in a single regression. The coefficients directly quantify the strength of each periodic component.
- **Periodogram / FFT spectral analysis** — applies the Fast Fourier Transform to the return series to identify dominant frequencies in the data. This would empirically confirm (or reject) whether 4-year, annual, and weekly cycles are truly present as spectral peaks rather than assumed by construction.
- **Wavelet analysis** — unlike FFT which gives a global frequency picture, wavelets localise frequency content in time. This means you can ask "was the annual seasonal pattern stronger in 2016–2020 than in 2020–2024?" — something a static Fourier decomposition cannot answer. Useful for detecting whether cycle patterns are strengthening or fading as the asset matures.

### Probabilistic and ML forecasting

- **Prophet (Meta)** — purpose-built for time series with multiple overlapping seasonalities. Handles yearly, weekly, and user-defined periodicities (e.g. the 4-year halving cycle as a custom regressor) in one model. Robust to missing data and outliers, produces uncertainty intervals, and allows explicit modelling of known events (halvings, regulatory announcements) as changepoints. Likely the fastest path from the current pipeline to a usable forecast.
- **LSTM / Temporal Convolutional Networks** — deep learning models that learn complex non-linear seasonal patterns across multiple time scales without manual feature engineering. Require significantly more data than we currently have for reliable training; more appropriate once hourly data is available.
- **XGBoost / LightGBM with seasonal features** — tree-based models using engineered inputs: `sin(2π × day_of_year / 365)`, `cos(2π × cycle_month / 48)`, rolling returns, and volatility features. Faster to train and more interpretable than neural networks, and competitive in accuracy on financial time series.
- **Gaussian Process regression** — provides full predictive distributions rather than point forecasts. Particularly well-suited here because the kernel function can encode prior beliefs about the periodicity of the halving cycle and annual seasonality directly into the model structure.

### Regime detection

- **Hidden Markov Models (HMM)** — instead of hardcoding EXPANSION / DISTRIBUTION / ACCUMULATION based on fixed day thresholds, HMMs learn latent market regimes directly from return and volatility data. The discovered regimes may not align with halving boundaries at all — which is either a useful correction or evidence that the halving cycle is a valid driver, depending on the result.
- **Markov-switching GARCH** — combines regime detection with volatility modelling. Estimates separate GARCH parameters per regime, capturing the fact that BTC volatility behaves very differently in bull vs. bear phases.

### Calendar anomalies

- **Week-of-month effect** — the first and last week of each month behave differently due to institutional rebalancing, options expiry (monthly and quarterly), and futures settlement. Add `week_of_month` (1–5) as a grouping dimension.
- **Quarter** — collapse `calendar_month` into Q1–Q4 for a coarser but higher-sample view.
- **US Presidential cycle** — BTC has coincidentally aligned with the 4-year US election cycle. Including a `presidential_year` column (1–4) alongside `cycle_month` would allow testing whether macro political seasonality compounds or dilutes the halving effect.
- **Hourly intraday seasonality** — if hourly OHLCV data is available, the same enrichment logic applies at hourly resolution. Known patterns include Asian session weakness and US market-open volatility spikes.

### Richer cycle modelling

- **Recency-weighted profile** — the 2012 cycle (BTC under $1,000, largely illiquid) may not be comparable to 2024. Applying exponential decay weights across cycles before computing `avg_return_pct` would reduce the influence of less relevant history.
- **Drawdown profile by cycle phase** — beyond returns, track peak-to-trough drawdown distributions per phase. Useful for position sizing and understanding the risk shape of each regime, not just the average return.
- **Intra-cycle trajectory** — compute a rolling 30-day return within the current cycle and compare it to the same window in prior cycles. This gives a real-time "ahead / behind historical average" signal at any point in the cycle.

### Multi-asset and cross-market

- The `DataEnricher` is asset-agnostic. Adding ETH, SOL, or traditional assets (SPX, Gold, DXY) to the source bucket and running the same pipeline enables cross-asset seasonal correlation analysis, lead-lag detection between BTC cycle position and altcoin performance, and macro regime comparison.
- **On-chain data integration** — combining price seasonality with on-chain metrics (MVRV ratio, NUPL, exchange flows) as additional regressors in any of the models above. On-chain data provides fundamentals context that price-only models lack entirely.

### Validation of signal quality

- **Walk-forward backtest** — for each completed cycle, use the profile built from all *prior* cycles to generate an expected return, then compare to actual. This is the most important future step: it answers whether the seasonal edges are genuinely predictive or just historical coincidence overfitted to 3 cycles.
- **Out-of-sample R²** — a simple measure of how much of the 2024-cycle returns are explained by the 2016+2020 seasonal profile.
- **Sharpe ratio of a seasonality-based strategy** — translate the profile into a mechanical long/short rule and compute risk-adjusted returns, drawdown, and win rate on out-of-sample data.
