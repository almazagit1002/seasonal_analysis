# On-Chain Metrics Integration — Separate Project Spec

**Status**: Nice to have — separate project  
**Project name (suggested)**: `btc_cycle_intelligence`  
**Relationship to `seasonal_analysis`**: consumes its outputs (`btc_phase_table.parquet`, `btc_phase_probability.parquet`) as inputs; does not modify or duplicate the seasonal analysis pipeline

---

## Why this is a separate project

On-chain metrics require:

1. **A new data source** — Glassnode, Cryptoquant, or a self-hosted Bitcoin node. This is categorically different from price data and cannot be added to the existing S3 bronze layer without architectural decisions about rate limits, API credentials, and data freshness.

2. **A different ingestion cadence** — on-chain data updates hourly or daily, but some metrics (like exchange flows) require near-real-time feeds. The `seasonal_analysis` pipeline is designed as a batch daily job.

3. **Different output consumers** — on-chain analytics is a trading/research product, not a seasonal aggregation product. Its outputs (real-time phase scores, cycle regime alerts) go to different downstream systems than the seasonal profile tables.

4. **Avoiding scope creep** — `seasonal_analysis` has a clean bounded scope. Embedding on-chain logic would balloon the codebase and make the project harder to reason about.

The separation also creates a clean dependency direction: `btc_cycle_intelligence` reads `seasonal_analysis` outputs and adds on-chain context. `seasonal_analysis` never imports from `btc_cycle_intelligence`. This is one-way and explicit.

---

## What on-chain metrics add that price cannot

Price is a *result* of investor behaviour. On-chain data reveals the *mechanism*:

| Metric | What it measures | Why price can't show this |
|---|---|---|
| MVRV Ratio | Market cap vs realised cap (aggregate cost basis) | Price doesn't show whether holders are in profit |
| SOPR | Profit/loss on coins spent today | Price doesn't show at what cost today's sellers bought |
| Exchange Inflows/Outflows | Coins moving to/from exchanges | Price doesn't distinguish selling intent from holding |
| Long-term Holder Supply | Supply held >155 days (conviction holders) | Price doesn't show who's holding vs who's transacting |
| NVT Ratio | Network value vs on-chain transaction volume | Price doesn't show whether valuation is supported by usage |
| Realised Price | Average cost basis of all circulating supply | Price doesn't show the "fair value" anchor of the market |
| Puell Multiple | Daily miner revenue vs 365-day average | Price doesn't show miner economic stress (forced sellers) |

Historical observation: MVRV > 3.5 has preceded every major DISTRIBUTION phase. MVRV < 1.0 has coincided with every ACCUMULATION bottom. These thresholds are not guaranteed to hold forever but provide a fundamentally different signal from price dynamics alone.

---

## Project architecture

```
btc_cycle_intelligence/
├── src/
│   ├── main.py                        # pipeline entry point
│   ├── data/
│   │   ├── onchain_loader.py          # Glassnode/Cryptoquant API client
│   │   └── seasonal_loader.py         # reads seasonal_analysis S3 outputs
│   ├── analysis/
│   │   ├── onchain_enricher.py        # computes derived metrics (MVRV bands, etc.)
│   │   ├── signal_aggregator.py       # combines on-chain signals into composite score
│   │   └── cycle_intelligence.py      # merges with phase probabilities → final output
│   └── utils/
│       ├── config_reader.py
│       ├── validators.py
│       ├── s3_utils.py
│       └── logger.py
├── config/
│   └── config.yaml
└── tests/
```

---

## Data sources

### Option A: Glassnode API (recommended starting point)

**Coverage**: all major on-chain metrics, daily granularity, well-documented.  
**Cost**: ~$29/month for the Studio tier (covers MVRV, SOPR, exchange flows).  
**Rate limits**: 1 request/second on Studio tier.  
**Auth**: API key in environment variable `GLASSNODE_API_KEY`.

Key endpoints:
```
GET /v1/metrics/market/mvrv_z_score          → MVRV Z-score daily
GET /v1/metrics/indicators/sopr              → Spent Output Profit Ratio
GET /v1/metrics/distribution/exchange_net_position_change → exchange net flows
GET /v1/metrics/supply/lth_supply            → long-term holder supply
GET /v1/metrics/indicators/nvt               → NVT ratio
GET /v1/metrics/mining/puell_multiple        → Puell multiple
GET /v1/metrics/market/price_realized_usd    → realised price
```

### Option B: Cryptoquant API

Similar coverage, slightly different metric definitions. Good alternative for exchange flow data.

### Option C: Self-hosted Bitcoin node + custom indexer

Maximum control and zero API cost, but requires running a full node (~600GB storage), an indexer (electrs or similar), and custom metric computation. Only worthwhile if the project scales significantly.

### Option D: CoinMetrics Community API

Free tier with limited metrics (MVRV not available free). Useful for NVT and basic on-chain volume.

---

## Pipeline design

```
Schedule: daily at 06:00 UTC (after seasonal_analysis pipeline completes)

Step 1: Load
  - seasonal_loader.py reads:
      btc-seasonal-analysis-faadab22/btc_phase_probability.parquet
      btc-seasonal-analysis-faadab22/btc_phase_table.parquet
  - onchain_loader.py fetches the last N days of each metric from Glassnode API

Step 2: Enrich
  - onchain_enricher.py computes derived signals:
      mvrv_signal:    MVRV Z-score percentile vs historical distribution per phase
      sopr_signal:    30-day smoothed SOPR > 1 / < 1 flag
      exchange_signal: net exchange flow direction (inflow = bearish)
      lth_signal:     LTH supply trend (accumulating vs distributing)

Step 3: Aggregate into composite score
  - signal_aggregator.py:
      composite_score = weighted_mean(mvrv_signal, sopr_signal, exchange_signal, lth_signal)
      score range: [-1, +1]  where +1 = strong EXPANSION signal, -1 = strong DISTRIBUTION

Step 4: Merge with phase probabilities
  - cycle_intelligence.py:
      For each day: take phase probability from seasonal_analysis
      Update with on-chain composite score as additional likelihood
      Output: final_phase_probs + on_chain_composite + signal_breakdown

Step 5: Validate and upload
  - btc-cycle-intelligence-{account_id}/btc_cycle_intelligence.parquet
```

---

## Output schema

**`btc_cycle_intelligence.parquet`** — one row per day for the ongoing cycle:

| Column | Type | Description |
|---|---|---|
| `date` | DatetimeIndex | date |
| `cycle_day` | int | days since halving |
| `mvrv_z_score` | float | raw Glassnode MVRV Z-score |
| `mvrv_signal` | float | normalised [0,1] vs historical phase distribution |
| `sopr_30d` | float | 30-day smoothed SOPR |
| `sopr_signal` | float | normalised [-1,1] |
| `exchange_net_flow_btc` | float | daily BTC net exchange flow |
| `exchange_signal` | float | smoothed direction [-1,1] |
| `lth_supply_pct` | float | % of circulating supply held by LTH |
| `lth_signal` | float | trend normalised [-1,1] |
| `composite_onchain_score` | float | weighted composite [-1,1] |
| `p_expansion_seasonal` | float | from btc_phase_probability.parquet (time prior) |
| `p_distribution_seasonal` | float | |
| `p_accumulation_seasonal` | float | |
| `p_expansion_final` | float | seasonal × on-chain posterior |
| `p_distribution_final` | float | |
| `p_accumulation_final` | float | |
| `dominant_phase_final` | str | most likely phase combining all signals |
| `cycle_regime_alert` | str | `null` \| `"distribution_warning"` \| `"accumulation_signal"` |

---

## Key implementation details

### MVRV integration

MVRV historically exceeds 3.5 at cycle peaks and falls below 1.0 at cycle troughs. Rather than using the raw threshold, compute a percentile:

```python
# For each day, compute: what percentile is today's MVRV within the
# historical MVRV distribution during each confirmed phase?
# P(MVRV | EXPANSION) fits a distribution to MVRV values during confirmed EXPANSION days
# Similarly for DISTRIBUTION and ACCUMULATION
# Then: mvrv_signal per phase = N(mvrv_today; mu_phase, sigma_phase).pdf(...)
# Same Gaussian likelihood approach as the drawdown signal in seasonal_analysis
```

This avoids hard thresholds (which may shift as BTC matures) and instead treats MVRV as another Gaussian-likelihood feature.

### SOPR smoothing

Raw SOPR is very noisy (single-day transactions dominate). Use a 30-day exponential moving average before computing signals.

### On-chain composite score weights

Initial weights (tune on historical cycles):

```yaml
onchain_signal_weights:
  mvrv:     0.35   # strongest historical signal
  sopr:     0.25
  exchange: 0.20
  lth:      0.20
```

Store in config. Weights should be revalidated each time a new cycle completes.

### Combining with seasonal_analysis probabilities

The seasonal analysis already outputs `p_expansion_prior`, `p_distribution_prior`, `p_accumulation_prior`. The on-chain project treats these as the time prior and applies the composite on-chain score as a likelihood update using the same Naïve Bayes structure:

```python
# composite_score in [-1, +1]
# Map to likelihood vector for [EXPANSION, DISTRIBUTION, ACCUMULATION]:
# score = +1 → L = [high, low, low] → boost EXPANSION
# score = -1 → L = [low, high, low] → boost DISTRIBUTION
# score = 0  → L = [1/3, 1/3, 1/3] → no update

def composite_to_likelihood(score: float) -> list[float]:
    """Map composite on-chain score to phase likelihood vector."""
    # Softmax-style mapping: score drives the EXPANSION vs DISTRIBUTION split
    l_exp  = np.exp( score * 2.0)
    l_dist = np.exp(-score * 2.0)
    l_accum = 1.0  # on-chain doesn't distinguish ACCUM from late DIST well
    total = l_exp + l_dist + l_accum
    return [l_exp/total, l_dist/total, l_accum/total]
```

### Cycle regime alerts

Generate alerts when the on-chain composite score crosses actionable thresholds:

| Alert | Condition |
|---|---|
| `distribution_warning` | composite_score < -0.6 AND dominant_phase_seasonal ≠ "DISTRIBUTION" |
| `accumulation_signal` | composite_score > 0.3 AND p_accumulation_seasonal > 0.4 |
| `mvrv_peak_zone` | mvrv_z_score > 3.5 |
| `mvrv_trough_zone` | mvrv_z_score < 0.5 |

Alerts are written to the output table as a string column; no push notifications in the pipeline itself (that's a downstream consumer concern).

---

## Config for `btc_cycle_intelligence`

```yaml
data:
  glassnode_api_key_env: "GLASSNODE_API_KEY"
  glassnode_base_url: "https://api.glassnode.com"
  metrics:
    - { name: "mvrv_z_score",  endpoint: "/v1/metrics/market/mvrv_z_score" }
    - { name: "sopr",          endpoint: "/v1/metrics/indicators/sopr" }
    - { name: "exchange_flow", endpoint: "/v1/metrics/distribution/exchange_net_position_change" }
    - { name: "lth_supply",    endpoint: "/v1/metrics/supply/lth_supply" }
  lookback_days: 90   # fetch last N days on each run

seasonal_analysis:
  bucket: "btc-seasonal-analysis-faadab22"
  phase_prob_key: "btc_phase_probability.parquet"
  phase_table_key: "btc_phase_table.parquet"

output:
  bucket: "btc-cycle-intelligence-{account_id}"
  intelligence_key: "btc_cycle_intelligence.parquet"

onchain_signal_weights:
  mvrv:     0.35
  sopr:     0.25
  exchange: 0.20
  lth:      0.20

alert_thresholds:
  distribution_warning: -0.60
  accumulation_signal:   0.30
  mvrv_peak_zone:        3.50
  mvrv_trough_zone:      0.50
```

---

## Estimated effort

| Task | Effort |
|---|---|
| Glassnode API client + caching | 2–3 days |
| On-chain enricher (metric normalisation) | 2 days |
| Signal aggregator | 1 day |
| Merge with seasonal_analysis outputs | 1 day |
| Validators + tests | 2 days |
| Config + pipeline wiring | 1 day |
| **Total** | **~10 days** |

The main uncertainty is the Glassnode API data quality and whether the historical metric values align with the phase labels from `seasonal_analysis`. A 1-week exploratory phase (notebook-driven, in `exploration/onchain/`) before building the pipeline is strongly recommended.

---

## Risks

1. **API dependency**: Glassnode is a commercial API. Rate limits, pricing changes, or endpoint deprecation could break the pipeline. Cache all fetched data in S3 immediately after fetching.

2. **Metric stationarity**: MVRV historically had clear phase signals, but as BTC becomes an institutional asset, investor cost basis distributions may shift. The 3.5 threshold may not hold forever — use percentile-based signals rather than fixed thresholds.

3. **Lookahead bias in backtesting**: when validating on-chain signals against past phases, ensure the on-chain data for date T was actually available at date T (some metrics are revised retroactively). Use timestamp of data publication, not data date.

4. **Phase label dependency**: the on-chain signal classifier trains on phase labels from `seasonal_analysis`. Any errors in phase boundary detection (from `phase_detector.py`) propagate into the on-chain model. Always compare on-chain signal distributions between phases visually before training.
