# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
source venv/bin/activate
pip install -e .
```

## Commands

```bash
# Run the full pipeline
python src/main.py

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/unit/test_enricher.py

# Run a single test
pytest tests/unit/test_enricher.py::test_cycle_phase_boundaries
```

The pipeline requires AWS credentials in the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, or an IAM role) — it reads from and writes to S3.

## Architecture

The pipeline runs in `src/main.py` and executes five sequential steps:

1. **Load** — `DataLoader` reads a Parquet file from the bronze S3 bucket, sorts by date, enforces daily frequency with forward-fill.
2. **Enrich** — `DataEnricher` stamps each row with halving-cycle dimensions (`days_since_halving`, `cycle_month`, `cycle_phase`) and calendar dimensions (`calendar_year`, `calendar_month`, `day_of_week`, etc.). Rows before the first halving get NaN for all cycle columns.
3. **Aggregate** — `SeasonalAggregator` groups the enriched table by `(cycle_month, cycle_phase, calendar_month, month_name, day_of_week, day_name)` and computes `avg_return_pct`, `median_return_pct`, `volatility_pct`, `win_rate_pct`, and `sample_size`.
4. **Validate** — `validate_enriched` and `validate_profile` in `src/utils/validators.py` assert schema and data-quality invariants. The pipeline aborts rather than uploading malformed data.
5. **Upload** — both DataFrames are serialised to Parquet and written to the silver/gold keys in the output S3 bucket.

### Key design decisions

- **Config-driven thresholds**: halving dates, cycle phase boundaries (`bull_days`/`bear_days`), and S3 bucket/key paths all live in `config/config.yaml`. Nothing is hardcoded in the analysis classes.
- **Cycle phase logic**: `EXPANSION` = days 0–450, `DISTRIBUTION` = 451–800, `ACCUMULATION` = 801+. Future halvings in the config are excluded from the reference-point calculation so the current cycle is always bounded by the most recent past halving.
- **src layout**: `conftest.py` at the root inserts `src/` into `sys.path`, which is how test imports resolve without installing the package first.
- **Logging**: `get_logger(name, log_to_file=True)` writes timestamped logs to `src/logs/`. The file-handler path is relative to wherever the process is invoked, so run from the repo root.

### Package layout

```
src/
  main.py                  # pipeline entry point
  data/data_loader.py      # S3 read + time-series preparation
  analysis/
    enricher.py            # cycle + calendar stamping
    aggregator.py          # groupby → seasonal profile
  utils/
    config_reader.py       # YAML loader (resolves path relative to src/)
    validators.py          # schema contracts for both output tables
    s3_utils.py            # boto3 read/upload helpers
    logger.py              # shared logging setup
config/config.yaml         # all tunable parameters
tests/
  unit/                    # pure-logic tests (mock ConfigReader via patch)
  contract/                # schema/validator tests
```

### S3 buckets

| Bucket | Purpose |
|---|---|
| `crypto-historical-price-faadab22` | Source — `bronze/btc.parquet` (raw price history) |
| `btc-seasonal-analysis-faadab22` | Output — `btc_enriched.parquet` (silver), `btc_seasonal_profile.parquet` (gold) |

### Output schemas

**Enriched table** (`btc_enriched.parquet`) — one row per calendar day, 4,757 rows × 11 columns. Cycle columns are NaN for dates before November 2012.

**Seasonal profile** (`btc_seasonal_profile.parquet`) — one row per `(cycle_month × cycle_phase × calendar_month × day_of_week)` combination, 1,811 rows × 11 columns. Rows with `sample_size < 3` are anecdotal (only 3 completed halving cycles exist in the data).
