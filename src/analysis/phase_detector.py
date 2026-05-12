"""
Empirical BTC halving-cycle phase detection.

Phases are derived from actual price history, not hardcoded day thresholds:
  EXPANSION   : halving day  →  confirmed cycle ATH
  DISTRIBUTION: cycle ATH   →  lowest price after the ATH
  ACCUMULATION: cycle low   →  next halving (or latest data point for ongoing cycle)

All tunable parameters (drawdown thresholds, confirmation windows, halving
dates) are passed in from config — nothing is hardcoded in this module.

The confirmed-peak algorithm:
  Walk forward through the cycle tracking the running maximum.
  A peak is *confirmed* when:
    1. Price drops ≥ drawdown_threshold below the running max.
    2. Over the next confirmation_days, price does NOT recover above
       running_max × (1 – drawdown_threshold / 2).
  If a confirmed peak is later invalidated by a new all-time-high the
  algorithm falls back to the highest price within the first
  fallback_cap_days days.
"""
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__, log_to_file=True)


# ─── cycle assignment ─────────────────────────────────────────────────────────

def assign_cycle_id(df: pd.DataFrame, halvings_cfg: list) -> pd.DataFrame:
    """
    Assign each row to a halving cycle.

    Parameters
    ----------
    df           : enriched DataFrame with days_since_halving and calendar columns
    halvings_cfg : list of {label, date, tolerance_days} from config

    A halving is an eligible snap target when its date is within
    tolerance_days of the dataset's last date.  This gates future halvings
    out of the candidate set until the dataset is close to their expected date.
    """
    logger.info("assign_cycle_id: reconstructing halving dates per row")
    df = df.copy()

    last_date = pd.Timestamp(df.index.max()).replace(tzinfo=None)

    eligible = [
        h for h in halvings_cfg
        if pd.Timestamp(h["date"]) <= last_date + pd.Timedelta(days=int(h["tolerance_days"]))
    ]

    if not eligible:
        logger.warning("No eligible halvings found — all cycle_id set to None")
        df["cycle_id"] = None
        return df

    halving_labels   = [h["label"] for h in eligible]
    halving_dates_np = np.array([h["date"] for h in eligible], dtype="datetime64[D]")

    logger.info(
        f"  eligible halvings: {halving_labels} "
        f"(of {len(halvings_cfg)} total in config)"
    )

    has_cycle = df["days_since_halving"].notna()

    approx_mid = pd.to_datetime(
        df["calendar_year"].astype(str) + "-" +
        df["calendar_month"].astype(str).str.zfill(2) + "-15"
    )
    approx_halving = (
        approx_mid - pd.to_timedelta(df["days_since_halving"].fillna(0), unit="D")
    ).values.astype("datetime64[D]").astype("int64")

    halving_int = halving_dates_np.astype("int64")
    diffs       = np.abs(approx_halving[:, np.newaxis] - halving_int[np.newaxis, :])
    nearest_idx = np.argmin(diffs, axis=1)

    cycle_ids = np.where(has_cycle.values, np.array(halving_labels)[nearest_idx], None)
    df["cycle_id"] = cycle_ids

    for label in halving_labels:
        n = int((df["cycle_id"] == label).sum())
        logger.info(f"  cycle {label}: {n} rows assigned")
    logger.info(f"  pre-halving (no cycle): {int(has_cycle.eq(False).sum())} rows")
    return df


# ─── peak detection ───────────────────────────────────────────────────────────

def find_confirmed_peak(
    prices:             np.ndarray,
    days:               np.ndarray,
    drawdown_threshold: float = 0.40,
    confirmation_days:  int   = 120,
    cycle_id:           str   = "?",
    fallback_cap_days:  int   = 800,
) -> tuple[int, float]:
    """
    Walk-forward confirmed-peak algorithm.

    Parameters
    ----------
    prices             : daily closing prices sorted by days_since_halving
    days               : days_since_halving values (same order)
    drawdown_threshold : fraction below running max that triggers a check
    confirmation_days  : forward window; price must stay below
                         peak × (1 – threshold/2) throughout
    cycle_id           : label used in log messages only
    fallback_cap_days  : search window for the fallback when no peak is confirmed

    Returns
    -------
    (peak_day, peak_price)
    """
    n = len(prices)
    if n == 0:
        raise ValueError(f"[{cycle_id}] empty price array")

    logger.info(f"[{cycle_id}] ── peak search ──────────────────────────────────────")
    logger.info(
        f"[{cycle_id}] rows={n}  "
        f"day_range=[{int(days[0])}, {int(days[-1])}]  "
        f"price_range=[${prices.min():,.0f}, ${prices.max():,.0f}]"
    )
    logger.info(
        f"[{cycle_id}] params: drawdown_threshold={drawdown_threshold:.0%}  "
        f"confirmation_days={confirmation_days}  fallback_cap_days={fallback_cap_days}"
    )

    running_max_price    = float(prices[0])
    running_max_day      = int(days[0])
    confirmed_peak_day   = None
    confirmed_peak_price = None
    n_new_highs          = 0
    n_threshold_hits     = 0

    for i in range(1, n):
        p = float(prices[i])
        d = int(days[i])

        if p > running_max_price:
            n_new_highs += 1
            if confirmed_peak_day is not None:
                logger.warning(
                    f"[{cycle_id}] new ATH ${p:,.0f} at day {d} — "
                    f"invalidates confirmed peak "
                    f"day={confirmed_peak_day} ${confirmed_peak_price:,.0f}"
                )
                confirmed_peak_day   = None
                confirmed_peak_price = None
            else:
                logger.debug(
                    f"[{cycle_id}] new high ${p:,.0f} at day {d} "
                    f"(was ${running_max_price:,.0f} at day {running_max_day})"
                )
            running_max_price = p
            running_max_day   = d
            continue

        drawdown = (running_max_price - p) / running_max_price
        if drawdown < drawdown_threshold:
            continue

        if confirmed_peak_day == running_max_day:
            continue

        n_threshold_hits += 1
        confirm_end  = d + confirmation_days
        fwd_mask     = (days > d) & (days <= confirm_end)
        fwd_prices   = prices[fwd_mask]
        coverage     = len(fwd_prices) / confirmation_days

        logger.info(
            f"[{cycle_id}] drawdown {drawdown:.1%} at day {d} "
            f"(price ${p:,.0f}) from peak ${running_max_price:,.0f} day {running_max_day}  |  "
            f"confirmation window coverage {coverage:.0%} ({len(fwd_prices)}/{confirmation_days} days)"
        )

        if coverage < 0.5:
            logger.info(f"[{cycle_id}] insufficient forward data — candidate rejected")
            continue

        recovery_level = running_max_price * (1.0 - drawdown_threshold / 2.0)
        recovered      = bool(np.any(fwd_prices > recovery_level))

        if recovered:
            best_recovery = float(fwd_prices[fwd_prices > recovery_level].max())
            logger.info(
                f"[{cycle_id}] recovery ${best_recovery:,.0f} detected "
                f"(above recovery threshold ${recovery_level:,.0f}) — "
                f"peak day {running_max_day} rejected"
            )
        else:
            confirmed_peak_day   = running_max_day
            confirmed_peak_price = running_max_price
            logger.info(
                f"[{cycle_id}] PEAK CONFIRMED — "
                f"day {confirmed_peak_day}  price ${confirmed_peak_price:,.0f}  "
                f"(no recovery above ${recovery_level:,.0f} in {len(fwd_prices)}d)"
            )

    logger.info(
        f"[{cycle_id}] scan done — "
        f"new_highs={n_new_highs}  threshold_hits={n_threshold_hits}"
    )

    if confirmed_peak_day is not None:
        logger.info(
            f"[{cycle_id}] RESULT (confirmed): "
            f"day {confirmed_peak_day}  ${confirmed_peak_price:,.0f}"
        )
        return confirmed_peak_day, confirmed_peak_price

    logger.warning(
        f"[{cycle_id}] no valid confirmed peak — "
        f"falling back to highest price in first {fallback_cap_days} days"
    )
    cap_mask      = days <= fallback_cap_days
    search_prices = prices[cap_mask] if cap_mask.any() else prices
    search_days   = days[cap_mask]   if cap_mask.any() else days
    best_idx      = int(np.argmax(search_prices))
    peak_day      = int(search_days[best_idx])
    peak_price    = float(search_prices[best_idx])
    logger.info(f"[{cycle_id}] RESULT (fallback): day {peak_day}  ${peak_price:,.0f}")
    return peak_day, peak_price


# ─── trough detection ─────────────────────────────────────────────────────────

def find_trough_after_peak(
    prices:   np.ndarray,
    days:     np.ndarray,
    peak_day: int,
    cycle_id: str = "?",
) -> tuple[int, float]:
    """Lowest price strictly after peak_day."""
    logger.info(f"[{cycle_id}] ── trough search (after day {peak_day}) ─────────────────")

    post_mask = days > peak_day
    if not post_mask.any():
        logger.warning(
            f"[{cycle_id}] no data after peak day {peak_day} — trough set to peak"
        )
        idx = int(np.searchsorted(days, peak_day, side="left"))
        return peak_day, float(prices[min(idx, len(prices) - 1)])

    post_prices  = prices[post_mask]
    post_days    = days[post_mask]
    trough_idx   = int(np.argmin(post_prices))
    trough_day   = int(post_days[trough_idx])
    trough_price = float(post_prices[trough_idx])

    peak_price_val = float(prices[days <= peak_day].max())
    drawdown       = (peak_price_val - trough_price) / peak_price_val

    logger.info(
        f"[{cycle_id}] trough: day {trough_day}  ${trough_price:,.0f}  "
        f"({drawdown:.1%} below peak)  "
        f"[{int(post_days[0])}–{int(post_days[-1])} searched, {len(post_prices)} days]"
    )
    return trough_day, trough_price


# ─── trough confirmation (ongoing cycle only) ────────────────────────────────

def _is_trough_confirmed(
    prices:        np.ndarray,
    days:          np.ndarray,
    trough_day:    int,
    trough_price:  float,
    recovery_pct:  float = 0.40,
    min_days:      int   = 90,
    cycle_id:      str   = "?",
) -> bool:
    """
    A trough is confirmed when price has recovered >= recovery_pct above the
    provisional trough AND held that level for at least min_days consecutive days.
    """
    recovery_level = trough_price * (1.0 + recovery_pct)
    post_mask      = days > trough_day
    post_prices    = prices[post_mask]
    post_days      = days[post_mask]

    current_price  = float(prices[-1])
    current_day    = int(days[-1])
    days_since     = current_day - trough_day
    recovery_now   = (current_price - trough_price) / trough_price

    logger.info(
        f"[{cycle_id}] trough confirmation check — "
        f"trough ${trough_price:,.0f} day {trough_day}  |  "
        f"recovery_level ${recovery_level:,.0f} (+{recovery_pct:.0%})  |  "
        f"min_days {min_days}  |  "
        f"current ${current_price:,.0f} ({recovery_now:+.1%}) at day {current_day} "
        f"({days_since}d since trough)"
    )

    if len(post_prices) < min_days:
        logger.info(
            f"[{cycle_id}] only {len(post_prices)} post-trough days — "
            f"need {min_days} minimum → NOT confirmed"
        )
        return False

    above = post_prices >= recovery_level
    for i in range(len(above) - min_days + 1):
        if np.all(above[i : i + min_days]):
            run_start_day = int(post_days[i])
            run_end_day   = int(post_days[i + min_days - 1])
            logger.info(
                f"[{cycle_id}] trough CONFIRMED — "
                f"{min_days} consecutive days above ${recovery_level:,.0f} "
                f"(days {run_start_day}–{run_end_day})"
            )
            return True

    max_run = 0
    cur_run = 0
    for a in above:
        cur_run = cur_run + 1 if a else 0
        max_run = max(max_run, cur_run)

    logger.info(
        f"[{cycle_id}] trough NOT confirmed — "
        f"longest run above ${recovery_level:,.0f}: {max_run} days "
        f"(need {min_days})  |  current recovery {recovery_now:+.1%} "
        f"(need +{recovery_pct:.0%})"
    )
    return False


# ─── combined entry point ─────────────────────────────────────────────────────

def compute_cycle_boundaries(
    df:          pd.DataFrame,
    halvings_cfg: list,
    peak_cfg:    dict,
    trough_cfg:  dict,
) -> dict:
    """
    Compute empirical phase boundaries for every occurred halving cycle.

    Parameters
    ----------
    df           : DataFrame with cycle_id, days_since_halving, price columns
    halvings_cfg : list of {label, date, tolerance_days} from config
    peak_cfg     : {drawdown_threshold, confirmation_days, fallback_cap_days}
    trough_cfg   : {recovery_pct, confirmation_days}

    The ongoing cycle is derived dynamically as the most recent halving whose
    date has passed relative to the dataset's last row — no label is hardcoded.

    Returns
    -------
    {cycle_label: {peak_day, peak_price, trough_day, trough_price,
                   max_day, is_complete, trough_confirmed}}
    """
    drawdown_threshold = peak_cfg["drawdown_threshold"]
    confirmation_days  = peak_cfg["confirmation_days"]
    fallback_cap_days  = peak_cfg["fallback_cap_days"]

    logger.info("=" * 70)
    logger.info("compute_cycle_boundaries START")
    logger.info(
        f"  drawdown_threshold={drawdown_threshold:.0%}  "
        f"confirmation_days={confirmation_days}  "
        f"fallback_cap={fallback_cap_days}d"
    )
    logger.info("=" * 70)

    last_date = pd.Timestamp(df.index.max()).replace(tzinfo=None)
    occurred  = [h for h in halvings_cfg if pd.Timestamp(h["date"]) <= last_date]

    if not occurred:
        logger.warning("No occurred halvings in config — returning empty boundaries")
        return {}

    ongoing_label = occurred[-1]["label"]
    logger.info(
        f"  occurred cycles: {[h['label'] for h in occurred]}  "
        f"ongoing: {ongoing_label}"
    )

    result = {}

    for h in occurred:
        cycle_id    = h["label"]
        is_complete = (cycle_id != ongoing_label)

        rows = (
            df[(df["cycle_id"] == cycle_id) & df["days_since_halving"].notna()]
            .sort_values("days_since_halving")
            .dropna(subset=["price"])
        )

        if len(rows) < 2:
            logger.warning(f"[{cycle_id}] only {len(rows)} rows — skipped")
            continue

        logger.info(
            f"\n[{cycle_id}] processing {len(rows)} rows  "
            f"days [{rows['days_since_halving'].min():.0f}, "
            f"{rows['days_since_halving'].max():.0f}]"
        )

        prices = rows["price"].values.astype(float)
        days   = rows["days_since_halving"].values.astype(float)

        peak_day, peak_price = find_confirmed_peak(
            prices, days,
            drawdown_threshold=drawdown_threshold,
            confirmation_days=confirmation_days,
            cycle_id=cycle_id,
            fallback_cap_days=fallback_cap_days,
        )
        trough_day, trough_price = find_trough_after_peak(
            prices, days, peak_day, cycle_id
        )
        max_day = int(rows["days_since_halving"].max())

        trough_confirmed = True if is_complete else _is_trough_confirmed(
            prices, days, trough_day, trough_price,
            recovery_pct=trough_cfg["recovery_pct"],
            min_days=trough_cfg["confirmation_days"],
            cycle_id=cycle_id,
        )

        result[cycle_id] = {
            "peak_day":         peak_day,
            "peak_price":       peak_price,
            "trough_day":       trough_day,
            "trough_price":     trough_price,
            "max_day":          max_day,
            "is_complete":      is_complete,
            "trough_confirmed": trough_confirmed,
        }

        accum_label = (
            f"ACCUMULATION {trough_day}→{max_day} ({max_day - trough_day}d)"
            if trough_confirmed
            else "ACCUMULATION not yet confirmed"
        )
        logger.info(
            f"[{cycle_id}] BOUNDARY SUMMARY — "
            f"EXPANSION 0→{peak_day} ({peak_day}d) | "
            f"DISTRIBUTION {peak_day}→{trough_day} ({trough_day - peak_day}d) | "
            f"{accum_label}"
        )

    logger.info("\ncompute_cycle_boundaries DONE")
    return result
