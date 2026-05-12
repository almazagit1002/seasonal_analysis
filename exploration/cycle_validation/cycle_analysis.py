"""
Cycle validation analysis
    1. Visual overlay — cumulative return by cycle_month for each halving cycle
    2. Phase table    — per (cycle x phase): duration, total return, wealth multiplier
                        Phase boundaries are derived from actual price data (ATH + cycle trough),
                        not from fixed day thresholds.

Run from the repo root:
    python exploration/cycle_validation/cycle_analysis.py

Outputs written to exploration/cycle_validation/output/
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.s3_utils import read_s3_file
from utils.logger import get_logger
from analysis.phase_detector import assign_cycle_id, compute_cycle_boundaries

logger = get_logger(__name__, log_to_file=True)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_HALVING_LABELS = ["2012", "2016", "2020", "2024"]

_COLORS = {
    "2012": "#e74c3c",
    "2016": "#2ecc71",
    "2020": "#3498db",
    "2024": "#f39c12",
}

_PHASE_ORDER = ["EXPANSION", "DISTRIBUTION", "ACCUMULATION"]


# assign_cycle_id and compute_cycle_boundaries are imported from
# analysis.phase_detector — see src/analysis/phase_detector.py


def _empirical_phase(days: float, b: dict) -> str:
    if days <= b["peak_day"]:
        return "EXPANSION"
    if days <= b["trough_day"]:
        return "DISTRIBUTION"
    return "ACCUMULATION"


def _phase_status(cycle_id: str, phase: str, b: dict) -> str:
    if b["is_complete"]:
        return "complete"
    cur = b["max_day"]
    if phase == "EXPANSION":
        return "complete" if cur > b["peak_day"] else "ongoing"
    if phase == "DISTRIBUTION":
        # DISTRIBUTION is only "complete" when the trough is confirmed
        if b.get("trough_confirmed", True) and cur > b["trough_day"]:
            return "complete"
        return "ongoing"
    # ACCUMULATION — only reachable if trough_confirmed is True (see phase assignment)
    return "ongoing"


# ─── shared stat helper ───────────────────────────────────────────────────────

def _wealth_multiplier(series: pd.Series) -> float:
    r = series.dropna().values
    return float(np.prod(1 + r / 100))


# ─── analysis 1 ───────────────────────────────────────────────────────────────

def analysis_1_cycle_overlay(df: pd.DataFrame, boundaries: dict) -> None:
    """
    Cumulative return by cycle_month, one line per cycle.
    Vertical markers show each cycle's empirical ATH (solid) and trough (dashed).
    """
    valid = df.dropna(subset=["cycle_month", "cycle_id", "daily_return_pct"]).copy()
    valid["cycle_month"] = valid["cycle_month"].astype(int)

    monthly_avg = (
        valid.groupby(["cycle_id", "cycle_month"])["daily_return_pct"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    for cycle_id, group in monthly_avg.groupby("cycle_id"):
        group  = group.sort_values("cycle_month")
        cumret = group["daily_return_pct"].cumsum().values
        is_ongoing = cycle_id == "2024"
        ax.plot(
            group["cycle_month"].values,
            cumret,
            label=f"Cycle {cycle_id}" + (" (ongoing)" if is_ongoing else ""),
            color=_COLORS.get(cycle_id, "gray"),
            linewidth=2.5,
            linestyle="--" if is_ongoing else "-",
            alpha=0.95,
        )

        # Empirical peak and trough markers
        if cycle_id in boundaries:
            b = boundaries[cycle_id]
            color = _COLORS.get(cycle_id, "gray")
            peak_cm   = min(int(b["peak_day"]   // 30) + 1, 48)
            trough_cm = min(int(b["trough_day"] // 30) + 1, 48)
            ax.axvline(peak_cm,   color=color, linewidth=1.2, linestyle=":",  alpha=0.7)
            ax.axvline(trough_cm, color=color, linewidth=1.2, linestyle="-.", alpha=0.7)

    # Legend entries for markers
    ax.plot([], [], color="gray", linewidth=1.2, linestyle=":",  label="cycle ATH (by cycle)")
    ax.plot([], [], color="gray", linewidth=1.2, linestyle="-.", label="cycle trough (by cycle)")

    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(1, 48)
    ax.xaxis.set_major_locator(plt.MultipleLocator(6))
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Cycle Month  (month 1 = first 30 days after halving)", fontsize=12)
    ax.set_ylabel("Cumulative avg daily return (%)", fontsize=12)
    ax.set_title(
        "BTC Cumulative Return by Cycle Month — All Halving Cycles Overlaid\n"
        "dotted = cycle ATH  |  dash-dot = cycle trough  (empirical, per cycle)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "cycle_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─── analysis 2 ───────────────────────────────────────────────────────────────

def analysis_2_phase_table(df: pd.DataFrame, boundaries: dict) -> None:
    """
    Per (cycle x phase): duration, avg daily return, total compound return,
    wealth multiplier, and status.  Phase boundaries from compute_cycle_boundaries.
    """
    valid = df[
        df["cycle_id"].notna() &
        df["days_since_halving"].notna() &
        df["daily_return_pct"].notna()
    ].copy()

    # Assign empirical phase to every row
    def get_phase(row):
        cid = row["cycle_id"]
        return _empirical_phase(row["days_since_halving"], boundaries[cid]) if cid in boundaries else None

    valid["emp_phase"] = valid.apply(get_phase, axis=1)
    valid = valid.dropna(subset=["emp_phase"])

    # For the ongoing cycle, if the trough is not yet confirmed, treat all
    # post-peak data as DISTRIBUTION — do not assume ACCUMULATION has started.
    b24 = boundaries.get("2024", {})
    if not b24.get("trough_confirmed", True):
        mask = (valid["cycle_id"] == "2024") & (valid["emp_phase"] == "ACCUMULATION")
        if mask.any():
            logger.info(
                f"[2024] trough not confirmed — "
                f"merging {mask.sum()} ACCUMULATION rows into DISTRIBUTION"
            )
            valid.loc[mask, "emp_phase"] = "DISTRIBUTION"

    # ── boundary summary ──────────────────────────────────────────────────────
    W = 90
    print("\n" + "=" * W)
    print("EMPIRICAL CYCLE BOUNDARIES  (ATH = end of EXPANSION, trough = end of DISTRIBUTION)")
    print("=" * W)
    for cid in _HALVING_LABELS:
        if cid not in boundaries:
            continue
        b = boundaries[cid]
        if not b["is_complete"]:
            trough_tag = (
                "  ← provisional low (+29% recovery, 90d sustained needed — NOT YET CONFIRMED)"
                if not b.get("trough_confirmed", True)
                else "  ← confirmed"
            )
        else:
            trough_tag = ""
        print(
            f"  Cycle {cid}:  ATH day {b['peak_day']:>4}  (${b['peak_price']:>10,.0f})  "
            f"| trough day {b['trough_day']:>4}  (${b['trough_price']:>10,.0f}){trough_tag}"
        )

    # ── compute stats ─────────────────────────────────────────────────────────
    grp    = valid.groupby(["cycle_id", "emp_phase"])["daily_return_pct"]
    wealth = grp.apply(_wealth_multiplier)
    stats  = pd.DataFrame({
        "duration_days":  grp.count(),
        "avg_daily_%":    grp.mean().round(4),
        "total_return_%": ((wealth - 1) * 100).round(1),
        "wealth_mult":    wealth.round(2),
    }).reset_index().rename(columns={"emp_phase": "cycle_phase"})

    stats["status"] = stats.apply(
        lambda r: _phase_status(r["cycle_id"], r["cycle_phase"], boundaries.get(r["cycle_id"], {})),
        axis=1,
    )

    stats["cycle_id"]    = pd.Categorical(stats["cycle_id"],    categories=_HALVING_LABELS, ordered=True)
    stats["cycle_phase"] = pd.Categorical(stats["cycle_phase"], categories=_PHASE_ORDER,    ordered=True)
    stats = stats.sort_values(["cycle_id", "cycle_phase"]).reset_index(drop=True)

    # ── printed table ─────────────────────────────────────────────────────────
    _ICON = {"complete": "✓", "ongoing": "▶"}
    b24   = boundaries.get("2024", {})
    cur   = b24.get("max_day", "?")
    print("\n" + "=" * W)
    print(f"PHASE BREAKDOWN PER CYCLE  (2024 currently at day {cur})")
    print("=" * W)
    print(f"{'Cycle':<8} {'Phase':<14} {'Days':>6}  {'Avg daily%':>11}  {'Total ret%':>11}  {'Wealth mult':>12}  Status")
    print("-" * W)
    prev = None
    for _, row in stats.iterrows():
        if row["cycle_id"] != prev and prev is not None:
            print()
        prev = row["cycle_id"]
        icon = _ICON.get(row["status"], "")
        note = "  (partial — trough not confirmed)" if row["status"] == "ongoing" and row["cycle_phase"] == "DISTRIBUTION" else ""
        print(
            f"{str(row['cycle_id']):<8} {row['cycle_phase']:<14} {row['duration_days']:>6}  "
            f"{row['avg_daily_%']:>+11.4f}  {row['total_return_%']:>+11.1f}%  "
            f"{row['wealth_mult']:>11.2f}x   {icon} {row['status']}{note}"
        )
    print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "phase_table.csv")
    stats.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ── heatmap ───────────────────────────────────────────────────────────────
    pivot_ret  = stats.pivot(index="cycle_id", columns="cycle_phase", values="total_return_%").reindex(columns=_PHASE_ORDER)
    pivot_mult = stats.pivot(index="cycle_id", columns="cycle_phase", values="wealth_mult").reindex(columns=_PHASE_ORDER)
    pivot_dur  = stats.pivot(index="cycle_id", columns="cycle_phase", values="duration_days").reindex(columns=_PHASE_ORDER)
    pivot_stat = stats.pivot(index="cycle_id", columns="cycle_phase", values="status").reindex(columns=_PHASE_ORDER)

    vals = pivot_ret.values.astype(float)
    vmax = np.nanmax(np.abs(vals))

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(_PHASE_ORDER)))
    ax.set_xticklabels(_PHASE_ORDER, fontsize=12)
    ax.set_yticks(range(len(pivot_ret.index)))
    ax.set_yticklabels([f"Cycle {c}" for c in pivot_ret.index], fontsize=11)

    for i, cid in enumerate(pivot_ret.index):
        for j, phase in enumerate(_PHASE_ORDER):
            tot  = pivot_ret.loc[cid, phase]
            mult = pivot_mult.loc[cid, phase]
            dur  = pivot_dur.loc[cid, phase]
            st   = pivot_stat.loc[cid, phase] if cid in pivot_stat.index else "complete"

            if pd.isna(tot):
                # 2024 ACCUMULATION — trough not yet confirmed
                label = "not yet\nconfirmed" if cid == "2024" else "no data"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=9, color="gray", style="italic")
                continue

            text_color  = "white" if abs(tot) > vmax * 0.55 else "black"
            dur_label   = f"{int(dur)}d ▶" if st == "ongoing" else f"{int(dur)}d"
            annotation  = f"{tot:+.1f}%\n{mult:.2f}x\n{dur_label}"
            ax.text(j, i, annotation, ha="center", va="center",
                    fontsize=9.5, color=text_color, linespacing=1.5)

    cbar = plt.colorbar(im, ax=ax, label="Total compound return (%)", pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(
        "BTC Phase Performance per Halving Cycle  (empirical boundaries)\n"
        "cell: total compound return  ·  wealth multiplier  ·  days",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, "phase_heatmap.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading enriched data from S3...")
    df = read_s3_file(bucket="btc-seasonal-analysis-faadab22", key="btc_enriched.parquet")
    df = assign_cycle_id(df)
    print(f"Loaded {len(df)} rows | Cycles: {sorted(df['cycle_id'].dropna().unique())}")

    boundaries = compute_cycle_boundaries(df)

    analysis_1_cycle_overlay(df, boundaries)
    analysis_2_phase_table(df, boundaries)

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
