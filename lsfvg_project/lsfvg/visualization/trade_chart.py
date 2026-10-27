"""Per-setup charts so you can check the code sees the same setups you would.

Top panel = MTF (structure: pools, sweep, MSS, FVG).  Bottom panel = LTF (order lifecycle:
limit, fill / expiry / cancel, stop, targets, and - once Phase 2 exists - partials and exit).
Skipped setups are drawn dashed with the skip reasons in the title.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..config.schema import VizCfg  # noqa: E402
from ..core.timeutil import to_ns  # noqa: E402
from ..core.types import OrderOutcome, Setup, TradeRecord  # noqa: E402
from ..signals.engine import Prepared, ScanResult, SignalEngine  # noqa: E402
from ..signals.entry import resolve_entry_order  # noqa: E402

UP, DOWN = "#1b9e77", "#d95f02"
BUY_POOL, SELL_POOL = "#c0392b", "#2471a3"


def _candles(ax, x: np.ndarray, o, h, l, c, width: float = 0.6) -> None:
    col = np.where(c >= o, UP, DOWN)
    ax.vlines(x, l, h, colors=col, linewidth=0.8, zorder=2)
    body = np.maximum(np.abs(c - o), (h - l) * 0.02 + 1e-12)
    ax.bar(x, body, bottom=np.minimum(o, c), width=width, color=col, zorder=3, linewidth=0)


def _ticks(ax, times: pd.DatetimeIndex, x: np.ndarray, tz: str, k: int = 8) -> None:
    if len(x) == 0:
        return
    pos = np.linspace(0, len(x) - 1, min(k, len(x))).astype(int)
    ax.set_xticks(x[pos])
    ax.set_xticklabels([times[i].tz_convert(tz).strftime("%m-%d\n%H:%M") for i in pos], fontsize=7)


def _hline(ax, y: float, x0: float, x1: float, color: str, ls: str, label: str, lw: float = 1.0, fs: int = 7) -> None:
    ax.hlines(y, x0, x1, colors=color, linestyles=ls, linewidth=lw, zorder=4)
    ax.text(x1, y, f" {label}", color=color, fontsize=fs, va="center", ha="left", zorder=5, clip_on=False)


def plot_setup(
    setup: Setup,
    prep: Prepared,
    path: str | Path,
    *,
    viz: Optional[VizCfg] = None,
    outcome: Optional[OrderOutcome] = None,
    trade: Optional[TradeRecord] = None,
) -> Path:
    viz = viz or VizCfg()
    stack = prep.stack
    mtf, ltf, tz = stack.mtf, stack.ltf, stack.tz
    s = setup.signal_idx
    lo = max(0, setup.sweep.breach_idx - viz.mtf_bars_before)
    hi = min(prep.n - 1, s + viz.mtf_bars_after)
    skipped = setup.status != "valid"
    ls_main = "--" if skipped else "-"
    short = setup.direction == "short"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 2]})
    # ------------------------------------------------------------------ MTF
    x = np.arange(lo, hi + 1)
    sl = slice(lo, hi + 1)
    _candles(ax1, x, prep.o[sl], prep.h[sl], prep.l[sl], prep.c[sl])
    ymin, ymax = prep.l[sl].min(), prep.h[sl].max()
    pad = (ymax - ymin) * 0.12
    ax1.set_ylim(ymin - pad, ymax + pad)
    right = hi + 0.5
    for p in prep.pools:
        if p.valid_idx > hi or (p.swept_idx is not None and p.swept_idx < lo) or p.expire_idx < lo:
            continue
        if not (ymin - pad <= p.level <= ymax + pad):
            continue
        x0 = max(p.valid_idx, lo)
        x1 = min(p.swept_idx if p.swept_idx is not None else hi, hi)
        mine = p.pool_id == setup.pool_id
        col = BUY_POOL if p.side == "buy" else SELL_POOL
        ax1.hlines(p.level, x0, x1, colors=col, linestyles=":" if not mine else "-",
                   linewidth=1.8 if mine else 0.9, alpha=1.0 if mine else 0.7, zorder=1)
        ax1.text(x0, p.level, f"{p.kind}", color=col, fontsize=6, va="bottom", zorder=5)
    sw = setup.sweep
    ax1.scatter([sw.extreme_idx], [sw.extreme_price], marker="v" if short else "^", s=70, color="black", zorder=6)
    ax1.annotate("sweep", (sw.extreme_idx, sw.extreme_price), xytext=(0, 9 if not short else -14),
                 textcoords="offset points", fontsize=8, ha="center")
    m = setup.mss
    ax1.hlines(m.level, max(m.swing_idx, lo), m.idx, colors="purple", linestyles="--", linewidth=1.0, zorder=4)
    ax1.scatter([m.idx], [prep.c[m.idx]], marker="D", s=40, color="purple", zorder=6)
    ax1.annotate("MSS", (m.idx, prep.c[m.idx]), xytext=(6, -12 if short else 8), textcoords="offset points",
                 fontsize=8, color="purple")
    f = setup.fvg
    ax1.add_patch(Rectangle((f.mid_idx - 0.5, f.bottom), hi - f.mid_idx + 1, f.size, color="#f1c40f", alpha=0.28,
                            zorder=0, linewidth=0))
    ax1.annotate("FVG", (f.mid_idx, f.top), xytext=(0, 3), textcoords="offset points", fontsize=7, color="#9a7d0a")
    ax1.axvline(s + 0.5, color="grey", linestyle=":", linewidth=0.8)
    for y, lab, col in ((setup.entry_price, "entry", "black"), (setup.stop_price, "stop", "#c0392b")):
        _hline(ax1, y, s + 0.5, right, col, ls_main, lab)
    tp = setup.targets
    if tp.reference_target is not None:
        _hline(ax1, tp.reference_target, s + 0.5, right, "#1e8449", ls_main,
               "target" if tp.final_target is not None else "ref target")
    if tp.partial_price is not None:
        _hline(ax1, tp.partial_price, s + 0.5, right, "#7d6608", ":", "partial")
    _ticks(ax1, mtf.index[lo:hi + 1], x, tz)
    ax1.set_xlim(lo - 1, hi + 6)
    ax1.set_ylabel("MTF")
    ax1.grid(alpha=0.15)

    # ------------------------------------------------------------------ LTF
    l_open = to_ns(ltf.index)
    t_lo = int(to_ns(mtf.index[max(0, s - 12)::1][:1])[0])
    place_ns = int(pd.Timestamp(setup.signal_time).value)
    j0 = int(np.searchsorted(l_open, t_lo))
    jp = int(np.searchsorted(l_open, place_ns))
    jend = jp + 30
    if outcome is not None and outcome.bar_idx is not None:
        jend = max(jend, outcome.bar_idx + 15)
    if trade is not None and trade.exits:
        last_exit = int(np.searchsorted(l_open, max(int(e.time.value) for e in trade.exits)))
        jend = max(jend, last_exit + 10)
    jend = min(len(ltf) - 1, jend)
    xl = np.arange(j0, jend + 1)
    lo_, hi_, lw_, cl_ = (ltf[k].to_numpy(float)[j0:jend + 1] for k in ("open", "high", "low", "close"))
    _candles(ax2, xl, lo_, hi_, lw_, cl_, width=0.7)
    y0, y1 = lw_.min(), hi_.max()
    for lvl in (setup.stop_price, setup.entry_price, tp.reference_target):
        if lvl is not None:
            y0, y1 = min(y0, lvl), max(y1, lvl)
    pd_ = (y1 - y0) * 0.08
    ax2.set_ylim(y0 - pd_, y1 + pd_)
    ax2.add_patch(Rectangle((jp - 0.5, f.bottom), jend - jp + 1.5, f.size, color="#f1c40f", alpha=0.28, linewidth=0))
    for y, lab, col in ((setup.entry_price, "entry", "black"), (setup.stop_price, "stop", "#c0392b")):
        _hline(ax2, y, jp - 0.5, jend + 0.5, col, ls_main, lab)
    if tp.reference_target is not None:
        _hline(ax2, tp.reference_target, jp - 0.5, jend + 0.5, "#1e8449", ls_main, "target")
    if tp.partial_price is not None:
        _hline(ax2, tp.partial_price, jp - 0.5, jend + 0.5, "#7d6608", ":", "partial")
    if outcome is not None and outcome.bar_idx is not None:
        mk = {"filled": ("o", "black"), "expired": ("x", "grey"), "cancelled_fvg_through": ("x", "red"),
              "cancelled_target_taken": ("x", "red")}[outcome.status]
        yy = outcome.fill_price if outcome.fill_price is not None else setup.entry_price
        ax2.scatter([outcome.bar_idx], [yy], marker=mk[0], s=60, color=mk[1], zorder=7)
        ax2.annotate(outcome.status, (outcome.bar_idx, yy), xytext=(5, 8), textcoords="offset points", fontsize=7)
    if trade is not None:
        jx = int(np.searchsorted(l_open, trade.entry_time.value))
        ax2.scatter([jx], [trade.entry_price], marker=">", s=70, color="black", zorder=8)
        for e in trade.exits:
            je = int(np.searchsorted(l_open, e.time.value))
            ax2.scatter([je], [e.price], marker="s", s=55, color="#1e8449" if e.reason in ("TP", "partial", "take_profit", "take_profit_same_bar", "take_profit_ambiguous_bar", "partial_take_profit") else "#c0392b",
                        zorder=8)
            ax2.annotate(f"{e.reason} {e.fraction:.0%}", (je, e.price), xytext=(4, 6), textcoords="offset points", fontsize=7)
    _ticks(ax2, ltf.index[j0:jend + 1], xl, tz)
    ax2.set_xlim(j0 - 1, jend + 8)
    ax2.set_ylabel("LTF")
    ax2.grid(alpha=0.15)

    rr = f"RR {tp.planned_rr:.2f} (net {tp.effective_rr:.2f})"
    head = f"{setup.setup_id}  {setup.direction.upper()}  {rr}  zone={setup.bias_zone}  mode={tp.mode}"
    if skipped:
        head = "SKIPPED [" + ", ".join(setup.reasons) + "]  " + head
        for a in (ax1, ax2):
            for sp in a.spines.values():
                sp.set_edgecolor("#c0392b")
                sp.set_linewidth(1.6)
    fig.suptitle(head, fontsize=10, color="#c0392b" if skipped else "black")
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=viz.dpi)
    plt.close(fig)
    return path


def export_all(
    res: ScanResult,
    prep: Prepared,
    engine: SignalEngine,
    out_dir: str | Path,
    *,
    viz: Optional[VizCfg] = None,
    trades: Optional[dict[str, TradeRecord]] = None,
) -> list[Path]:
    """Write one PNG per setup (valid first, then skipped) to ``out_dir`` (e.g. reports/trades/)."""
    viz = viz or VizCfg()
    out = Path(out_dir)
    paths: list[Path] = []
    ordered = list(res.setups) + (list(res.skipped) if viz.include_skipped else [])
    for k, st in enumerate(ordered[: viz.max_charts]):
        outcome = resolve_entry_order(engine.entry_order(st), prep.stack.ltf) if st.status == "valid" else None
        tr = (trades or {}).get(st.setup_id)
        paths.append(plot_setup(st, prep, out / f"{k:03d}_{st.setup_id}_{st.status}.png", viz=viz,
                                outcome=outcome, trade=tr))
    return paths
