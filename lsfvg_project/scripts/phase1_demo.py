#!/usr/bin/env python
"""Phase 1 end-to-end run: data -> quality report -> resample -> signals -> decision log -> charts.

Offline demo (synthetic EUR/USD, only for checking the plumbing - NOT evidence about the strategy):
    python scripts/phase1_demo.py --days 240 --out reports/demo

Your own data (UTC or set data.csv.tz in the config):
    python scripts/phase1_demo.py --csv data/EURUSD_M5.csv --instrument EUR_USD --out reports/eurusd

OANDA practice candles (needs OANDA_API_TOKEN in the environment):
    python scripts/phase1_demo.py --oanda --instrument EUR_USD --start 2025-06-01 --end 2025-12-01
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lsfvg.config import config_hash, load_config, save_snapshot            # noqa: E402
from lsfvg.data.loader import data_version_hash, load_csv                   # noqa: E402
from lsfvg.data.news import load_news_csv                                   # noqa: E402
from lsfvg.data.quality import DataQualityError, enforce, run_quality_report  # noqa: E402
from lsfvg.data.resample import build_stack                                 # noqa: E402
from lsfvg.data.spread import SpreadModel, add_modeled_bid_ask              # noqa: E402
from lsfvg.data.synthetic import random_walk_bars                           # noqa: E402
from lsfvg.signals.engine import SignalEngine, write_decision_log           # noqa: E402
from lsfvg.signals.entry import resolve_entry_order                         # noqa: E402
from lsfvg.signals.targets import breakeven_win_rate                        # noqa: E402
from lsfvg.visualization.trade_chart import export_all                      # noqa: E402

RELAXED = {  # loosens gates so a short synthetic sample produces enough setups to look at
    "bias": {"enabled": False}, "filters": {"volatility": {"enabled": False}},
    "stops": {"max_stop_atr": 4.0}, "targets": {"min_rr": 1.0}, "signals": {"disp_mult": 0.8},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--instrument", default="EUR_USD")
    ap.add_argument("--csv", help="base-timeframe CSV (mid or bid/ask OHLC)")
    ap.add_argument("--oanda", action="store_true", help="download candles from the OANDA practice API")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--days", type=int, default=240, help="synthetic days (no --csv/--oanda)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase1_run"))
    ap.add_argument("--max-charts", type=int, default=None)
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--relaxed", action="store_true", help="apply the RELAXED gate overrides (demo only)")
    ap.add_argument("--allow-data-errors", action="store_true", help="continue despite data-quality ERRORS (reported)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ov = dict(RELAXED) if a.relaxed else {}
    if a.max_charts is not None:
        ov = {**ov, "viz": {"max_charts": a.max_charts}}
    cfg = load_config(a.config, ov or None)
    seed = cfg.run.seed if a.seed is None else a.seed
    name, ins = a.instrument, cfg.instruments[a.instrument]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data
    notes = {}
    if a.csv:
        ld = load_csv(a.csv, cfg.data.csv)
        df, notes = ld.df, ld.notes
    elif a.oanda:
        from lsfvg.data.oanda import OandaCandleClient
        df = OandaCandleClient(practice=True).fetch(ins.symbol, cfg.timeframes.base,
                                                    pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC"))
    else:
        if name != "EUR_USD":
            ap.error("synthetic data exists for EUR_USD only; pass --csv for other instruments")
        df = random_walk_bars(n_days=a.days, seed=seed)
        print(f"[synthetic data: seed={seed}, {a.days} days - plumbing check only]")
    if "bid_close" not in df.columns:
        df = add_modeled_bid_ask(df, SpreadModel(ins, cfg.run.tz_sessions))
        print("[no recorded bid/ask: using mid +/- MODELED spread]")

    rep = run_quality_report(df, instrument_name=name, ins=ins, base_tf=cfg.timeframes.base,
                             tz=cfg.run.tz_sessions, qcfg=cfg.data.quality, load_notes=notes)
    pj, pm = rep.save(out / "data_quality")
    print(f"data quality: {'OK' if rep.ok else 'ERRORS'} -> {pm}")
    try:
        enforce(rep, cfg.data.quality)
    except DataQualityError as e:
        if not a.allow_data_errors:
            print(f"STOP: {e}  (see {pm}; fix the data or pass --allow-data-errors)")
            return 2

    # ---------------- signals
    news = load_news_csv(cfg.filters.news.csv_path) if cfg.filters.news.enabled else None
    stack = build_stack(df, cfg.timeframes, ins, cfg.run.tz_sessions)
    eng = SignalEngine(cfg, name, news)
    prep = eng.prepare(stack)
    res = eng.scan(prep)
    write_decision_log(res, out / "decision_log.jsonl")
    rows = [s.to_dict() for s in res.all_setups]
    pd.DataFrame(rows).to_csv(out / "setups.csv", index=False)
    save_snapshot(cfg, out)
    (out / "manifest.json").write_text(json.dumps({
        "instrument": name, "seed": seed, "config_hash": config_hash(cfg), "data_hash": data_version_hash(df),
        "bars_base": len(df), "bars_mtf": len(stack.mtf), "bars_htf": len(stack.htf), "pools": len(prep.pools),
        "python": platform.python_version(), "pandas": pd.__version__,
        "spread_modeled": bool(df.attrs.get("spread_modeled", False)),
    }, indent=2))

    # ---------------- summary
    outcomes = collections.Counter(resolve_entry_order(eng.entry_order(s), stack.ltf).status for s in res.setups)
    print(f"\nbars: base={len(df):,} ltf={len(stack.ltf):,} mtf={len(stack.mtf):,} htf={len(stack.htf):,}   pools={len(prep.pools)}")
    print(f"setups: {len(res.setups)} valid, {len(res.skipped)} skipped, {len(res.rejected_sweeps)} sweeps that never became a setup")
    print("skip reasons:", dict(res.skip_counts()))
    print("sweeps that stopped early:", dict(collections.Counter(r['reason'] for r in res.rejected_sweeps)))
    print("limit-order outcomes (valid setups):", dict(outcomes))
    if res.setups:
        avg = sum(s.targets.effective_rr for s in res.setups) / len(res.setups)
        print(f"planned effective RR (mean over valid setups): {avg:.2f}  -> break-even win rate if realised: {breakeven_win_rate(avg):.1%}")
    if not a.no_charts:
        paths = export_all(res, prep, eng, out / "trades", viz=cfg.viz)
        print(f"{len(paths)} charts -> {out / 'trades'}")
    print(f"decision log -> {out / 'decision_log.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
