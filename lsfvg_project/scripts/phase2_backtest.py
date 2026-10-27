#!/usr/bin/env python
"""Phase 2 event-driven backtest: CSV/synthetic data -> Phase 1 signals -> risk-aware simulation.

Examples:
  python scripts/phase2_backtest.py --csv data/EURUSD_M5.csv --instrument EUR_USD --out reports/phase2_eurusd
  python scripts/phase2_backtest.py --synthetic-days 120 --out reports/phase2_synthetic

Synthetic results validate plumbing only; they are not evidence of trading profitability.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lsfvg.backtest.engine import run_backtest  # noqa: E402
from lsfvg.config import config_hash, load_config, save_snapshot  # noqa: E402
from lsfvg.data.loader import data_version_hash, load_csv  # noqa: E402
from lsfvg.data.news import load_news_csv  # noqa: E402
from lsfvg.data.quality import DataQualityError, enforce, run_quality_report  # noqa: E402
from lsfvg.data.resample import build_stack  # noqa: E402
from lsfvg.data.spread import SpreadModel, add_modeled_bid_ask  # noqa: E402
from lsfvg.data.synthetic import random_walk_bars  # noqa: E402
from lsfvg.signals.engine import SignalEngine  # noqa: E402
from lsfvg.visualization.trade_chart import export_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--instrument", default="EUR_USD")
    ap.add_argument("--csv", help="base-timeframe CSV (mid or bid/ask OHLC)")
    ap.add_argument("--synthetic-days", type=int, default=0, help="offline plumbing test only")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase2_run"))
    ap.add_argument("--allow-data-errors", action="store_true")
    ap.add_argument("--ai-mode", action="store_true", help="run AI policy path; without a model it fails closed by default")
    ap.add_argument("--compare-ai", action="store_true", help="run deterministic baseline and fail-closed AI path side by side")
    ap.add_argument("--spread-stress", type=float, default=None, help="multiply modeled spread, e.g. 2.0")
    ap.add_argument("--slippage-stress", type=float, default=None, help="multiply entry/exit slippage, e.g. 2.0")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.csv and a.synthetic_days <= 0:
        ap.error("provide --csv or --synthetic-days (synthetic is plumbing-only)")
    if a.spread_stress is not None and a.spread_stress <= 0:
        ap.error("--spread-stress must be > 0")
    if a.slippage_stress is not None and a.slippage_stress <= 0:
        ap.error("--slippage-stress must be > 0")

    overrides = {"backtest": {}}
    if a.spread_stress is not None:
        overrides["backtest"]["default_spread_stress"] = a.spread_stress
    if a.slippage_stress is not None:
        overrides["backtest"]["slippage_stress"] = a.slippage_stress
    cfg = load_config(a.config, overrides if overrides["backtest"] else None)
    if a.instrument not in cfg.instruments:
        ap.error(f"instrument {a.instrument!r} is not defined in config")
    name, ins = a.instrument, cfg.instruments[a.instrument]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    notes = {}
    if a.csv:
        loaded = load_csv(a.csv, cfg.data.csv)
        df, notes = loaded.df, loaded.notes
    else:
        if name != "EUR_USD":
            ap.error("synthetic data is only configured for EUR_USD; pass --csv for other instruments")
        seed = cfg.run.seed if a.seed is None else a.seed
        df = random_walk_bars(n_days=a.synthetic_days, seed=seed)
        print(f"[SYNTHETIC DATA seed={seed}; plumbing test only, not strategy evidence]")
    if "bid_close" not in df.columns:
        df = add_modeled_bid_ask(df, SpreadModel(ins, cfg.run.tz_sessions))
        print("[No recorded bid/ask: modeled quotes added from configured spread]")

    report = run_quality_report(df, instrument_name=name, ins=ins, base_tf=cfg.timeframes.base,
                                tz=cfg.run.tz_sessions, qcfg=cfg.data.quality, load_notes=notes)
    q_json, q_md = report.save(out / "data_quality")
    print(f"Data quality: {'OK' if report.ok else 'ERRORS'} -> {q_md}")
    try:
        enforce(report, cfg.data.quality)
    except DataQualityError as exc:
        if not a.allow_data_errors:
            print(f"STOP: {exc}. Fix data or explicitly pass --allow-data-errors.")
            return 2
        print("WARNING: continuing despite data errors; results are not trusted.")

    news = load_news_csv(cfg.filters.news.csv_path) if cfg.filters.news.enabled and cfg.filters.news.csv_path else None
    stack = build_stack(df, cfg.timeframes, ins, cfg.run.tz_sessions)
    signal_engine = SignalEngine(cfg, name, news)
    prepared = signal_engine.prepare(stack)
    comparison = None
    if a.compare_ai:
        baseline = run_backtest(signal_engine, prepared, cfg, ai_mode=False)
        baseline_paths = baseline.save(out / "baseline")
        ai_result = run_backtest(signal_engine, prepared, cfg, ai_mode=True)
        ai_paths = ai_result.save(out / "ai_mode")
        comparison = pd.DataFrame([
            {"mode": "deterministic_baseline", **baseline.metrics},
            {"mode": "ai_mode_no_policy", **ai_result.metrics},
        ])
        comparison.to_csv(out / "baseline_vs_ai.csv", index=False)
        result = baseline
        print("AI comparison: no trained policy is present in this Phase 1 package; AI path fails closed.")
    else:
        result = run_backtest(signal_engine, prepared, cfg, ai_mode=a.ai_mode)
    paths = result.save(out)
    # Add actual backtest entry/exit markers to the existing Phase 1 setup visualizations.
    scan_for_charts = signal_engine.scan(prepared)
    trade_map = {tr.setup_id: tr for tr in result.trades}
    chart_paths = export_all(scan_for_charts, prepared, signal_engine, out / "trades", viz=cfg.viz, trades=trade_map)
    paths["trade_charts"] = f"{len(chart_paths)} charts in {out / 'trades'}"
    if comparison is not None:
        paths["baseline_vs_ai.csv"] = str(out / "baseline_vs_ai.csv")
        paths["baseline_reports"] = str(out / "baseline")
        paths["ai_reports"] = str(out / "ai_mode")
    save_snapshot(cfg, out)
    manifest = {
        "instrument": name, "config_hash": config_hash(cfg), "data_hash": data_version_hash(df),
        "bars_base": len(df), "bars_ltf": len(stack.ltf), "bars_mtf": len(stack.mtf), "bars_htf": len(stack.htf),
        "pools": len(prepared.pools), "spread_modeled": bool(df.attrs.get("spread_modeled", False)),
        "python": sys.version, "quality_json": str(q_json), "quality_markdown": str(q_md),
        "outputs": paths, "metrics": result.metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("\n=== PHASE 2 BACKTEST SUMMARY ===")
    for key in ("trade_count", "win_rate_pct", "net_pnl", "total_return_pct", "profit_factor",
                "expectancy_r", "max_drawdown_pct", "setups_valid", "setups_skipped", "sweeps_rejected_before_setup"):
        print(f"{key}: {result.metrics.get(key)}")
    print(f"Halted: {result.metrics.get('halted')} ({result.metrics.get('halt_reason') or 'no kill-switch trigger'})")
    if (a.ai_mode or a.compare_ai) and result.metrics.get("ai_policy_available") is False:
        print("AI MODE: no trained policy was supplied; the AI path rejects candidates by default (fail-closed).")
    print(f"Reports -> {out.resolve()}")
    print("Live execution was not invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
