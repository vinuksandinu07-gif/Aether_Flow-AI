# Liquidity Sweep + FVG system - Phase 1 + Phase 2

Top-down strategy: **4H** bias/target liquidity -> **15m** sweep, displacement/MSS, FVG -> **5m** limit-order entry.
One signal engine is used for backtest, paper and live (`lsfvg/signals/engine.py`).

## Run it
```bash
pip install -r requirements.txt
python tests/run_tests.py                      # or: pytest
python scripts/phase1_demo.py --days 240 --out reports/demo            # synthetic plumbing check, default (strict) config
python scripts/phase1_demo.py --days 240 --relaxed --out reports/demo_relaxed
python scripts/phase1_demo.py --csv data/EURUSD_M5.csv --instrument EUR_USD --out reports/eurusd
python scripts/phase2_backtest.py --csv data/EURUSD_M5.csv --instrument EUR_USD --out reports/phase2_eurusd
python scripts/phase2_backtest.py --synthetic-days 120 --compare-ai --out reports/phase2_compare
```
Outputs per run: `data_quality/*.md|json`, `decision_log.jsonl` (why every setup was taken/skipped), `setups.csv`,
`config_snapshot.yaml`, `manifest.json` (config hash, data hash, seed), `trades/*.png` (one chart per setup).

## Phase 2 backtester and risk manager

`python scripts/phase2_backtest.py --csv ...` runs the same Phase 1 `SignalEngine` through a chronological LTF event loop.
It simulates limit orders, bid/ask-aware fills (or modeled spread), entry/exit slippage, commission, stop/target handling,
partial-runner exits, breakeven offsets, trailing stops, time stops, optional session/weekend exits, position sizing,
margin checks, daily/weekly loss limits, consecutive-loss cooldown, and the maximum-drawdown kill switch. Live execution is
never invoked by the backtest script.

Default risk settings are 0.5% risk per trade, 2% daily loss limit, 4% weekly loss limit, 10% maximum drawdown, at most two
trades per NY trading day, and a 12-LTF-bar cooldown after three consecutive losses. All can be changed in `config/default.yaml`.
The simulator uses stop-first handling when a stop and target are both touched in the same bar by default. P&L sizing is
appropriate for the initial USD-quoted `EUR_USD` and `SPX500_USD` instruments; other account-currency conversions must be
implemented/verified before using those instruments in sizing.

The Phase 1 package does not include a trained AI policy. `--ai-mode` therefore rejects candidates by default; it does not
pretend to have learned market behaviour. `--compare-ai` writes a baseline-versus-AI-path report so the absence of a model
is explicit. A validated callable policy can be supplied to `run_backtest(..., ai_policy=...)` after the Phase 3 training layer
exists. Synthetic runs validate plumbing only, not profitability.

Phase 2 outputs include `trades.csv`, `decisions.csv`, `equity_curve.csv`, `metrics.json`, equity/drawdown charts, data-quality
reports, config snapshots, and setup/trade charts.

## Layout
```
config/default.yaml        all parameters (validated at startup; unknown keys are errors)
lsfvg/config/              schema + loader
lsfvg/core/                timeutil (DST-safe), shared dataclasses
lsfvg/data/                loader, oanda client, quality report, resample, spread model, news, rolls, synthetic
lsfvg/signals/             indicators, swings, pools, sweep, structure (MSS), fvg, bias, stops, targets,
                           filters, entry (order lifecycle), engine
lsfvg/visualization/       trade_chart.py
tests/                     97 tests incl. lookahead/repaint and Phase 2 risk/backtest suite
lsfvg/risk/ and lsfvg/backtest/  Phase 2 risk controls, event-driven simulator, metrics and reports
lsfvg/execution/ and lsfvg/ops/  placeholders for Phase 4 broker execution / monitoring
docs/ASSUMPTIONS.md        every interpretation I made - please review
```

## How to review visually
Open `reports/<run>/trades/*.png`. Top panel = 15m (pools dotted, sweep marker, purple MSS line, yellow FVG, entry/stop/target).
Bottom = 5m order lifecycle (fill / expired / cancelled). Skipped setups are dashed with a red frame and the reasons in the title.
