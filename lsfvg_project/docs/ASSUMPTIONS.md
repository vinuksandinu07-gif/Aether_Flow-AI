# Phase 1 assumptions (please confirm or correct)

## Environment
1. Target Python 3.11. Phase 1 was verified on Python 3.12 with numpy 2.4 / pandas 3.0 / matplotlib 3.10.
2. `pytest` could not be installed where this was built (no network). Tests are plain-`assert` functions; run them with
   `pytest` or, without it, `python tests/run_tests.py`.
3. The OANDA candle client was tested against a fake HTTP session only (paging, retry, token hygiene).
   It has never talked to the real API. Check one small download on a practice account first.

## Data
4. Canonical frame: UTC index = bar OPEN time; mid `open/high/low/close/volume`, optional `bid_*`/`ask_*`.
5. Signals run on MID candles. Bid/ask (recorded, or mid +/- modeled half-spread) is used for fills, spread and costs.
   If a file has no bid/ask the loader/demo adds a MODELED spread and the quality report says so.
6. Quality checks are detect-and-report. Bars are never repaired silently; `drop_flagged()` is an explicit call.
   Error-level issues (duplicates, OHLC inconsistency, crossed quotes, NaN, bad timestamps) stop a run unless
   `--allow-data-errors`. Bad-tick spikes and missing bars are warnings.
7. Trading day = 17:00 -> 17:00 America/New_York (per-instrument `day_anchor_ny`); 4H bars open 17/21/01/05/09/13 NY;
   trading week = Sun 17:00 -> Fri 17:00. Bins are measured on the wall clock, which is exact because FX/CFD are closed
   when DST changes. (Not valid for 24/7 markets.)
8. Resampling is from the single base dataset; the trailing incomplete bin and the dataset's first partial bin are dropped.
9. SPX500_USD is an OANDA CFD (continuous): no contract rolls. `data/rolls.py` implements back-adjusted
   (difference/ratio) stitching for futures data; config must name the roll method if `is_futures: true`.

## Strategy rules (where the spec left room)
10. Direction: buy-side pool sweep -> SHORT setup; sell-side sweep -> LONG.
11. Pools: PDH/PDL and PWH/PWL (first, possibly partial, period dropped; current period does not exist yet), completed
    session highs/lows (`pool_sessions`, separate from entry windows), equal highs/lows on MTF (internal) and HTF (external).
    Pools expire by age (config). TP scope `any|external`.
12. "Swept" = first strict trade-through of the level on an MTF bar. A *sweep setup* additionally needs depth >=
    `min_sweep_atr` x ATR and a close back inside within `reclaim_bars` (breach bar counts as 0).
13. ATR thresholds use the ATR BEFORE the bar being tested (`atr_prior`); stop buffer and min/max stop use ATR at the signal bar.
14. MSS: last confirmed opposing swing formed before the sweep extreme; candle must be directional, close beyond it, and have
    body >= `disp_mult` x ATR, within `mss_max_bars` after the reclaim. A close back beyond the pool level first = failed sweep.
15. FVG: its MIDDLE candle must lie between the sweep extreme and the MSS bar; known at the next candle's close.
    Already-mitigated gaps (entry traded or closed through) are discarded. Largest gap wins (`fvg_select`).
16. Signal time = close of the bar that completes MSS + FVG. Entry/stop/TP are computed on mid levels; the order goes live
    on LTF bars from that close.
17. Bias: HTF dealing range = latest confirmed HTF swing high/low (closed HTF bars only). Zone is judged at the entry price.
    Optional HTF-structure trend alignment (HH+HL / LH+LL) as filter; alignment is also the first ranking key.
18. Session filter tests the signal bar's OPEN time against NY windows (half-open). The order may fill after the window ends.
19. Stops: structural stop +/- (spread + `stop_buffer_atr` x ATR); < min -> widen (or skip); > max -> skip. Rounded away from entry.
20. Effective RR = (reward - C) / (risk + C), C = one round trip of spread + entry slippage + exit slippage + commission.
    partial_runner gates on the blended planned RR; trailing gates on the RR to the nearest opposing pool (else `fixed_rr`).
21. `be_offset` is expressed in R (default 0.1R). Default `min_rr` 2 is applied AFTER costs, as specified.
22. Order rules: target taken before fill cancels (checked first, so a same-bar target+entry is NOT a fill); fills need
    `fill_through_ticks` through the limit; gaps fill at the open; "close fully through the FVG" cancels (with edge/midpoint
    entries a fill normally happens first, so this fires mainly when spread keeps quotes from reaching the limit).
23. Ranking happens among valid setups that complete on the same MTF bar (HTF aligned > nearest target > larger FVG).
    Cross-bar rules (one open position, pending-order replacement, opposite-signal rule, max 2 trades/day) are Phase 2.

## Not in Phase 1 (by design)
Trade management (partials, breakeven, trailing, time stop), position sizing, loss limits, backtester, metrics,
optimization, acceptance gates, broker execution. Config fields for them are already validated.

## Phase 2 event-driven backtester assumptions

- The simulator consumes the same `SignalEngine.scan()` candidates as Phase 1. It does not alter the sweep/MSS/FVG definitions.
- One instrument is simulated per `run_backtest()` call; each run permits one open position for that instrument. The configured maximum is therefore enforced by the single-position event loop.
- A new accepted setup replaces older pending orders. A filled order cancels all remaining pending orders for that instrument.
- Limit fills require the configured number of ticks through the limit on the executable quote side. If a bar gaps through the limit, the opening quote is used when better, then configured entry slippage is applied adversely.
- Target-pool take before fill cancels the pending order before the fill test. FVG close-through and expiry are checked after fill evaluation.
- Stop/target ambiguity is stop-first by default. Same-bar entry-and-stop is conservatively treated as a loss if the stop was touched in that bar; finer-grained candles can reduce this ambiguity.
- Existing bid/ask candles are used when present. Otherwise a configured spread is modelled around mid OHLC. Spread stress scales recorded quotes around mid or scales the modelled spread. Exit slippage is adverse on all exit types.
- P&L and margin assumptions are intended for the initially configured USD-quoted EUR_USD and SPX500_USD instruments. Cross-currency account conversion is not silently estimated; implement and verify it before adding non-USD-quoted instruments.
- Position size is floor-rounded to instrument unit precision using equity times risk percentage divided by stop distance plus stop slippage. It is capped by instrument max units and checked against min units and configured margin rate.
- Daily and weekly loss limits block/cancel new entries; the max-drawdown kill switch halts new entries and closes an open position at the current executable close. Daily/weekly baselines reset at the configured New York trading-day anchor.
- `partial_runner` scales out at the planned partial level and moves the remaining stop to breakeven plus the configured R offset. `trailing` activates at the configured threshold and trails using confirmed three-bar local extrema or the setup FVG boundary.
- The current Phase 1 ZIP contains no trained AI model. AI integration is an explicit callable interface; AI mode rejects setups if no policy is provided unless `backtest.allow_ai_without_model` is explicitly enabled. No model learning or predictive performance is claimed in Phase 2.
- Synthetic runs are software smoke tests only and are not strategy-performance evidence.
