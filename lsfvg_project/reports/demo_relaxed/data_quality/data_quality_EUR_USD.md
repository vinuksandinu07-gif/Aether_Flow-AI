# Data quality - EUR_USD (5m)

* bars: **49,548**  (2026-01-04 22:00:00+00:00 -> 2026-09-01 21:55:00+00:00)
* prices: mid + MODELED spread
* verdict: **OK**

| severity | kind | count | detail |
|---|---|---:|---|
| info | dst_transitions | 1 | America/New_York DST changes inside the data (sessions are DST-aware) e.g. 2026-03-08: UTC-5->-4 (1730 bars within +/-3d) |
| info | weekend_gaps | 34 | expected weekly closures |

## Stats
```json
{
  "spread": {
    "median": 0.00010000000000021103,
    "p95": 0.00010000000000021103,
    "max": 0.00010000000000021103
  },
  "gaps": {
    "weekend": 34,
    "daily_break": 0,
    "long_gap_weekday": 0,
    "missing_bars": 0
  },
  "missing_bars_total": 0,
  "dst_transitions": [
    "2026-03-08: UTC-5->-4 (1730 bars within +/-3d)"
  ],
  "spread_modeled": true
}
```