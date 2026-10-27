import numpy as np

from lsfvg.signals.indicators import atr_prior, atr_wilder, efficiency_ratio, rolling_percentile, true_range


def test_atr_wilder_matches_hand_calculation_and_is_causal():
    h = np.array([10, 12, 11, 13, 12.5, 14.0])
    l = np.array([9, 10, 9.5, 11, 11.5, 12.0])
    c = np.array([9.5, 11, 10, 12, 12, 13.5])
    tr = true_range(h, l, c)
    assert np.allclose(tr, [1, 2.5, 1.5, 3.0, 1.0, 2.0])
    atr = atr_wilder(h, l, c, 3)
    assert np.isnan(atr[:2]).all() and abs(atr[2] - (1 + 2.5 + 1.5) / 3) < 1e-12
    assert abs(atr[3] - (atr[2] * 2 + 3.0) / 3) < 1e-12
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    h2[-1] += 100                                                      # changing the FUTURE bar must not change the past
    assert np.allclose(atr_wilder(h2, l2, c2, 3)[:5], atr[:5], equal_nan=True)
    assert np.isnan(atr_prior(atr)[0]) and atr_prior(atr)[3] == atr[2]


def test_efficiency_ratio_extremes_and_percentile_rank_is_trailing():
    trend = np.arange(30, dtype=float)
    chop = np.tile([1.0, 2.0], 15)
    assert abs(efficiency_ratio(trend, 10)[-1] - 1.0) < 1e-12 and efficiency_ratio(chop, 10)[-1] < 0.2
    x = np.r_[np.arange(300, dtype=float), 0.0]
    p = rolling_percentile(x, 100)
    assert p[299] == 1.0 and p[300] < 0.02                              # new high -> top rank; new low -> bottom rank
    y = x.copy()
    y[-1] = 1e9                                                         # future change leaves earlier ranks untouched
    assert np.allclose(rolling_percentile(y, 100)[:300], p[:300], equal_nan=True)
