import numpy as np
import pandas as pd
import pytest

from scorecard import indicators as ind


def frame(values):
    return pd.DataFrame({"A": values}, index=pd.bdate_range("2024-01-01", periods=len(values)),
                        dtype=float)


def test_sma_matches_manual_average():
    out = ind.SMA(3)(frame([1, 2, 3, 4, 5]))["A"]
    assert out.isna().sum() == 2
    assert out.iloc[-1] == pytest.approx(4.0)


def test_rsi_is_100_without_losses_and_matches_wilder_loop():
    assert ind.RSI(14)(frame(range(1, 40)))["A"].iloc[-1] == pytest.approx(100.0)

    prices = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 80))
    got = ind.RSI(14)(frame(prices))["A"].iloc[-1]
    # Reference implementation: Wilder's recursive smoothing written as a loop.
    change = np.diff(prices)
    gain, loss = np.clip(change, 0, None), np.clip(-change, 0, None)
    avg_g, avg_l = gain[0], loss[0]
    for g, l in zip(gain[1:], loss[1:]):
        avg_g = avg_g + (g - avg_g) / 14
        avg_l = avg_l + (l - avg_l) / 14
    assert got == pytest.approx(100 - 100 / (1 + avg_g / avg_l))


def test_max_drawdown_finds_worst_fall():
    out = ind.MaxDrawdown(4)(frame([100, 120, 90, 110]))["A"]
    assert out.iloc[-1] == pytest.approx(90 / 120 - 1)


def test_beta_of_leveraged_stock_is_two():
    rng = np.random.default_rng(3)
    market_ret = rng.normal(0, 0.01, 300)
    idx = pd.bdate_range("2024-01-01", periods=300)
    index_close = pd.Series(100 * np.cumprod(1 + market_ret), index=idx, name="XU100")
    stock = pd.DataFrame({"A": 100 * np.cumprod(1 + 2 * market_ret)}, index=idx)
    assert ind.Beta(100)(stock, index_close=index_close)["A"].iloc[-1] == pytest.approx(2.0)


def test_trend_r2_is_one_for_steady_growth_and_minus_one_for_decline():
    up = frame(100 * 1.01 ** np.arange(60))
    assert ind.TrendR2(50)(up)["A"].iloc[-1] == pytest.approx(1.0)
    assert ind.TrendR2(50)(1 / up)["A"].iloc[-1] == pytest.approx(-1.0)


def test_obv_slope_on_only_rising_prices():
    close = frame(range(1, 30))
    volume = frame([1000] * 29)
    assert ind.OBVSlope(10)(close, volume=volume)["A"].iloc[-1] == pytest.approx(1.0)


def test_amihud_is_relative_to_the_median_stock():
    idx = pd.bdate_range("2024-01-01", periods=30)
    moves = np.where(np.arange(30) % 2 == 0, 1.01, 0.99)
    close = pd.DataFrame({"A": 100 * np.cumprod(moves), "B": 100 * np.cumprod(moves),
                          "C": 100 * np.cumprod(moves)}, index=idx)
    volume = pd.DataFrame({"A": 1000.0, "B": 1000.0, "C": 250.0}, index=idx)
    out = ind.AmihudIlliquidity(10)(close, volume=volume).iloc[-1]
    assert out["A"] == pytest.approx(1.0)
    assert out["C"] == pytest.approx(4.0, rel=0.05)      # 4x less traded -> 4x the price impact


def test_every_indicator_is_registered():
    assert {"SMA", "RSI", "MACDHistogram", "Beta", "MaxDrawdown", "Alpha"} <= set(ind.REGISTRY)
