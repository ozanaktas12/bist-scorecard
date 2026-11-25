"""Synthetic market used by the tests, so they run offline and are reproducible."""
import numpy as np
import pandas as pd
import pytest

from scorecard.data import MarketData


@pytest.fixture(scope="session")
def market() -> MarketData:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", periods=900)
    tickers = [f"S{i:02d}" for i in range(12)]
    index_ret = rng.normal(0.0006, 0.012, len(dates))
    betas = np.linspace(0.6, 1.6, len(tickers))
    stock_ret = index_ret[:, None] * betas + rng.normal(0, 0.015, (len(dates), len(tickers)))
    close = pd.DataFrame(100 * np.cumprod(1 + stock_ret, axis=0), index=dates, columns=tickers)
    volume = pd.DataFrame(rng.integers(1_000, 50_000, close.shape).astype(float),
                          index=dates, columns=tickers)
    index_close = pd.Series(1000 * np.cumprod(1 + index_ret), index=dates, name="XU100")
    sectors = pd.Series(["Bank"] * 5 + ["Energy"] * 5 + ["Sport"] * 2, index=tickers)
    return MarketData(close, volume, index_close, sectors)
