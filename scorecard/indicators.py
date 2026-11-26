"""Technical indicators, implemented from scratch with NumPy and pandas.

Every indicator works on a whole panel at once (a DataFrame with one column
per ticker) and is *causal*: the value on day t uses only data up to day t.
That property is what makes the backtest free of look-ahead bias, and it is
verified in tests/test_no_lookahead.py.

Indicators register themselves in `REGISTRY` through the `@register`
decorator, so the UI and the report can list every available indicator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Panel = pd.DataFrame
REGISTRY: dict[str, type["Indicator"]] = {}


def register(cls: type["Indicator"]) -> type["Indicator"]:
    REGISTRY[cls.__name__] = cls
    return cls


class Indicator(ABC):
    """Base class: subclasses implement `compute`, calling the object runs it."""

    @abstractmethod
    def compute(self, close: Panel, **context) -> Panel: ...

    def __call__(self, close: Panel, **context) -> Panel:
        return self.compute(close, **context)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


def daily_returns(close: Panel) -> Panel:
    return close.pct_change(fill_method=None)


# --------------------------------------------------------------------------- trend
@register
class SMA(Indicator):
    """Simple moving average."""

    def __init__(self, window: int):
        self.window = window

    def compute(self, close, **_):
        return close.rolling(self.window, min_periods=self.window).mean()


@register
class EMA(Indicator):
    """Exponential moving average with smoothing factor 2 / (span + 1)."""

    def __init__(self, span: int):
        self.span = span

    def compute(self, close, **_):
        return close.ewm(span=self.span, adjust=False, min_periods=self.span).mean()


@register
class TrendR2(Indicator):
    """Signed R² of a linear fit of log-price on time.

    +1 means a perfectly straight up-trend, -1 a straight down-trend and values
    near 0 a choppy market without direction. R² equals the squared
    correlation between log-price and time.
    """

    def __init__(self, window: int):
        self.window = window

    def compute(self, close, **_):
        log_price = np.log(close)
        time = pd.Series(np.arange(len(close), dtype=float), index=close.index)
        corr = log_price.rolling(self.window, min_periods=self.window).corr(time)
        return np.sign(corr) * corr**2


# ------------------------------------------------------------------------ momentum
@register
class RSI(Indicator):
    """Relative Strength Index with Wilder's smoothing (alpha = 1 / period)."""

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, close, **_):
        change = close.diff()
        gain = change.clip(lower=0)
        loss = -change.clip(upper=0)
        avg_gain = self._wilder_average(gain)
        avg_loss = self._wilder_average(loss)
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        # No losses at all in the window -> RSI is 100 by definition.
        return rsi.mask(avg_loss == 0, 100.0)

    def _wilder_average(self, values: Panel) -> Panel:
        """Wilder's smoothing: new_avg = old_avg + (value - old_avg) / period."""
        return values.ewm(alpha=1 / self.period, adjust=False, min_periods=self.period).mean()


@register
class MACDHistogram(Indicator):
    """MACD line minus its signal line, divided by price to compare stocks."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast, self.slow, self.signal = fast, slow, signal

    def compute(self, close, **_):
        macd = EMA(self.fast)(close) - EMA(self.slow)(close)
        signal = macd.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()
        return (macd - signal) / close


@register
class HighProximity(Indicator):
    """Price divided by its highest close of the last `window` days (1.0 = at the high)."""

    def __init__(self, window: int = 252):
        self.window = window

    def compute(self, close, **_):
        return close / close.rolling(self.window, min_periods=self.window).max()


@register
class PeriodReturn(Indicator):
    """Return over `window` days that ends `skip` days ago."""

    def __init__(self, window: int, skip: int = 0):
        self.window, self.skip = window, skip

    def compute(self, close, **_):
        return close.shift(self.skip) / close.shift(self.skip + self.window) - 1


@register
class RiskAdjustedReturn(Indicator):
    """Period return divided by the volatility expected over that period (a z-score)."""

    def __init__(self, window: int, skip: int = 0, vol_window: int = 63):
        self.window, self.skip, self.vol_window = window, skip, vol_window

    def compute(self, close, **_):
        ret = PeriodReturn(self.window, self.skip)(close)
        vol = Volatility(self.vol_window, annualize=False)(close)
        return ret / (vol * np.sqrt(self.window))


# ---------------------------------------------------------------------------- risk
@register
class Volatility(Indicator):
    """Standard deviation of daily returns, annualised with 252 trading days."""

    def __init__(self, window: int, annualize: bool = True):
        self.window, self.annualize = window, annualize

    def compute(self, close, **_):
        vol = daily_returns(close).rolling(self.window, min_periods=self.window).std()
        return vol * np.sqrt(252) if self.annualize else vol


@register
class MaxDrawdown(Indicator):
    """Worst peak-to-trough fall inside the last `window` days (e.g. -0.35 = -35 %)."""

    def __init__(self, window: int):
        self.window = window

    def compute(self, close, **_):
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for col in close.columns:
            values = close[col].to_numpy(dtype=float)
            if len(values) < self.window:
                continue
            windows = sliding_window_view(values, self.window)
            running_peak = np.maximum.accumulate(windows, axis=1)
            drawdown = np.min(windows / running_peak - 1, axis=1)
            result.iloc[self.window - 1:, result.columns.get_loc(col)] = drawdown
        return result


@register
class Beta(Indicator):
    """Sensitivity of the stock to the market: cov(r_stock, r_index) / var(r_index)."""

    def __init__(self, window: int):
        self.window = window

    def compute(self, close, *, index_close: pd.Series, **_):
        r = daily_returns(close)
        m = daily_returns(index_close)
        cov = r.rolling(self.window, min_periods=self.window).cov(m)
        var = m.rolling(self.window, min_periods=self.window).var()
        return cov.div(var, axis=0)


# -------------------------------------------------------------------------- volume
@register
class OBVSlope(Indicator):
    """Change of On-Balance Volume over `window` days, scaled by the volume traded.

    OBV adds the day's volume on up days and subtracts it on down days. The
    result lies in [-1, 1]: +1 means every share in the window traded on an up day.
    """

    def __init__(self, window: int):
        self.window = window

    def compute(self, close, *, volume: Panel, **_):
        direction = np.sign(close.diff()).fillna(0.0)
        signed = direction * volume
        flow = signed.rolling(self.window, min_periods=self.window).sum()
        total = volume.rolling(self.window, min_periods=self.window).sum()
        return flow / total.replace(0, np.nan)


@register
class AmihudIlliquidity(Indicator):
    """Price impact of trading (Amihud, 2002): average |daily return| per TL traded.

    Inflation inflates TL volumes every year, so the raw value is divided by
    the median of the index members on the same day: 1.0 = typical BIST100
    liquidity, 4.0 = trading moves the price four times as much as for a
    typical stock.
    """

    def __init__(self, window: int):
        self.window = window

    def compute(self, close, *, volume: Panel, members: Panel | None = None, **_):
        traded_value = (close * volume).replace(0, np.nan)
        impact = daily_returns(close).abs() / traded_value
        average = impact.rolling(self.window, min_periods=self.window // 2).mean()
        typical = (average if members is None else average.where(members)).median(axis=1)
        return average.div(typical, axis=0)


@register
class RelativeVolume(Indicator):
    """Recent average volume relative to a longer average (1.0 = normal interest)."""

    def __init__(self, short: int = 20, long: int = 60):
        self.short, self.long = short, long

    def compute(self, close, *, volume: Panel, **_):
        recent = volume.rolling(self.short, min_periods=self.short).mean()
        normal = volume.rolling(self.long, min_periods=self.long).mean()
        return recent / normal.replace(0, np.nan)


# --------------------------------------------------------------- relative strength
@register
class Alpha(Indicator):
    """Return in excess of what the stock's beta predicts: r_i - beta * r_index."""

    def __init__(self, window: int, skip: int = 0, beta_window: int = 252):
        self.window, self.skip, self.beta_window = window, skip, beta_window

    def compute(self, close, *, index_close: pd.Series, **_):
        stock_ret = PeriodReturn(self.window, self.skip)(close)
        index_ret = PeriodReturn(self.window, self.skip)(index_close.to_frame())[index_close.name]
        beta = Beta(self.beta_window)(close, index_close=index_close)
        return stock_ret - beta.mul(index_ret, axis=0)


@register
class SectorRelativeReturn(Indicator):
    """Return minus the median return of the *other* index members in the same sector.

    On days when a sector has fewer than `min_size` members (the stock itself
    included), the stock is compared with the index instead.
    """

    def __init__(self, window: int, skip: int = 0, min_size: int = 3):
        self.window, self.skip, self.min_size = window, skip, min_size

    def compute(self, close, *, sectors: pd.Series, index_close: pd.Series,
                members: Panel | None = None, **_):
        ret = PeriodReturn(self.window, self.skip)(close)
        index_ret = PeriodReturn(self.window, self.skip)(index_close.to_frame())[index_close.name]
        peer_ret = ret if members is None else ret.where(members)
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for sector, names in sectors.groupby(sectors).groups.items():
            names = [t for t in names if t in close.columns]
            for ticker in names:
                peers = peer_ret[[t for t in names if t != ticker]]
                enough = peers.notna().sum(axis=1) >= self.min_size - 1
                benchmark = peers.median(axis=1).where(enough, index_ret)
                result[ticker] = ret[ticker] - benchmark.fillna(index_ret)
        return result
