"""Data layer: where price data comes from and how it is cached.

`DataProvider` is an abstract interface; `YFinanceProvider` implements it with
Yahoo Finance, and `CachedProvider` wraps *any* provider with an on-disk CSV
cache (decorator pattern), so the rest of the system never knows the source.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Aligned daily data for the whole universe (rows = dates, columns = tickers)."""

    close: pd.DataFrame
    volume: pd.DataFrame
    index_close: pd.Series
    sectors: pd.Series          # ticker -> sector name
    members: pd.DataFrame | None = None   # dates x tickers: in BIST100 that day? (None = always)
    unavailable: tuple[str, ...] = ()     # past members without price data (delisted)

    def __post_init__(self) -> None:
        if self.members is None:
            self.members = pd.DataFrame(True, index=self.close.index, columns=self.close.columns)

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def until(self, date: pd.Timestamp) -> "MarketData":
        """Return a copy that contains only data available on `date` (no look-ahead)."""
        return MarketData(
            close=self.close.loc[:date],
            volume=self.volume.loc[:date],
            index_close=self.index_close.loc[:date],
            sectors=self.sectors,
            members=self.members.loc[:date],
            unavailable=self.unavailable,
        )

    def members_on(self, date: pd.Timestamp) -> list[str]:
        row = self.members.loc[date]
        return list(row.index[row])


class DataProvider(ABC):
    @abstractmethod
    def fetch_prices(self, tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (close, volume) frames with one column per ticker."""

    @abstractmethod
    def fetch_index(self, ticker: str, start: str) -> pd.Series:
        """Return the daily close series of a market index."""


class YFinanceProvider(DataProvider):
    """Yahoo Finance quotes Borsa Istanbul symbols with an '.IS' suffix."""

    SUFFIX = ".IS"

    def fetch_prices(self, tickers, start):
        import yfinance as yf  # imported lazily: tests run without network access

        symbols = [t + self.SUFFIX for t in tickers]
        raw = yf.download(symbols, start=start, auto_adjust=True, progress=False,
                          group_by="column", threads=True)
        close = raw["Close"].rename(columns=lambda s: s.removesuffix(self.SUFFIX))
        volume = raw["Volume"].rename(columns=lambda s: s.removesuffix(self.SUFFIX))
        # Bulk downloads sometimes skip a few symbols silently: retry those one by one.
        for ticker in [t for t in tickers if t not in close or close[t].isna().all()]:
            for _ in range(3):
                single = yf.download(ticker + self.SUFFIX, start=start, auto_adjust=True,
                                     progress=False)
                if len(single):
                    close[ticker] = single["Close"].squeeze()
                    volume[ticker] = single["Volume"].squeeze()
                    break
        return close.reindex(columns=tickers), volume.reindex(columns=tickers)

    def fetch_index(self, ticker, start):
        import yfinance as yf

        raw = yf.download(ticker + self.SUFFIX, start=start, auto_adjust=True, progress=False)
        return raw["Close"].squeeze().rename(ticker)


class CachedProvider(DataProvider):
    """Stores every download as CSV and serves later calls from disk."""

    def __init__(self, inner: DataProvider, cache_dir: str | Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.csv.gz"   # compressed: ~170 stocks x 8 years

    def fetch_prices(self, tickers, start):
        close_path, volume_path = self._path("close"), self._path("volume")
        if close_path.exists() and volume_path.exists():
            close = pd.read_csv(close_path, index_col=0, parse_dates=True)
            volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
            if set(tickers) <= set(close.columns):
                log.info("Prices loaded from cache (%s)", close_path)
                return close[tickers], volume[tickers]
        close, volume = self.inner.fetch_prices(tickers, start)
        close.to_csv(close_path)
        volume.to_csv(volume_path)
        return close, volume

    def fetch_index(self, ticker, start):
        path = self._path(f"index_{ticker}")
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns").rename(ticker)
        series = self.inner.fetch_index(ticker, start)
        series.to_frame().to_csv(path)
        return series
