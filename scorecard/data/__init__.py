"""Loading the BIST100 universe into a clean `MarketData` object."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import SETTINGS, DataConfig
from .cleaning import Adjustment, apply_adjustments, find_corporate_actions
from .provider import CachedProvider, DataProvider, MarketData, YFinanceProvider

__all__ = ["MarketData", "DataProvider", "YFinanceProvider", "CachedProvider",
           "Adjustment", "load_universe", "load_market_data"]

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_universe(cfg: DataConfig = SETTINGS.data) -> pd.Series:
    table = pd.read_csv(PROJECT_ROOT / cfg.universe_file)
    return table.set_index("ticker")["sector"]


def load_market_data(provider: DataProvider | None = None,
                     cfg: DataConfig = SETTINGS.data) -> tuple[MarketData, list[Adjustment]]:
    """Download (or read from cache), align and clean the data of the universe.

    Returns the clean data and the list of corporate-action corrections that
    were applied, so they can be reported.
    """
    sectors = load_universe(cfg)
    provider = provider or CachedProvider(YFinanceProvider(), PROJECT_ROOT / cfg.cache_dir)

    close, volume = provider.fetch_prices(list(sectors.index), cfg.start_date)
    index_close = provider.fetch_index(cfg.index_ticker, cfg.start_date)

    # Trading calendar = days on which the index traded.
    dates = index_close.dropna().index
    close = close.reindex(dates).ffill(limit=5)
    volume = volume.reindex(dates).fillna(0.0)
    index_close = index_close.reindex(dates)

    missing = [t for t in sectors.index if close[t].notna().sum() == 0]
    if missing:
        log.warning("No price data for %s, they are dropped", missing)
        close, volume = close.drop(columns=missing), volume.drop(columns=missing)
        sectors = sectors.drop(missing)

    adjustments = find_corporate_actions(close, index_close, cfg.daily_limit, cfg.market_move_tolerance)
    close, volume = apply_adjustments(close, volume, adjustments)
    log.info("%d tickers loaded, %d corporate-action corrections", close.shape[1], len(adjustments))
    return MarketData(close, volume, index_close, sectors), adjustments
