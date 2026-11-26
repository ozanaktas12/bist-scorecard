"""Loading the BIST100 universe into a clean `MarketData` object."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import SETTINGS, DataConfig
from .cleaning import Adjustment, apply_adjustments, find_corporate_actions
from .provider import CachedProvider, DataProvider, MarketData, YFinanceProvider
from .universe import MembershipError, ever_members, membership_panel, membership_periods

__all__ = ["MarketData", "DataProvider", "YFinanceProvider", "CachedProvider", "Adjustment",
           "MembershipError", "load_universe", "load_periods", "load_market_data"]

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_universe(cfg: DataConfig = SETTINGS.data) -> pd.Series:
    """Today's official BIST100: ticker -> sector."""
    table = pd.read_csv(PROJECT_ROOT / cfg.universe_file)
    return table.set_index("ticker")["sector"]


def load_periods(cfg: DataConfig = SETTINGS.data) -> dict[pd.Timestamp, frozenset[str]]:
    """Member list of every index period, rebuilt backwards from today's list."""
    current = list(load_universe(cfg).index)
    path = PROJECT_ROOT / cfg.changes_file
    if not path.exists():
        log.warning("%s not found: today's members are used for the whole history", path)
        return {pd.Timestamp.min: frozenset(current)}
    changes = pd.read_csv(path, comment="#")
    renames_path = PROJECT_ROOT / cfg.renames_file
    if renames_path.exists():
        # Announcements use the code of their time; prices are stored under today's code.
        renames = pd.read_csv(renames_path)
        changes["ticker"] = changes["ticker"].replace(dict(zip(renames.old_ticker, renames.new_ticker)))
    return membership_periods(current, changes)


def load_market_data(provider: DataProvider | None = None,
                     cfg: DataConfig = SETTINGS.data) -> tuple[MarketData, list[Adjustment]]:
    """Load and clean the prices of every stock that was ever in BIST100.

    Also returns the corporate-action corrections that were applied.
    """
    periods = load_periods(cfg)
    tickers = ever_members(periods)
    sectors = pd.read_csv(PROJECT_ROOT / cfg.sectors_file).set_index("ticker")["sector"]
    sectors = sectors.reindex(tickers).fillna(load_universe(cfg)).fillna("Other")
    provider = provider or CachedProvider(YFinanceProvider(), PROJECT_ROOT / cfg.cache_dir)

    close, volume = provider.fetch_prices(tickers, cfg.start_date)
    index_close = provider.fetch_index(cfg.index_ticker, cfg.start_date)

    # Trading calendar = days on which the index traded.
    dates = index_close.dropna().index
    close = close.reindex(dates).ffill(limit=5)
    volume = volume.reindex(dates).fillna(0.0).astype(float)
    index_close = index_close.reindex(dates)

    missing = [t for t in tickers if close[t].notna().sum() == 0]
    if missing:
        # Delisted companies: Yahoo Finance no longer has their prices.
        log.warning("No price data for %s, they are dropped", missing)
        close, volume = close.drop(columns=missing), volume.drop(columns=missing)
        sectors = sectors.drop(missing)

    adjustments = find_corporate_actions(close, index_close, cfg.daily_limit, cfg.market_move_tolerance)
    close, volume = apply_adjustments(close, volume, adjustments)
    members = membership_panel(periods, dates, list(close.columns))
    log.info("%d tickers loaded (%d ever in BIST100 without data), %d corporate-action corrections",
             close.shape[1], len(missing), len(adjustments))
    return MarketData(close, volume, index_close, sectors, members, tuple(missing)), adjustments
