"""Data cleaning for Borsa Istanbul price series.

Yahoo Finance sometimes misses corporate actions (e.g. bonus share issues,
"bedelsiz sermaye artırımı"), which shows up as a one-day jump far beyond the
+/-10 % daily price limit of Borsa Istanbul. A genuine move of that size is
only possible when the *whole market* gaps (for example on 15 Feb 2023, when
the exchange reopened after the earthquake closure), so a jump is treated as
a data error only if it is not explained by the index move on the same day.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Adjustment:
    ticker: str
    date: pd.Timestamp
    raw_return: float
    index_return: float
    factor: float           # all prices before `date` are multiplied by this


def find_corporate_actions(close: pd.DataFrame, index_close: pd.Series,
                           daily_limit: float, tolerance: float) -> list[Adjustment]:
    returns = close.pct_change(fill_method=None)
    index_returns = index_close.reindex(close.index).pct_change(fill_method=None).fillna(0.0)
    found: list[Adjustment] = []
    for ticker in close.columns:
        r = returns[ticker]
        suspicious = r[(r.abs() > daily_limit) & ((r - index_returns).abs() > tolerance)]
        for date, value in suspicious.items():
            m = index_returns.loc[date]
            # Rescale history so that the jump becomes equal to the market move.
            factor = (1 + value) / (1 + m)
            found.append(Adjustment(ticker, date, float(value), float(m), float(factor)))
    return found


def apply_adjustments(close: pd.DataFrame, volume: pd.DataFrame,
                      adjustments: list[Adjustment]) -> tuple[pd.DataFrame, pd.DataFrame]:
    close, volume = close.copy(), volume.copy()
    for adj in adjustments:
        before = close.index < adj.date
        close.loc[before, adj.ticker] *= adj.factor
        # A split multiplies the share count, so past volumes scale the other way.
        volume.loc[before, adj.ticker] /= adj.factor
    return close, volume
