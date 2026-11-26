"""Which stocks were in BIST100 on which day (point-in-time universe).

Scoring the stocks that are in the index *today* over 2021-2026 would test
only the survivors: companies that fell out of the index, often after a
crash, would be missing, and the backtest would look better than reality.

Borsa Istanbul changes the index every quarter and announces the stocks that
enter and leave. Starting from today's official list and undoing those
changes one period at a time gives the exact list of every past period:

    members before a change = members after - added + removed
"""
from __future__ import annotations

import pandas as pd

INDEX_SIZE = 100


class MembershipError(ValueError):
    """The change list does not fit the official list (a change is missing or wrong)."""


def membership_periods(current: list[str], changes: pd.DataFrame,
                       size: int | None = INDEX_SIZE) -> dict[pd.Timestamp, frozenset[str]]:
    """Member list valid from each effective date on (the oldest gets `Timestamp.min`).

    `changes` has the columns effective_date, action ('add' / 'remove'), ticker.
    Every step is checked, so a wrong or missing announcement cannot slip in silently.
    """
    members = set(current)
    if size is not None and len(members) != size:
        raise MembershipError(f"the current list has {len(members)} stocks, not {size}")
    dates = sorted(pd.to_datetime(changes.effective_date).unique(), reverse=True)
    periods: dict[pd.Timestamp, frozenset[str]] = {}
    for date in dates:
        step = changes[pd.to_datetime(changes.effective_date) == date]
        added = set(step.loc[step.action == "add", "ticker"])
        removed = set(step.loc[step.action == "remove", "ticker"])
        if not added <= members:
            raise MembershipError(f"{date.date()}: added but not a member afterwards: {sorted(added - members)}")
        if removed & members:
            raise MembershipError(f"{date.date()}: removed but still a member: {sorted(removed & members)}")
        periods[pd.Timestamp(date)] = frozenset(members)
        members = (members - added) | removed
        if size is not None and len(members) != size:
            raise MembershipError(f"before {date.date()} the index would have {len(members)} stocks")
    periods[pd.Timestamp.min] = frozenset(members)
    return dict(sorted(periods.items()))


def membership_panel(periods: dict[pd.Timestamp, frozenset[str]], dates: pd.DatetimeIndex,
                     tickers: list[str]) -> pd.DataFrame:
    """Boolean panel (dates x tickers): True if the stock was in the index that day."""
    panel = pd.DataFrame(False, index=dates, columns=tickers)
    starts = list(periods)
    for start, end in zip(starts, starts[1:] + [None]):
        rows = (dates >= start) & ((dates < end) if end is not None else True)
        cols = [t for t in periods[start] if t in panel.columns]
        panel.loc[rows, cols] = True
    return panel


def ever_members(periods: dict[pd.Timestamp, frozenset[str]]) -> list[str]:
    return sorted(set().union(*periods.values()))
