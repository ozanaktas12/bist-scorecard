"""Risk flags: warnings shown next to the decision, they never change it.

The decision is left to the user ("BUY, but be careful"); the flags explain
*what* to be careful about. Each rule is evaluated on the full history as a
boolean panel, so the backtest can test whether flagged decisions really
performed worse.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .. import indicators as ind
from ..data import MarketData


class Level(str, Enum):
    RED = "red"          # critical
    ORANGE = "orange"    # caution
    GREEN = "green"      # positive signal


@dataclass(frozen=True)
class FlagRule:
    key: str
    level: Level
    title: str
    message: str


RULES: dict[str, FlagRule] = {r.key: r for r in [
    FlagRule("falling_knife", Level.RED, "Falling knife",
             "Below a falling 200-day average after a sharp 3-month decline."),
    FlagRule("overbought", Level.RED, "Overbought",
             "RSI(14) above 80: a short-term pull-back is likely."),
    FlagRule("speculative", Level.RED, "Speculative run",
             "Series of limit-up days on unusual volume; such runs often reverse sharply."),
    FlagRule("illiquid", Level.RED, "Thin trading",
             "Among the least traded stocks in the index; prices are easy to move."),
    FlagRule("overbought_mild", Level.ORANGE, "Getting overbought",
             "RSI(14) between 70 and 80."),
    FlagRule("overextended", Level.ORANGE, "Overextended",
             "More than 50 % above the 200-day average; a correction risk builds up."),
    FlagRule("high_volatility", Level.ORANGE, "High volatility",
             "Volatility is more than twice that of BIST100."),
    FlagRule("deep_drawdown", Level.ORANGE, "Deep drawdown",
             "More than 35 % below the 52-week high; recovery may take long."),
    FlagRule("turnaround", Level.GREEN, "Possible turnaround",
             "Still below the 200-day average, but reclaimed the 50-day average, "
             "RSI recovered from oversold and lows are rising."),
]}


class FlagEngine:
    """Evaluates every rule for all stocks and dates (horizon independent)."""

    def __init__(self, market: MarketData):
        c, v = market.close, market.volume
        ret = ind.daily_returns(c)
        sma50, sma200 = ind.SMA(50)(c), ind.SMA(200)(c)
        rsi = ind.RSI(14)(c)
        ret63 = ind.PeriodReturn(63)(c)
        index_vol = ind.Volatility(63)(market.index_close.to_frame()).iloc[:, 0]
        traded_value = (c * v).rolling(20, min_periods=20).mean()

        below_200 = c < sma200
        sma200_falling = sma200 < sma200.shift(20)
        reclaimed_50 = (c > sma50) & (c.shift(10) <= sma50.shift(10))
        rsi_recovered = (rsi.rolling(20).min() < 35) & (rsi > 45)
        higher_lows = c.rolling(10).min() > c.shift(10).rolling(20).min()
        turnaround = below_200 & reclaimed_50 & rsi_recovered & higher_lows

        limit_up_days = (ret >= 0.095).astype(float).rolling(10).sum()

        self.panels: dict[str, pd.DataFrame] = {
            "falling_knife": below_200 & sma200_falling & (ret63 < -0.15) & ~turnaround,
            "overbought": rsi > 80,
            "speculative": (limit_up_days >= 3) & (ind.RelativeVolume(20, 60)(c, volume=v) > 2),
            "illiquid": traded_value.rank(axis=1, pct=True) <= 0.05,
            "overbought_mild": (rsi > 70) & (rsi <= 80),
            "overextended": c / sma200 - 1 > 0.50,
            "high_volatility": ind.Volatility(63)(c).gt(2 * index_vol, axis=0),
            "deep_drawdown": ind.HighProximity(252)(c) < 0.65,
            "turnaround": turnaround,
        }

    def flags_at(self, ticker: str, date: pd.Timestamp) -> list[FlagRule]:
        return [RULES[k] for k, panel in self.panels.items() if bool(panel.at[date, ticker])]

    def has_level(self, level: Level) -> pd.DataFrame:
        """Boolean panel: does the stock carry at least one flag of `level`?"""
        keys = [k for k, r in RULES.items() if r.level == level]
        out = self.panels[keys[0]].copy()
        for k in keys[1:]:
            out |= self.panels[k]
        return out
