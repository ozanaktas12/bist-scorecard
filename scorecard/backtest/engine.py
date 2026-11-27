"""Backtest: a decision every month using past data only, checked afterwards.

Trades happen one day after the decision. For the 3- and 12-month horizons
the monthly portfolios overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import HORIZONS, SETTINGS, Decision, DecisionRules, Horizon
from ..data import MarketData
from ..scoring import FlagEngine, Level, ScoreEngine, decide

BACKTEST_START = "2021-01-01"
MONTHS_PER_HORIZON = {Horizon.SHORT: 1, Horizon.MEDIUM: 3, Horizon.LONG: 12}


def decision_dates(dates: pd.DatetimeIndex, start: str = BACKTEST_START) -> pd.DatetimeIndex:
    """First trading day of every month from `start` on."""
    dates = dates[dates >= start]
    firsts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    return pd.DatetimeIndex(firsts.values)


def forward_return(prices: pd.DataFrame | pd.Series, days: int, lag: int = 0):
    """Return from day t + lag to day t + lag + days. Only used to evaluate decisions."""
    return prices.shift(-(lag + days)) / prices.shift(-lag) - 1


def newey_west_t(series: pd.Series, lags: int) -> float:
    """t-statistic of the mean, corrected for overlapping returns (Newey-West)."""
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    variance = e @ e / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(variance / n))


@dataclass
class DecisionLog:
    """One row per (decision date, stock)."""

    horizon: Horizon
    profile: str
    table: pd.DataFrame

    @property
    def resolved(self) -> pd.DataFrame:
        """Decisions whose evaluation period has already ended."""
        return self.table.dropna(subset=["excess"])

    def _of(self, decision: Decision) -> pd.DataFrame:
        t = self.resolved
        return t[t.decision == decision.value]

    def hit_rates(self) -> pd.Series:
        t = self.resolved
        buy, sell = self._of(Decision.BUY), self._of(Decision.SELL)
        base = (t.excess > 0).mean()
        return pd.Series({
            "BUY decisions": len(buy),
            "BUY hit rate": (buy.excess > 0).mean(),
            "Base rate (any stock beats index)": base,
            "SELL decisions": len(sell),
            "SELL hit rate": (sell.excess < 0).mean(),
            "Base rate (any stock lags index)": 1 - base,
            "Avg excess BUY": buy.excess.mean(),
            "Avg excess HOLD": self._of(Decision.HOLD).excess.mean(),
            "Avg excess SELL": sell.excess.mean(),
            "Avg excess all": t.excess.mean(),
        })

    def monthly_spreads(self) -> pd.DataFrame:
        """Per decision date: BUY average minus the average of all stocks, and BUY minus SELL."""
        t = self.resolved
        all_mean = t.groupby("date").excess.mean()
        buy_mean = self._of(Decision.BUY).groupby("date").excess.mean()
        sell_mean = self._of(Decision.SELL).groupby("date").excess.mean()
        return pd.DataFrame({"BUY vs all": buy_mean - all_mean, "BUY vs SELL": buy_mean - sell_mean})

    def significance(self) -> pd.Series:
        """Mean monthly spreads and their Newey-West t-statistics (|t| > 2 ~ significant)."""
        spreads = self.monthly_spreads()
        lags = MONTHS_PER_HORIZON[self.horizon] - 1
        return pd.Series({
            "BUY vs all (mean)": spreads["BUY vs all"].mean(),
            "BUY vs all (t)": newey_west_t(spreads["BUY vs all"], lags),
            "BUY vs SELL (mean)": spreads["BUY vs SELL"].mean(),
            "BUY vs SELL (t)": newey_west_t(spreads["BUY vs SELL"], lags),
            "months": len(spreads.dropna()),
        })

    def by_score_bucket(self, buckets: int = 5) -> pd.DataFrame:
        """Average excess return per score quintile: should rise from Q1 to Q5."""
        t = self.resolved.copy()
        t["bucket"] = t.groupby("date")["score"].transform(
            lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False) + 1)
        grouped = t.groupby("bucket")["excess"]
        return pd.DataFrame({"mean": grouped.mean(), "median": grouped.median(),
                             "count": grouped.size(),
                             "share beating index": grouped.apply(lambda e: (e > 0).mean())})

    @staticmethod
    def _summary(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame({name: {
            "count": len(g),
            "avg excess": g.excess.mean(),
            "median excess": g.excess.median(),
            "hit rate": (g.excess > 0).mean(),
            "std of excess": g.excess.std(),
            "worst 10%": g.excess.quantile(0.10),
        } for name, g in groups.items()}).T

    def flag_effect(self) -> pd.DataFrame:
        """BUY decisions with and without flags: average, spread and worst 10 %."""
        buys = self._of(Decision.BUY)
        return self._summary({
            "BUY with red flag": buys[buys.red],
            "BUY with orange flag only": buys[~buys.red & buys.orange],
            "BUY without flags": buys[~buys.red & ~buys.orange],
        })

class Backtester:
    def __init__(self, market: MarketData, rules: DecisionRules = SETTINGS.rules,
                 start: str = BACKTEST_START, lag: int = SETTINGS.execution_lag):
        self.market, self.rules, self.lag = market, rules, lag
        self.dates = decision_dates(market.dates, start)
        self.flags = FlagEngine(market)
        self.engines = {h: ScoreEngine(market, h, self.flags, rules) for h in Horizon}
        self.red = self.flags.has_level(Level.RED)
        self.orange = self.flags.has_level(Level.ORANGE)
        self.stock_month, self.index_month = self._monthly_returns()

    def _monthly_returns(self) -> tuple[pd.DataFrame, pd.Series]:
        """Returns between consecutive trade days (decision day + lag)."""
        positions = self.market.dates.get_indexer(self.dates) + self.lag
        last = len(self.market.dates) - 1
        trade_days = self.market.dates[np.minimum(positions, last)]
        next_trade_days = list(trade_days[1:]) + [self.market.dates[last]]
        stock = (self.market.close.loc[next_trade_days].to_numpy() /
                 self.market.close.loc[trade_days].to_numpy() - 1)
        index = (self.market.index_close.loc[next_trade_days].to_numpy() /
                 self.market.index_close.loc[trade_days].to_numpy() - 1)
        return (pd.DataFrame(stock, index=self.dates, columns=self.market.tickers),
                pd.Series(index, index=self.dates))

    def excess_returns(self, horizon: Horizon) -> pd.DataFrame:
        """Future return over the horizon minus BIST100's, for every decision date."""
        days = HORIZONS[horizon].forward_days
        stock = forward_return(self.market.close, days, self.lag)
        index = forward_return(self.market.index_close, days, self.lag)
        return stock.sub(index, axis=0).loc[self.dates]

    # Decision level
    def decision_log(self, horizon: Horizon, profile: str, rules: DecisionRules | None = None,
                     total: pd.DataFrame | None = None) -> DecisionLog:
        """`total` lets the calibration pass scores built from its own weights."""
        engine, cfg = self.engines[horizon], HORIZONS[horizon]
        total = (total if total is not None else engine.total_score(profile)).loc[self.dates]
        decisions = decide(total, rules or self.rules)
        stock_fwd = forward_return(self.market.close, cfg.forward_days, self.lag).loc[self.dates]

        table = pd.DataFrame({
            "score": total.stack(),
            "decision": decisions.stack(),
            "return": stock_fwd.stack(),
            "excess": self.excess_returns(horizon).stack(),
            "red": self.red.loc[self.dates].stack(),
            "orange": self.orange.loc[self.dates].stack(),
        }).dropna(subset=["score"])
        table.index.names = ["date", "ticker"]
        return DecisionLog(horizon, profile, table.reset_index())

    def metric_ic(self, horizon: Horizon, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        """Average monthly rank correlation of each metric with the later excess return."""
        dates = self.dates if dates is None else dates
        excess = self.excess_returns(horizon).loc[dates]
        engine = self.engines[horizon]
        panels = dict(engine.features.raw)
        panels["TOTAL SCORE (Balanced)"] = engine.total_score("Balanced")
        rows = {}
        for key, panel in panels.items():
            ics = []
            for date in dates:
                a, b = panel.loc[date], excess.loc[date]
                ok = a.notna() & b.notna()
                if ok.sum() >= 20:
                    ics.append(a[ok].rank().corr(b[ok].rank()))
            ics = pd.Series(ics, dtype=float)
            rows[key] = {"mean IC": ics.mean(),
                         "t": newey_west_t(ics, MONTHS_PER_HORIZON[horizon] - 1)}
        return pd.DataFrame(rows).T

    # Portfolio level
    def portfolio(self, log: DecisionLog, selection: pd.DataFrame | None = None) -> pd.Series:
        """Monthly return of holding every BUY list for the horizon (`selection` replaces the BUY list)."""
        if selection is None:
            buys = log.table[log.table.decision == Decision.BUY.value]
            selection = pd.crosstab(buys.date, buys.ticker).astype(bool)
            selection = selection.reindex(index=self.dates, columns=self.market.tickers,
                                          fill_value=False)
        hold_months = MONTHS_PER_HORIZON[log.horizon]
        # Turnover: every month 1/K of the capital is sold and re-bought.
        monthly_cost = 2 * SETTINGS.transaction_cost / hold_months
        returns = []
        for i, date in enumerate(self.dates):
            cohorts = []
            for j in range(max(0, i - hold_months + 1), i + 1):
                picked = selection.iloc[j]
                if picked.any():
                    cohorts.append(self.stock_month.loc[date, picked].mean())
            if cohorts:
                returns.append(np.nanmean(cohorts) - monthly_cost)
            else:
                # Without any BUY the money waits in an index fund.
                returns.append(self.index_month[date])
        return pd.Series(returns, index=self.dates, name=f"{log.horizon.value}/{log.profile}")

    def equal_weight_universe(self) -> pd.Series:
        """Benchmark: every stock in BIST100 on the decision day, equal weights, monthly rebalanced."""
        members = self.market.members.loc[self.dates]
        return self.stock_month.where(members).mean(axis=1).rename("Equal-weight BIST100")


def load_cpi(path: str | None = None) -> pd.Series:
    """Monthly Turkish CPI change (TCMB), indexed by month period."""
    from ..data import PROJECT_ROOT

    table = pd.read_csv(path or PROJECT_ROOT / "data/macro/cpi_tr_monthly.csv", comment="#")
    return pd.Series(table.cpi_monthly_pct.to_numpy() / 100,
                     index=pd.PeriodIndex(table.month, freq="M"), name="cpi")


def real_returns(monthly: pd.Series, cpi: pd.Series) -> pd.Series:
    """Deflate monthly returns; months without published CPI are dropped."""
    inflation = cpi.reindex(monthly.index.to_period("M")).to_numpy()
    real = (1 + monthly) / (1 + inflation) - 1
    return pd.Series(real, index=monthly.index).dropna()


def performance(monthly: pd.Series, benchmark: pd.Series | None = None) -> pd.Series:
    """Summary statistics of a monthly return series."""
    monthly = monthly.dropna()
    growth = (1 + monthly).cumprod()
    years = len(monthly) / 12
    stats = {
        "Total return": growth.iloc[-1] - 1,
        "CAGR": growth.iloc[-1] ** (1 / years) - 1,
        "Volatility (ann.)": monthly.std() * np.sqrt(12),
        "Max drawdown": (growth / growth.cummax() - 1).min(),
    }
    if benchmark is not None:
        bench = benchmark.reindex(monthly.index)
        stats["Months beating benchmark"] = (monthly > bench).mean()
        stats["CAGR vs benchmark"] = stats["CAGR"] - ((1 + bench).prod() ** (1 / years) - 1)
    return pd.Series(stats)
