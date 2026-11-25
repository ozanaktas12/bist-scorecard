"""Backtest: replays the decision process on past dates and checks the outcome.

Decisions are taken on the first trading day of every month using only the
data known on that day (guaranteed by the causal indicators, see
tests/test_no_lookahead.py). Only the *evaluation* looks into the future: a
decision is judged by the stock's return over the horizon, relative to BIST100.

Two views are produced:
* decision level - every (date, stock) decision with its later excess return,
  used for hit rates, score buckets and the flag analysis;
* portfolio level - a simulated investor who buys the BUY list every month.
  For 3- and 12-month horizons each monthly portfolio is held for the whole
  horizon, so K portfolios overlap (Jegadeesh & Titman, 1993).
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


def forward_return(prices: pd.DataFrame | pd.Series, days: int):
    """Return from the close of day t to the close of day t + days (the future outcome)."""
    return prices.shift(-days) / prices - 1


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

    def hit_rates(self) -> pd.Series:
        t = self.resolved
        buy, sell = t[t.decision == Decision.BUY.value], t[t.decision == Decision.SELL.value]
        return pd.Series({
            "BUY decisions": len(buy),
            "BUY hit rate": (buy.excess > 0).mean(),
            "SELL decisions": len(sell),
            "SELL hit rate": (sell.excess < 0).mean(),
            "Avg excess BUY": buy.excess.mean(),
            "Avg excess HOLD": t[t.decision == Decision.HOLD.value].excess.mean(),
            "Avg excess SELL": sell.excess.mean(),
            "BUY-SELL spread": buy.excess.mean() - sell.excess.mean(),
        })

    def by_score_bucket(self, buckets: int = 5) -> pd.DataFrame:
        """Average excess return per score quintile: should rise from Q1 to Q5."""
        t = self.resolved.copy()
        t["bucket"] = t.groupby("date")["score"].transform(
            lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False) + 1)
        return t.groupby("bucket")["excess"].agg(["mean", "median", "count",
                                                  lambda e: (e > 0).mean()]).rename(
            columns={"<lambda_0>": "share beating index"})

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
        """Do BUY decisions with a red flag perform differently from clean ones?

        Flags are risk warnings, so besides the average the dispersion and the
        worst decile are reported: a flag may signal *risk* without lowering the mean.
        """
        buys = self.resolved[self.resolved.decision == Decision.BUY.value]
        return self._summary({
            "BUY with red flag": buys[buys.red],
            "BUY with orange flag only": buys[~buys.red & buys.orange],
            "BUY without flags": buys[~buys.red & ~buys.orange],
        })

    def flag_rules_effect(self, flag_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Outcome of every single flag rule over *all* decisions."""
        t = self.resolved
        flagged = {}
        for key, panel in flag_panels.items():
            hit = panel.stack()
            hit.index.names = ["date", "ticker"]
            mask = t.set_index(["date", "ticker"]).index.map(hit.to_dict().get).fillna(False)
            flagged[key] = t[np.asarray(mask, dtype=bool)]
        flagged["(no flag at all)"] = t[~t.red & ~t.orange & ~t.green]
        flagged["(all decisions)"] = t
        return self._summary(flagged)


class Backtester:
    def __init__(self, market: MarketData, rules: DecisionRules = SETTINGS.rules,
                 start: str = BACKTEST_START):
        self.market, self.rules = market, rules
        self.dates = decision_dates(market.dates, start)
        self.flags = FlagEngine(market)
        self.engines = {h: ScoreEngine(market, h, self.flags, rules) for h in Horizon}
        self.red = self.flags.has_level(Level.RED)
        self.orange = self.flags.has_level(Level.ORANGE)
        self.green = self.flags.has_level(Level.GREEN)
        # Monthly returns between consecutive decision dates (portfolio simulation).
        month_ends = list(self.dates[1:]) + [market.dates[-1]]
        self.stock_month = (market.close.loc[month_ends].to_numpy() /
                            market.close.loc[self.dates].to_numpy() - 1)
        self.stock_month = pd.DataFrame(self.stock_month, index=self.dates, columns=market.tickers)
        self.index_month = pd.Series(market.index_close.loc[month_ends].to_numpy() /
                                     market.index_close.loc[self.dates].to_numpy() - 1,
                                     index=self.dates)

    # ------------------------------------------------------------ decision level
    def decision_log(self, horizon: Horizon, profile: str, rules: DecisionRules | None = None,
                     weights_total: pd.DataFrame | None = None) -> DecisionLog:
        engine, cfg = self.engines[horizon], HORIZONS[horizon]
        total = weights_total if weights_total is not None else engine.total_score(profile)
        total = total.loc[self.dates]
        decisions = decide(total, rules or self.rules)
        stock_fwd = forward_return(self.market.close, cfg.forward_days).loc[self.dates]
        index_fwd = forward_return(self.market.index_close, cfg.forward_days).loc[self.dates]
        excess = stock_fwd.sub(index_fwd, axis=0)

        table = pd.DataFrame({
            "score": total.stack(),
            "decision": decisions.stack(),
            "return": stock_fwd.stack(),
            "excess": excess.stack(),
            "red": self.red.loc[self.dates].stack(),
            "orange": self.orange.loc[self.dates].stack(),
            "green": self.green.loc[self.dates].stack(),
        }).dropna(subset=["score"])
        table.index.names = ["date", "ticker"]
        return DecisionLog(horizon, profile, table.reset_index())

    # ----------------------------------------------------------- portfolio level
    def portfolio(self, log: DecisionLog, selection: pd.DataFrame | None = None) -> pd.Series:
        """Monthly returns of an investor who holds every BUY list for the horizon.

        `selection` (bool panel date x ticker) overrides the BUY list, which is
        how the random-selection benchmark reuses this function.
        """
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
            # Without any BUY the money waits in the index fund.
            month = np.nanmean(cohorts) if cohorts else self.index_month[date]
            returns.append(month - monthly_cost if cohorts else month)
        return pd.Series(returns, index=self.dates, name=f"{log.horizon.value}/{log.profile}")

    def equal_weight_universe(self) -> pd.Series:
        """Buy-and-hold benchmark: every available stock, equal weights."""
        return self.stock_month.mean(axis=1).rename("Equal-weight BIST100")

    def random_portfolios(self, log: DecisionLog, n: int = 200, seed: int = 7) -> pd.DataFrame:
        """Same number of stocks as the BUY list each month, but picked at random."""
        rng = np.random.default_rng(seed)
        buys = log.table[log.table.decision == Decision.BUY.value]
        counts = buys.groupby("date").size().reindex(self.dates, fill_value=0)
        available = log.table.pivot(index="date", columns="ticker", values="score").notna()
        available = available.reindex(index=self.dates, columns=self.market.tickers, fill_value=False)
        runs = {}
        for k in range(n):
            pick = pd.DataFrame(False, index=self.dates, columns=self.market.tickers)
            for date, count in counts.items():
                pool = available.columns[available.loc[date]]
                if count:
                    pick.loc[date, rng.choice(pool, size=count, replace=False)] = True
            runs[k] = self.portfolio(log, selection=pick)
        return pd.DataFrame(runs)


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
        "Sharpe (rf=0)": monthly.mean() / monthly.std() * np.sqrt(12),
        "Max drawdown": (growth / growth.cummax() - 1).min(),
    }
    if benchmark is not None:
        diff = monthly - benchmark.reindex(monthly.index)
        stats["Months beating index"] = (diff > 0).mean()
        stats["Excess CAGR vs index"] = stats["CAGR"] - (
            (1 + benchmark.reindex(monthly.index)).prod() ** (1 / years) - 1)
    return pd.Series(stats)
