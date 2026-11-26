"""Turns metric scores into pillar scores, a total score, a decision and a scorecard."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from ..config import HORIZONS, SETTINGS, Decision, DecisionRules, Horizon, Pillar, RiskProfile, get_profile
from ..data import MarketData
from .flags import FlagEngine, FlagRule, Level
from .metrics import FeatureBuilder, MetricSpec


def decide(total: pd.DataFrame, rules: DecisionRules = SETTINGS.rules) -> pd.DataFrame:
    """Cross-sectional decision for every date.

    BUY  = score >= buy_min_score AND among the best `top_fraction` of that day
    SELL = among the worst `bottom_fraction` of that day
    HOLD = everything else; stocks without a score get no decision.
    """
    pct_rank = total.rank(axis=1, pct=True, ascending=True)
    buy = (total >= rules.buy_min_score) & (pct_rank > 1 - rules.top_fraction)
    sell = pct_rank <= rules.bottom_fraction
    out = pd.DataFrame(Decision.HOLD.value, index=total.index, columns=total.columns, dtype=object)
    out = out.mask(sell, Decision.SELL.value).mask(buy, Decision.BUY.value)
    return out.where(total.notna())


@dataclass
class MetricResult:
    spec: MetricSpec
    raw: float
    score: float
    percentile: float      # share of the other stocks with a lower score on this metric

    @property
    def explanation(self) -> str:
        return self.spec.describe(self.raw)


@dataclass
class ScoreCard:
    ticker: str
    sector: str
    date: pd.Timestamp
    horizon: Horizon
    profile: str
    total: float
    rank: int
    universe_size: int
    decision: Decision
    pillars: dict[Pillar, float]
    metrics: list[MetricResult]
    flags: list[FlagRule] = field(default_factory=list)

    @property
    def red_flags(self) -> list[FlagRule]:
        return [f for f in self.flags if f.level == Level.RED]

    def strengths(self, n: int = 3) -> list[MetricResult]:
        """Metrics where this stock stands out from the others, not just scores high.

        Ranking by the absolute score made "low beta" the top strength of most
        stocks in a falling market; the percentile shows what is special.
        """
        good = [m for m in self.metrics if m.score >= 50 and m.percentile >= 0.5]
        return sorted(good, key=lambda m: -m.percentile)[:n]

    def weaknesses(self, n: int = 3) -> list[MetricResult]:
        bad = [m for m in self.metrics if m.score < 50 and m.percentile < 0.5]
        return sorted(bad, key=lambda m: m.percentile)[:n]


class ScoreEngine:
    """Scores the whole universe for one horizon; profiles only change the weights."""

    def __init__(self, market: MarketData, horizon: Horizon, flags: FlagEngine | None = None,
                 rules: DecisionRules = SETTINGS.rules):
        self.market, self.horizon, self.rules = market, horizon, rules
        self.cfg = HORIZONS[horizon]
        self.features = FeatureBuilder(market, self.cfg, SETTINGS.data.min_sector_size)
        self.metric_scores = self.features.scores()
        self.flags = flags or FlagEngine(market)

    @cached_property
    def pillar_scores(self) -> dict[Pillar, pd.DataFrame]:
        result = {}
        for pillar in Pillar:
            keys = [s.key for s in self.features.specs if s.pillar == pillar]
            stacked = [self.metric_scores[k] for k in keys]
            mean = sum(stacked) / len(stacked)   # NaN if any metric is missing
            result[pillar] = mean
        return result

    def total_score(self, profile: RiskProfile | str) -> pd.DataFrame:
        if isinstance(profile, str):
            profile = get_profile(profile, self.horizon)
        return sum(self.pillar_scores[p] * w for p, w in profile.weights.items())

    def decisions(self, profile: RiskProfile | str) -> pd.DataFrame:
        return decide(self.total_score(profile), self.rules)

    def latest_date(self) -> pd.Timestamp:
        totals = self.total_score("Balanced")
        return totals.dropna(how="all").index[-1]

    def ranking(self, profile: str, date: pd.Timestamp | None = None) -> pd.DataFrame:
        """Table of every stock on `date`, best score first."""
        date = date or self.latest_date()
        total = self.total_score(profile).loc[date]
        table = pd.DataFrame({
            "Sector": self.market.sectors,
            "Score": total,
            "Decision": self.decisions(profile).loc[date],
            **{p.value: self.pillar_scores[p].loc[date] for p in Pillar},
        })
        table["Red flags"] = [len([f for f in self.flags.flags_at(t, date) if f.level == Level.RED])
                              for t in table.index]
        return table.dropna(subset=["Score"]).sort_values("Score", ascending=False)

    def _share_below(self, key: str, date: pd.Timestamp, ticker: str) -> float:
        """Share of the *other* stocks that score lower on this metric (ties count half)."""
        ranks = self.metric_scores[key].loc[date].rank()
        return float((ranks[ticker] - 1) / max(ranks.count() - 1, 1))

    def scorecard(self, ticker: str, profile: str, date: pd.Timestamp | None = None) -> ScoreCard:
        date = date or self.latest_date()
        total_row = self.total_score(profile).loc[date]
        if pd.isna(total_row.get(ticker)):
            raise ValueError(f"{ticker} has not enough history for the {self.horizon.value} horizon on {date.date()}")
        ranks = total_row.rank(ascending=False, method="min")
        return ScoreCard(
            ticker=ticker,
            sector=str(self.market.sectors[ticker]),
            date=date,
            horizon=self.horizon,
            profile=profile,
            total=float(total_row[ticker]),
            rank=int(ranks[ticker]),
            universe_size=int(total_row.notna().sum()),
            decision=Decision(self.decisions(profile).at[date, ticker]),
            pillars={p: float(self.pillar_scores[p].at[date, ticker]) for p in Pillar},
            metrics=[MetricResult(s, float(self.features.raw[s.key].at[date, ticker]),
                                  float(self.metric_scores[s.key].at[date, ticker]),
                                  self._share_below(s.key, date, ticker))
                     for s in self.features.specs],
            flags=self.flags.flags_at(ticker, date),
        )
