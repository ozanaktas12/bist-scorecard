"""The 16 scored metrics and how each raw value becomes a 0-100 score.

A metric is declared once as a `MetricSpec`: which pillar it belongs to, how
its raw value is computed from the indicators, the piecewise-linear mapping
from raw value to score, and a sentence that explains the value to the user.

Scores use *absolute*, economically chosen breakpoints (not ranks), so a
score of 80 means "good" in any market, while the ranking step in
decision.py decides what is good *relative to the other stocks*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..config import HIGH_52W_WINDOW, SLOPE_LOOKBACK, HorizonConfig, Pillar
from ..data import MarketData

Points = list[tuple[float, float]]   # (raw value, score) breakpoints, raw values ascending


def piecewise_score(raw: pd.DataFrame, points: Points) -> pd.DataFrame:
    """Linear interpolation between breakpoints; values outside are clamped."""
    xs, ys = zip(*points)
    values = np.interp(raw.to_numpy(dtype=float), xs, ys)
    values[np.isnan(raw.to_numpy(dtype=float))] = np.nan
    return pd.DataFrame(values, index=raw.index, columns=raw.columns)


def symmetric(scale: float) -> Points:
    """-scale -> 0, 0 -> 50, +scale -> 100."""
    return [(-scale, 0.0), (0.0, 50.0), (scale, 100.0)]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    pillar: Pillar
    points: Points
    describe: Callable[[float], str]


class FeatureBuilder:
    """Computes the raw value panel of every metric for one horizon.

    Indicators are evaluated once on the full history; because each one is
    causal, reading row t gives exactly what was knowable on day t.
    """

    def __init__(self, market: MarketData, cfg: HorizonConfig, min_sector_size: int = 3):
        self.market, self.cfg = market, cfg
        self.min_sector_size = min_sector_size
        self.specs = build_specs(cfg)
        self.raw = self._compute_raw()

    def _compute_raw(self) -> dict[str, pd.DataFrame]:
        c, m, cfg = self.market.close, self.market, self.cfg
        ctx = dict(index_close=m.index_close, volume=m.volume, sectors=m.sectors)
        short_ma, long_ma = ind.SMA(cfg.short_ma)(c), ind.SMA(cfg.long_ma)(c)
        raw = {
            # Trend
            "dist_short": c / short_ma - 1,
            "dist_long": c / long_ma - 1,
            "long_slope": long_ma / long_ma.shift(SLOPE_LOOKBACK) - 1,
            "trend_r2": ind.TrendR2(cfg.long_ma)(c),
            # Momentum
            "rsi": ind.RSI(cfg.rsi_period)(c),
            "macd": ind.MACDHistogram(*cfg.macd)(c),
            "high_52w": ind.HighProximity(HIGH_52W_WINDOW)(c),
            "risk_adj_return": ind.RiskAdjustedReturn(cfg.momentum_window, cfg.momentum_skip,
                                                      cfg.vol_window)(c),
            # Risk
            "vol_ratio": ind.Volatility(cfg.vol_window)(c).div(
                ind.Volatility(cfg.vol_window)(m.index_close.to_frame()).iloc[:, 0], axis=0),
            "max_drawdown": ind.MaxDrawdown(cfg.vol_window)(c),
            "beta": ind.Beta(cfg.beta_window)(c, **ctx),
            # Relative strength
            "alpha": ind.Alpha(cfg.momentum_window, cfg.momentum_skip, cfg.beta_window)(c, **ctx),
            "sector_rel": ind.SectorRelativeReturn(cfg.momentum_window, cfg.momentum_skip,
                                                   self.min_sector_size)(c, **ctx),
            # Volume
            "obv_slope": ind.OBVSlope(cfg.volume_window)(c, **ctx),
            "liquidity": ind.AmihudIlliquidity(cfg.volume_window)(c, **ctx),
            "rel_volume": ind.RelativeVolume(20, 60)(c, **ctx),
        }
        missing = {s.key for s in self.specs} ^ set(raw)
        assert not missing, f"metric definitions and raw values differ: {missing}"
        return raw

    def scores(self) -> dict[str, pd.DataFrame]:
        return {s.key: piecewise_score(self.raw[s.key], s.points) for s in self.specs}


def build_specs(cfg: HorizonConfig) -> list[MetricSpec]:
    pct = lambda v: f"{v:+.1%}"
    return [
        # ---------------------------------------------------------------- Trend
        MetricSpec("dist_short", f"Price vs {cfg.short_ma}-day average", Pillar.TREND,
                   symmetric(cfg.distance_scale),
                   lambda v: f"Price is {pct(v)} away from its {cfg.short_ma}-day average"),
        MetricSpec("dist_long", f"Price vs {cfg.long_ma}-day average", Pillar.TREND,
                   symmetric(cfg.distance_scale * 1.5),
                   lambda v: f"Price is {pct(v)} away from its {cfg.long_ma}-day average"),
        MetricSpec("long_slope", f"Slope of {cfg.long_ma}-day average", Pillar.TREND,
                   symmetric(cfg.slope_scale),
                   lambda v: f"The {cfg.long_ma}-day average moved {pct(v)} in {SLOPE_LOOKBACK} days"),
        MetricSpec("trend_r2", "Trend consistency (R²)", Pillar.TREND,
                   [(-0.8, 0.0), (0.0, 50.0), (0.8, 100.0)],
                   lambda v: ("Steady up-trend" if v > 0.5 else "Steady down-trend" if v < -0.5
                              else "No clear direction") + f" (signed R² = {v:+.2f})"),
        # ------------------------------------------------------------- Momentum
        MetricSpec("rsi", f"RSI ({cfg.rsi_period})", Pillar.MOMENTUM,
                   # Hump-shaped: healthy strength (55-70) is best, overbought is penalised.
                   [(0, 10.0), (30, 20.0), (50, 60.0), (55, 90.0), (70, 100.0),
                    (80, 50.0), (90, 20.0), (100, 10.0)],
                   lambda v: f"RSI is {v:.0f}" + (" (overbought)" if v > 70 else
                                                 " (oversold)" if v < 30 else "")),
        MetricSpec("macd", "MACD histogram", Pillar.MOMENTUM, symmetric(cfg.macd_scale),
                   lambda v: "MACD is " + ("above" if v > 0 else "below") + " its signal line"),
        MetricSpec("high_52w", "Distance to 52-week high", Pillar.MOMENTUM,
                   [(0.60, 0.0), (0.85, 50.0), (1.00, 100.0)],
                   lambda v: f"Price is {1 - v:.1%} below its 52-week high" if v < 0.999
                   else "Price is at its 52-week high"),
        MetricSpec("risk_adj_return", "Risk-adjusted return", Pillar.MOMENTUM,
                   [(-1.5, 0.0), (0.0, 50.0), (1.5, 100.0)],
                   lambda v: f"Momentum z-score {v:+.2f} (return measured in units of its own risk)"),
        # ----------------------------------------------------------------- Risk
        MetricSpec("vol_ratio", "Volatility vs BIST100", Pillar.RISK,
                   # A single stock is naturally ~1.6x as volatile as the diversified index.
                   [(1.0, 100.0), (1.6, 60.0), (2.2, 30.0), (3.0, 0.0)],
                   lambda v: f"Volatility is {v:.1f}x the index volatility"),
        MetricSpec("max_drawdown", "Maximum drawdown", Pillar.RISK,
                   [(-0.60, 0.0), (-0.40, 20.0), (-0.20, 60.0), (0.0, 100.0)],
                   lambda v: f"Worst fall from a peak in the window: {v:.1%}"),
        MetricSpec("beta", "Beta", Pillar.RISK,
                   [(0.5, 100.0), (1.0, 70.0), (1.5, 30.0), (2.0, 0.0)],
                   lambda v: f"Beta is {v:.2f} (moves {v:.1f}x the market)"),
        # ---------------------------------------------------- Relative strength
        MetricSpec("alpha", "Alpha vs BIST100", Pillar.RELATIVE, symmetric(cfg.return_scale),
                   lambda v: f"Beat the beta-adjusted index return by {pct(v)}" if v >= 0
                   else f"Lagged the beta-adjusted index return by {pct(v)}"),
        MetricSpec("sector_rel", "Return vs sector peers", Pillar.RELATIVE,
                   symmetric(cfg.return_scale),
                   lambda v: f"{pct(v)} relative to the median of its sector peers"),
        # --------------------------------------------------------------- Volume
        MetricSpec("obv_slope", "On-balance volume", Pillar.VOLUME,
                   [(-0.3, 0.0), (0.0, 50.0), (0.3, 100.0)],
                   lambda v: ("Net buying volume" if v > 0 else "Net selling volume") + f" ({v:+.2f})"),
        MetricSpec("liquidity", "Liquidity (price impact)", Pillar.VOLUME,
                   [(0.25, 100.0), (1.0, 60.0), (4.0, 20.0), (10.0, 0.0)],
                   lambda v: f"Trading moves the price {v:.2f}x as much as for a typical BIST100 stock"
                   + (" (very liquid)" if v < 0.25 else " (thinly traded)" if v > 4 else "")),
        MetricSpec("rel_volume", "Relative volume (20/60 days)", Pillar.VOLUME,
                   [(0.5, 20.0), (1.0, 50.0), (1.5, 80.0), (2.5, 90.0)],
                   lambda v: f"Recent trading interest is {v:.2f}x normal"),
    ]
