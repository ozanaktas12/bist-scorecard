"""Short plain-language reading of a scorecard. Each sentence comes from a fixed rule."""
from __future__ import annotations

import math

from ..config import HORIZONS
from ..data.valuation import verdict
from .engine import ScoreCard
from .flags import Level


def _metric(card: ScoreCard, key: str):
    return next(m for m in card.metrics if m.spec.key == key)


def _character(card: ScoreCard) -> str | None:
    vol = _metric(card, "vol_ratio")
    share = sum(_metric(card, k).percentile for k in ("vol_ratio", "max_drawdown", "beta")) / 3
    if share >= 0.7:
        return f"One of the calmer stocks in the index: it swings {vol.raw:.1f}x as much as BIST100 itself."
    if share <= 0.3:
        return f"One of the more volatile stocks in the index: it swings {vol.raw:.1f}x as much as BIST100."
    return None


def _direction(card: ScoreCard) -> str:
    cfg = HORIZONS[card.horizon]
    r2 = _metric(card, "trend_r2").raw
    above = _metric(card, "dist_long").raw
    where = f"{abs(above):.0%} {'above' if above >= 0 else 'below'} its {cfg.long_ma}-day average"
    if r2 > 0.5 and above > 0:
        return f"It is in a steady up-trend and trades {where}."
    if r2 < -0.5 and above < 0:
        return f"It is in a steady down-trend and trades {where}."
    return f"It has no steady trend over the last {cfg.long_ma} days and trades {where}."


def _recent_change(card: ScoreCard) -> str | None:
    """The short average reacts first, so a split between the two shows a turn."""
    cfg = HORIZONS[card.horizon]
    short, long_ = _metric(card, "dist_short").raw, _metric(card, "dist_long").raw
    if long_ > 0 and short < 0:
        return (f"Recently it slipped below its {cfg.short_ma}-day average, "
                "so the up-move is losing pace.")
    if long_ < 0 and short > 0:
        return (f"Recently it climbed back above its {cfg.short_ma}-day average, "
                "an early sign that the fall may be easing.")
    return None


def _peers(card: ScoreCard) -> str | None:
    rel = _metric(card, "sector_rel")
    if rel.percentile >= 0.8:
        return f"It did clearly better than its sector peers ({rel.raw:+.0%} vs their median)."
    if rel.percentile <= 0.2:
        return f"It fell behind its sector peers ({rel.raw:+.0%} vs their median)."
    return None


def _interest(card: ScoreCard) -> str | None:
    rv = _metric(card, "rel_volume").raw
    if rv >= 1.5:
        return f"Trading interest has picked up: volume is {rv:.1f}x its usual level."
    if rv <= 0.6:
        return f"Trading interest has faded: volume is {rv:.1f}x its usual level."
    return None


def _valuation(row) -> str | None:
    if row is None:
        return None
    if row.get("loss_making"):
        return "The company made a loss in its last reported year, so it has no P/E."
    pb_vs = row.get("pb_vs_peer", math.nan)
    if math.isnan(pb_vs):
        return None
    peers = ("its sector peers" if row.get("peer_group") == "sector"
             else "the typical BIST100 stock")
    return f"On price-to-book ({row['pb']:.2f}) it looks {verdict(pb_vs)} {peers}."


def _warning(card: ScoreCard) -> str | None:
    red = [f.title.lower() for f in card.flags if f.level == Level.RED]
    if red:
        return "Warning: " + " and ".join(red) + "."
    return None


def describe(card: ScoreCard, valuation: dict | None = None) -> list[str]:
    """Two to six sentences, most important first."""
    parts = [_character(card), _direction(card), _recent_change(card), _peers(card),
             _interest(card), _valuation(valuation), _warning(card)]
    return [p for p in parts if p]
