"""Central configuration: investment horizons, risk profiles and decision rules.

Every tunable number of the system lives in this file so that the logic
modules stay free of "magic numbers" and the backtest can vary them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Horizon(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class Pillar(str, Enum):
    TREND = "Trend"
    MOMENTUM = "Momentum"
    RISK = "Risk"
    RELATIVE = "Relative Strength"
    VOLUME = "Volume"


class Decision(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass(frozen=True)
class HorizonConfig:
    """Look-back windows (in trading days) that define one investment horizon.

    The horizon changes *which windows* the indicators use and *how far ahead*
    a decision is evaluated; the scoring logic itself is identical.
    """

    horizon: Horizon
    label: str
    short_ma: int
    long_ma: int
    rsi_period: int
    macd: tuple[int, int, int]          # fast, slow, signal
    momentum_window: int                # return window for momentum / relative strength
    momentum_skip: int                  # most recent days excluded (short-term reversal effect)
    vol_window: int                     # volatility and max-drawdown window
    beta_window: int
    volume_window: int                  # OBV slope and up/down volume window
    forward_days: int                   # evaluation period of a decision
    # Scales used to turn raw values into 0-100 scores (wider horizon -> larger moves)
    distance_scale: float
    slope_scale: float
    macd_scale: float
    return_scale: float


HORIZONS: dict[Horizon, HorizonConfig] = {
    Horizon.SHORT: HorizonConfig(
        Horizon.SHORT, "Short term (1 month)",
        short_ma=20, long_ma=50, rsi_period=14, macd=(12, 26, 9),
        momentum_window=21, momentum_skip=0, vol_window=63, beta_window=126,
        volume_window=21, forward_days=21,
        distance_scale=0.08, slope_scale=0.04, macd_scale=0.010, return_scale=0.10,
    ),
    Horizon.MEDIUM: HorizonConfig(
        Horizon.MEDIUM, "Medium term (3 months)",
        short_ma=50, long_ma=100, rsi_period=14, macd=(24, 52, 18),
        momentum_window=63, momentum_skip=0, vol_window=126, beta_window=252,
        volume_window=63, forward_days=63,
        distance_scale=0.15, slope_scale=0.06, macd_scale=0.020, return_scale=0.20,
    ),
    Horizon.LONG: HorizonConfig(
        Horizon.LONG, "Long term (12 months)",
        short_ma=100, long_ma=200, rsi_period=28, macd=(50, 100, 30),
        momentum_window=252, momentum_skip=21, vol_window=252, beta_window=252,
        volume_window=126, forward_days=252,
        distance_scale=0.25, slope_scale=0.08, macd_scale=0.030, return_scale=0.40,
    ),
}

SLOPE_LOOKBACK = 20          # days over which the long moving average slope is measured
HIGH_52W_WINDOW = 252        # one trading year


@dataclass(frozen=True)
class RiskProfile:
    name: str
    weights: dict[Pillar, float]

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights of profile '{self.name}' must sum to 1, got {total}")


PROFILES: dict[str, RiskProfile] = {
    "Conservative": RiskProfile("Conservative", {
        Pillar.TREND: 0.30, Pillar.MOMENTUM: 0.15, Pillar.RISK: 0.30,
        Pillar.RELATIVE: 0.15, Pillar.VOLUME: 0.10,
    }),
    "Balanced": RiskProfile("Balanced", {
        Pillar.TREND: 0.25, Pillar.MOMENTUM: 0.25, Pillar.RISK: 0.20,
        Pillar.RELATIVE: 0.20, Pillar.VOLUME: 0.10,
    }),
    "Aggressive": RiskProfile("Aggressive", {
        Pillar.TREND: 0.20, Pillar.MOMENTUM: 0.35, Pillar.RISK: 0.10,
        Pillar.RELATIVE: 0.25, Pillar.VOLUME: 0.10,
    }),
}


@dataclass(frozen=True)
class DecisionRules:
    """BUY needs an absolute AND a relative condition; SELL needs either one."""

    buy_min_score: float = 60.0
    sell_max_score: float = 40.0
    top_fraction: float = 0.20       # BUY candidates: best 20 % of the universe on that date
    bottom_fraction: float = 0.20    # SELL candidates: worst 20 %


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2019-01-01"
    index_ticker: str = "XU100"
    universe_file: str = "data/universe/bist100.csv"
    cache_dir: str = "data/cache"
    daily_limit: float = 0.25              # beyond the widest BIST price limit ever used (+/-20 %)
    market_move_tolerance: float = 0.08    # jump vs. index move to be called a corporate action
    min_sector_size: int = 3               # smaller sectors fall back to the index


@dataclass(frozen=True)
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    rules: DecisionRules = field(default_factory=DecisionRules)
    transaction_cost: float = 0.002        # 0.2 % per trade (commission + spread)


SETTINGS = Settings()
