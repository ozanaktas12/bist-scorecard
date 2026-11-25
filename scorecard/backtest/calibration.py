"""Data-driven pillar weights, tested out of sample.

The hand-made profiles use weights chosen by judgement. Here the weights are
searched on a *training* period (2021-2023) and then evaluated on a *test*
period (2024 onwards) that the search never saw. If the calibrated weights
only shine in the training years, they were overfitted.

Objective: the average monthly rank correlation (IC) between the total score
and the later excess return. It uses every stock, not only the 20 BUYs, so it
is far less noisy than optimising the BUY return directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from ..config import PROFILES, Horizon, Pillar
from .engine import Backtester

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"


def weight_grid(step: float = 0.05, minimum: float = 0.05) -> list[tuple[float, ...]]:
    """Every combination of 5 weights on a `step` grid that sums to 1."""
    units, low = round(1 / step), round(minimum / step)
    combos = []
    for a, b, c, d in product(range(low, units + 1), repeat=4):
        e = units - a - b - c - d
        if e >= low:
            combos.append(tuple(round(x * step, 10) for x in (a, b, c, d, e)))
    return combos


def _rank_ic(scores: np.ndarray, excess: np.ndarray) -> float:
    """Mean over dates of the Spearman correlation (rows = dates)."""
    ics = []
    for s, e in zip(scores, excess):
        ok = ~np.isnan(s) & ~np.isnan(e)
        if ok.sum() >= 20:
            ics.append(pd.Series(s[ok]).rank().corr(pd.Series(e[ok]).rank()))
    return float(np.mean(ics)) if ics else float("nan")


@dataclass
class CalibrationResult:
    horizon: Horizon
    weights: dict[Pillar, float]
    table: pd.DataFrame          # train / test IC of calibrated vs hand-made profiles


def calibrate(bt: Backtester, horizon: Horizon, step: float = 0.05) -> CalibrationResult:
    engine = bt.engines[horizon]
    pillars = list(Pillar)
    excess = bt.excess_returns(horizon)

    # The training sample may only contain decisions whose outcome was known
    # before the test period starts, otherwise the test years leak in.
    days = engine.cfg.forward_days
    positions = bt.market.dates.get_indexer(bt.dates) + bt.lag + days
    outcome_dates = bt.market.dates[np.minimum(positions, len(bt.market.dates) - 1)]
    train = bt.dates[(outcome_dates <= pd.Timestamp(TRAIN_END)) &
                     (positions < len(bt.market.dates))]
    test = bt.dates[bt.dates >= TEST_START]

    stack = {name: np.stack([engine.pillar_scores[p].loc[dates].to_numpy() for p in pillars])
             for name, dates in (("train", train), ("test", test))}
    target = {"train": excess.loc[train].to_numpy(), "test": excess.loc[test].to_numpy()}

    def ic(weights, sample):
        total = np.tensordot(np.asarray(weights), stack[sample], axes=1)
        return _rank_ic(total, target[sample])

    best = max(weight_grid(step), key=lambda w: ic(w, "train"))
    rows = {"Calibrated": {"train IC": ic(best, "train"), "test IC": ic(best, "test")}}
    for name, profile in PROFILES.items():
        w = [profile.weights[p] for p in pillars]
        rows[name] = {"train IC": ic(w, "train"), "test IC": ic(w, "test")}
    table = pd.DataFrame(rows).T
    table.attrs["train months"], table.attrs["test months"] = len(train), len(test)
    return CalibrationResult(horizon, dict(zip(pillars, best)), table)
