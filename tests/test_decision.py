import numpy as np
import pandas as pd

from scorecard.config import DecisionRules
from scorecard.scoring import decide


def test_buy_needs_high_score_and_top_rank():
    scores = pd.DataFrame([np.arange(10, 101, 10, dtype=float)], columns=list("ABCDEFGHIJ"))
    out = decide(scores, DecisionRules()).iloc[0]
    assert list(out[out == "BUY"].index) == ["I", "J"]          # top 20 %, both >= 60
    assert set(out[out == "SELL"].index) == {"A", "B"}          # bottom 20 %


def test_bear_market_gives_no_buy():
    scores = pd.DataFrame([[20, 30, 35, 45, 55]], columns=list("ABCDE"), dtype=float)
    out = decide(scores, DecisionRules()).iloc[0]
    assert "BUY" not in set(out)        # best stock has only 55 < 60


def test_missing_score_gives_no_decision():
    scores = pd.DataFrame([[np.nan, 80.0, 30.0]], columns=list("ABC"))
    assert pd.isna(decide(scores, DecisionRules()).iloc[0]["A"])


def test_sell_is_relative_not_absolute():
    scores = pd.DataFrame([[70, 75, 80, 85, 90]], columns=list("ABCDE"), dtype=float)
    out = decide(scores, DecisionRules()).iloc[0]
    assert list(out[out == "SELL"].index) == ["A"]      # weakest 20 %, even with a score of 70
