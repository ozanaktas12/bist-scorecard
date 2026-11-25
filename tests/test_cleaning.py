import pandas as pd
import pytest

from scorecard.data.cleaning import apply_adjustments, find_corporate_actions

DATES = pd.bdate_range("2024-01-01", periods=5)


def test_bonus_issue_is_detected_and_removed():
    # A 1:1 bonus issue halves the quoted price on day 3 while the market is flat.
    close = pd.DataFrame({"A": [100, 102, 51, 52, 53]}, index=DATES, dtype=float)
    volume = pd.DataFrame({"A": [10, 10, 20, 20, 20]}, index=DATES, dtype=float)
    index_close = pd.Series([1000] * 5, index=DATES, dtype=float)

    found = find_corporate_actions(close, index_close, daily_limit=0.25, tolerance=0.08)
    assert [(a.ticker, a.date) for a in found] == [("A", DATES[2])]

    clean_close, clean_volume = apply_adjustments(close, volume, found)
    assert clean_close["A"].pct_change().abs().max() < 0.05
    assert clean_volume["A"].iloc[0] == pytest.approx(20)


def test_market_wide_gap_is_kept():
    # Reopening after a market closure: the stock and the index jump together.
    close = pd.DataFrame({"A": [100, 100, 130, 131, 132]}, index=DATES, dtype=float)
    index_close = pd.Series([1000, 1000, 1280, 1290, 1300], index=DATES, dtype=float)
    assert find_corporate_actions(close, index_close, daily_limit=0.25, tolerance=0.08) == []
