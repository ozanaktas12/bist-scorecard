"""Point-in-time index membership, rebuilt backwards from today's list."""
import pandas as pd
import pytest

from scorecard.data.universe import (MembershipError, ever_members, membership_panel,
                                     membership_periods)

TODAY = ["A", "B", "C", "D"]
CHANGES = pd.DataFrame({
    "effective_date": ["2022-01-03", "2022-01-03", "2023-07-03", "2023-07-03"],
    "action": ["add", "remove", "add", "remove"],
    "ticker": ["C", "X", "D", "Y"],
})


def test_changes_are_undone_period_by_period():
    periods = membership_periods(TODAY, CHANGES, size=4)
    first, mid, last = periods.values()
    assert first == {"A", "B", "X", "Y"}          # before 2022: C not yet in, X still in
    assert mid == {"A", "B", "C", "Y"}            # 2022 - mid 2023
    assert last == set(TODAY)
    assert ever_members(periods) == ["A", "B", "C", "D", "X", "Y"]


def test_panel_marks_members_day_by_day():
    periods = membership_periods(TODAY, CHANGES, size=4)
    dates = pd.bdate_range("2021-12-30", "2023-07-05")
    panel = membership_panel(periods, dates, ["C", "X"])
    assert not panel.at[pd.Timestamp("2021-12-31"), "C"] and panel.at[pd.Timestamp("2022-01-03"), "C"]
    assert panel.at[pd.Timestamp("2021-12-31"), "X"] and not panel.at[pd.Timestamp("2022-01-03"), "X"]


def test_an_inconsistent_change_list_is_rejected():
    wrong = CHANGES.copy()
    wrong.loc[0, "ticker"] = "Z"                  # "added" but not in today's list
    with pytest.raises(MembershipError, match="added but not a member"):
        membership_periods(TODAY, wrong, size=4)
    unbalanced = CHANGES.iloc[[0, 2, 3]]          # an addition without its removal
    with pytest.raises(MembershipError, match="would have 3 stocks"):
        membership_periods(TODAY, unbalanced, size=4)


def test_non_members_are_not_scored(market):
    from dataclasses import replace
    from scorecard.config import Horizon
    from scorecard.scoring import ScoreEngine

    members = market.members.copy()
    members.loc[:, "S00"] = False
    engine = ScoreEngine(replace(market, members=members), Horizon.MEDIUM)
    total = engine.total_score("Balanced")
    assert total["S00"].isna().all() and total["S01"].notna().any()
