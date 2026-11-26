"""Tests for the backtest arithmetic: the numbers in the report come from here."""
import numpy as np
import pandas as pd
import pytest

from scorecard.backtest import (Backtester, DecisionLog, decision_dates, forward_return,
                                performance, real_returns)
from scorecard.backtest.engine import newey_west_t
from scorecard.config import SETTINGS, Horizon


@pytest.fixture(scope="module")
def bt(market):
    return Backtester(market)


def test_forward_return_starts_after_the_lag():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    # Decision on day 0, trade on day 1 (110), outcome 2 days later on day 3 (133.1).
    assert forward_return(prices, days=2, lag=1).iloc[0] == pytest.approx(133.1 / 110 - 1)
    # The last days have no known future and must stay empty.
    assert forward_return(prices, days=2, lag=1).iloc[-3:].isna().all()


def test_decision_dates_are_first_trading_day_of_each_month():
    dates = pd.bdate_range("2021-01-01", "2021-03-31")
    firsts = decision_dates(dates, start="2021-01-01")
    assert list(firsts.strftime("%Y-%m-%d")) == ["2021-01-01", "2021-02-01", "2021-03-01"]


def test_newey_west_without_lags_is_the_plain_t_statistic():
    x = pd.Series([0.02, -0.01, 0.03, 0.00, 0.01, 0.04, -0.02, 0.02])
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_t(x, lags=0) == pytest.approx(plain)


def test_newey_west_is_more_careful_when_results_are_autocorrelated():
    # A slowly drifting series: neighbouring months look alike, like overlapping 3-month outcomes.
    x = pd.Series(np.repeat([0.03, 0.01, 0.02, -0.005, 0.025, 0.015], 3))
    assert newey_west_t(x, lags=2) < newey_west_t(x, lags=0)


def test_monthly_returns_use_the_trade_day_after_the_decision(bt, market):
    first = bt.dates[0]
    trade = market.dates[market.dates.get_loc(first) + 1]
    next_trade = market.dates[market.dates.get_loc(bt.dates[1]) + 1]
    expected = market.close.at[next_trade, "S00"] / market.close.at[trade, "S00"] - 1
    assert bt.stock_month.at[first, "S00"] == pytest.approx(expected)


def test_one_month_portfolio_is_the_chosen_stock_minus_costs(bt):
    pick = pd.DataFrame(False, index=bt.dates, columns=bt.market.tickers)
    pick["S03"] = True
    log = DecisionLog(Horizon.SHORT, "test", pd.DataFrame())
    result = bt.portfolio(log, selection=pick)
    expected = bt.stock_month["S03"] - 2 * SETTINGS.transaction_cost
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_three_month_portfolio_averages_the_overlapping_cohorts(bt):
    # Month 0 buys S01, month 1 buys S02, month 2 buys S01 again.
    pick = pd.DataFrame(False, index=bt.dates, columns=bt.market.tickers)
    pick.iloc[0, 1] = pick.iloc[1, 2] = pick.iloc[2, 1] = True
    log = DecisionLog(Horizon.MEDIUM, "test", pd.DataFrame())
    result = bt.portfolio(log, selection=pick)
    month2 = bt.dates[2]
    cohorts = [bt.stock_month.at[month2, "S01"], bt.stock_month.at[month2, "S02"],
               bt.stock_month.at[month2, "S01"]]
    assert result[month2] == pytest.approx(np.mean(cohorts) - 2 * SETTINGS.transaction_cost / 3)


def test_without_any_buy_the_money_follows_the_index(bt):
    pick = pd.DataFrame(False, index=bt.dates, columns=bt.market.tickers)
    log = DecisionLog(Horizon.SHORT, "test", pd.DataFrame())
    pd.testing.assert_series_equal(bt.portfolio(log, selection=pick), bt.index_month,
                                   check_names=False)


def test_performance_of_a_known_series():
    monthly = pd.Series([0.01] * 12 + [-0.5] + [0.0] * 11,
                        index=pd.date_range("2021-01-01", periods=24, freq="MS"))
    stats = performance(monthly)
    assert stats["Total return"] == pytest.approx(1.01 ** 12 * 0.5 - 1)
    assert stats["CAGR"] == pytest.approx((1.01 ** 12 * 0.5) ** 0.5 - 1)
    assert stats["Max drawdown"] == pytest.approx(-0.5)


def test_real_returns_remove_inflation():
    monthly = pd.Series([0.02], index=pd.DatetimeIndex(["2022-03-01"]))
    cpi = pd.Series([0.01], index=pd.PeriodIndex(["2022-03"], freq="M"))
    assert real_returns(monthly, cpi).iloc[0] == pytest.approx(1.02 / 1.01 - 1)
