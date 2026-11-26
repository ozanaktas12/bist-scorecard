"""Runs the backtest and saves every table to results/:  python scripts/run_backtest.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scorecard.backtest import Backtester, calibrate, load_cpi, performance, real_returns
from scorecard.backtest.calibration import TEST_START
from scorecard.config import PROFILE_NAMES, Horizon
from scorecard.data import MarketData, load_market_data, load_periods

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220)


def save(table: pd.DataFrame, name: str, title: str) -> None:
    table.to_csv(OUT / f"{name}.csv")
    print(f"\n=== {title} ===")
    print(table.round(3).to_string())


def todays_list_only(market: MarketData) -> MarketData:
    """The naive universe: today's members, treated as members on every past day."""
    today = market.members_on(market.dates[-1])
    return MarketData(market.close[today], market.volume[today], market.index_close,
                      market.sectors[today])


def survivorship(market: MarketData) -> pd.DataFrame:
    """Same system, two universes: how much does using today's list flatter the results?"""
    rows = {}
    for name, data in (("today's list", todays_list_only(market)), ("point-in-time", market)):
        bt = Backtester(data)
        ew = bt.equal_weight_universe()
        for p in ("Balanced", "Data-driven"):
            log = bt.decision_log(Horizon.MEDIUM, p)
            monthly = bt.portfolio(log)
            rows[(p, name)] = {
                "BUY hit rate": log.hit_rates()["BUY hit rate"],
                "t (BUY vs all)": log.significance()["BUY vs all (t)"],
                "CAGR": performance(monthly)["CAGR"],
                "equal-weight CAGR": performance(ew)["CAGR"],
                "CAGR vs equal-weight": performance(monthly, ew)["CAGR vs benchmark"],
            }
    return pd.DataFrame(rows).T


def universe_table(market: MarketData) -> pd.DataFrame:
    periods = load_periods()
    rows = {}
    for start, members in periods.items():
        rows["earlier" if start == pd.Timestamp.min else str(start.date())] = {
            "members": len(members),
            "with price data": len(members & set(market.tickers)),
            "still in today's list": len(members & set(market.members_on(market.dates[-1]))),
        }
    return pd.DataFrame(rows).T


def main() -> None:
    market, adjustments = load_market_data()
    save(universe_table(market), "universe", "BIST100 members per index period")
    print("Past members without price data (delisted):", list(market.unavailable) or "none")
    save(survivorship(market), "survivorship", "Today's list vs point-in-time universe (medium horizon)")
    bt = Backtester(market)
    cpi = load_cpi()
    ew = bt.equal_weight_universe()
    test = bt.dates >= TEST_START
    print(f"Decision dates: {len(bt.dates)} ({bt.dates[0].date()} - {bt.dates[-1].date()}), "
          f"trades executed {bt.lag} day after the decision")

    # 1. Calibration (train 2021-2023, test 2024+)
    for h in Horizon:
        result = calibrate(bt, h)
        save(result.table, f"calibration_{h.value}",
             f"Calibration {h.value}: weights {[f'{p.value} {w:.2f}' for p, w in result.weights.items()]}")

    # 2. Every horizon x profile, full period and test period only
    logs, rows = {}, {}
    for h in Horizon:
        for p in PROFILE_NAMES:
            log = bt.decision_log(h, p)
            logs[(h, p)] = log
            monthly = bt.portfolio(log)
            hr, sig = log.hit_rates(), log.significance()
            rows[(h.value, p)] = {
                "BUY hit rate": hr["BUY hit rate"],
                "base rate": hr["Base rate (any stock beats index)"],
                "SELL hit rate": hr["SELL hit rate"],
                "BUY vs all": sig["BUY vs all (mean)"],
                "t (BUY vs all)": sig["BUY vs all (t)"],
                "BUY vs SELL": sig["BUY vs SELL (mean)"],
                "t (BUY vs SELL)": sig["BUY vs SELL (t)"],
                "CAGR": performance(monthly)["CAGR"],
                "CAGR vs equal-weight": performance(monthly, ew)["CAGR vs benchmark"],
                "real CAGR": performance(real_returns(monthly, cpi))["CAGR"],
                "test CAGR vs equal-weight": performance(monthly[test], ew[test])["CAGR vs benchmark"],
            }
    save(pd.DataFrame(rows).T, "combinations", "Horizon x profile (t > 2 = statistically significant)")

    bench = pd.DataFrame({
        "BIST100": performance(bt.index_month),
        "Equal-weight": performance(ew),
        "BIST100 real": performance(real_returns(bt.index_month, cpi)),
        "Equal-weight real": performance(real_returns(ew, cpi)),
    }).T[["Total return", "CAGR", "Max drawdown"]]
    save(bench, "benchmarks", "Benchmarks")

    # 3. Which metric predicts what? (rank IC, full period)
    save(pd.concat({h.value: bt.metric_ic(h) for h in Horizon}, axis=1), "metric_ic",
         "Rank IC of every metric with the later excess return (t = Newey-West)")

    # 4. Year by year
    yearly = {}
    for p in ("Balanced", "Data-driven"):
        monthly = bt.portfolio(logs[(Horizon.MEDIUM, p)])
        yearly[p] = monthly.groupby(monthly.index.year).apply(lambda r: (1 + r).prod() - 1)
    yearly["BIST100"] = bt.index_month.groupby(bt.index_month.index.year).apply(lambda r: (1 + r).prod() - 1)
    yearly["Equal-weight"] = ew.groupby(ew.index.year).apply(lambda r: (1 + r).prod() - 1)
    save(pd.DataFrame(yearly), "yearly", "Calendar years (medium horizon)")

    # 5. Score quintiles
    quint = pd.concat({f"{h.value}/{p}": logs[(h, p)].by_score_bucket()["mean"]
                       for h in Horizon for p in ("Balanced", "Data-driven")}, axis=1)
    save(quint, "score_quintiles", "Average excess return by score quintile")

    # 6. Risk flags
    main_log = logs[(Horizon.MEDIUM, "Balanced")]
    save(main_log.flag_effect(), "flags_buy", "BUY decisions with / without flags (medium, Balanced)")

    # 7. Data cleaning
    save(pd.DataFrame([a.__dict__ for a in adjustments]), "corporate_actions",
         "Corporate-action corrections")


if __name__ == "__main__":
    main()
