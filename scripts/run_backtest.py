"""Full backtest of BIST ScoreCard; writes every table to results/ for the report.

    python scripts/run_backtest.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scorecard.backtest import Backtester, calibrate, load_cpi, performance, real_returns  # noqa: E402
from scorecard.backtest.calibration import TEST_START  # noqa: E402
from scorecard.config import PROFILE_NAMES, SETTINGS, Horizon  # noqa: E402
from scorecard.data import load_market_data  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220)


def save(table: pd.DataFrame, name: str, title: str) -> None:
    table.to_csv(OUT / f"{name}.csv")
    print(f"\n=== {title} ===")
    print(table.round(3).to_string())


def main() -> None:
    market, adjustments = load_market_data()
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

    # 4. Random selection: same number of stocks, picked by chance
    random_rows = {}
    for p in ("Balanced", "Data-driven"):
        log = logs[(Horizon.MEDIUM, p)]
        runs = bt.random_portfolios(log, n=300)
        random_cagr = runs.apply(lambda r: performance(r)["CAGR"])
        cagr = performance(bt.portfolio(log))["CAGR"]
        random_rows[p] = {"strategy CAGR": cagr, "random median": random_cagr.median(),
                          "random 95th pct": random_cagr.quantile(0.95),
                          "share of random runs beaten": (random_cagr < cagr).mean()}
    save(pd.DataFrame(random_rows).T, "random_benchmark", "Medium horizon vs 300 random portfolios")

    # 5. Year by year
    yearly = {}
    for p in ("Balanced", "Data-driven"):
        monthly = bt.portfolio(logs[(Horizon.MEDIUM, p)])
        yearly[p] = monthly.groupby(monthly.index.year).apply(lambda r: (1 + r).prod() - 1)
    yearly["BIST100"] = bt.index_month.groupby(bt.index_month.index.year).apply(lambda r: (1 + r).prod() - 1)
    yearly["Equal-weight"] = ew.groupby(ew.index.year).apply(lambda r: (1 + r).prod() - 1)
    save(pd.DataFrame(yearly), "yearly", "Calendar years (medium horizon)")

    # 6. Score quintiles
    quint = pd.concat({f"{h.value}/{p}": logs[(h, p)].by_score_bucket()["mean"]
                       for h in Horizon for p in ("Balanced", "Data-driven")}, axis=1)
    save(quint, "score_quintiles", "Average excess return by score quintile")

    # 7. Sensitivity of the top-fraction rule
    sens = {}
    for frac in (0.10, 0.20, 0.30):
        rules = replace(SETTINGS.rules, top_fraction=frac, bottom_fraction=frac)
        for p in ("Balanced", "Data-driven"):
            log = bt.decision_log(Horizon.MEDIUM, p, rules=rules)
            sens[(f"top {frac:.0%}", p)] = {
                "BUY decisions": log.hit_rates()["BUY decisions"],
                "BUY vs all": log.significance()["BUY vs all (mean)"],
                "CAGR": performance(bt.portfolio(log))["CAGR"],
            }
    save(pd.DataFrame(sens).T, "sensitivity_top_fraction", "Sensitivity of the top-fraction rule (medium)")

    # 8. Risk flags
    main_log = logs[(Horizon.MEDIUM, "Balanced")]
    save(main_log.flag_effect(), "flags_buy", "BUY decisions with / without flags (medium, Balanced)")
    save(main_log.flag_rules_effect(bt.flags.panels), "flags_rules", "Every flag rule, all decisions")

    # 9. Data cleaning
    save(pd.DataFrame([a.__dict__ for a in adjustments]), "corporate_actions",
         "Corporate-action corrections")


if __name__ == "__main__":
    main()
