"""Full backtest of BIST ScoreCard; writes tables to results/ for the report.

    python scripts/run_backtest.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scorecard.backtest import Backtester, load_cpi, performance, real_returns  # noqa: E402
from scorecard.config import PROFILES, SETTINGS, Horizon  # noqa: E402
from scorecard.data import load_market_data  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200)


def save(table: pd.DataFrame, name: str, title: str) -> None:
    table.to_csv(OUT / f"{name}.csv")
    print(f"\n=== {title} ===")
    print(table.round(3).to_string())  # noqa: T201


def main() -> None:
    market, adjustments = load_market_data()
    bt = Backtester(market)
    cpi = load_cpi()
    ew = bt.equal_weight_universe()
    print(f"Decision dates: {len(bt.dates)} ({bt.dates[0].date()} - {bt.dates[-1].date()})")

    # 1. All horizon x profile combinations
    rows = {}
    logs = {}
    for h in Horizon:
        for p in PROFILES:
            log = bt.decision_log(h, p)
            logs[(h, p)] = log
            monthly = bt.portfolio(log)
            hr = log.hit_rates()
            perf = performance(monthly, bt.index_month)
            real = performance(real_returns(monthly, cpi))
            rows[(h.value, p)] = {
                "BUY decisions": hr["BUY decisions"],
                "BUY hit rate": hr["BUY hit rate"],
                "SELL hit rate": hr["SELL hit rate"],
                "avg excess BUY": hr["Avg excess BUY"],
                "avg excess HOLD": hr["Avg excess HOLD"],
                "avg excess SELL": hr["Avg excess SELL"],
                "CAGR": perf["CAGR"],
                "CAGR vs equal-weight": perf["CAGR"] - performance(ew)["CAGR"],
                "real CAGR": real["CAGR"],
                "max drawdown": perf["Max drawdown"],
            }
    save(pd.DataFrame(rows).T, "combinations", "Horizon x profile")

    bench = pd.DataFrame({
        "BIST100": performance(bt.index_month),
        "Equal-weight": performance(ew),
        "BIST100 real": performance(real_returns(bt.index_month, cpi)),
        "Equal-weight real": performance(real_returns(ew, cpi)),
    }).T[["Total return", "CAGR", "Max drawdown"]]
    save(bench, "benchmarks", "Benchmarks")

    main_log = logs[(Horizon.MEDIUM, "Balanced")]
    main_monthly = bt.portfolio(main_log)

    # 2. Random selection: same number of stocks, picked by chance
    random_runs = bt.random_portfolios(main_log, n=300)
    random_cagr = random_runs.apply(lambda r: performance(r)["CAGR"])
    strategy_cagr = performance(main_monthly)["CAGR"]
    save(pd.DataFrame({"value": {
        "strategy CAGR": strategy_cagr,
        "random median CAGR": random_cagr.median(),
        "random 5th pct": random_cagr.quantile(0.05),
        "random 95th pct": random_cagr.quantile(0.95),
        "share of random runs beaten": (random_cagr < strategy_cagr).mean(),
    }}), "random_benchmark", "Medium/Balanced vs 300 random portfolios")

    # 3. Year by year (does it work in every market regime?)
    yearly = pd.DataFrame({
        "ScoreCard": main_monthly, "BIST100": bt.index_month, "Equal-weight": ew,
    }).groupby(main_monthly.index.year).apply(lambda d: (1 + d).prod() - 1)
    yearly["vs equal-weight"] = yearly["ScoreCard"] - yearly["Equal-weight"]
    save(yearly, "yearly", "Calendar years (medium / Balanced)")

    # 4. Score quintiles per horizon
    quint = pd.DataFrame({h.value: logs[(h, "Balanced")].by_score_bucket()["mean"] for h in Horizon})
    save(quint, "score_quintiles", "Average excess return by score quintile (Balanced)")

    # 5. Sensitivity of the BUY threshold (top 10 / 20 / 30 %)
    sens = {}
    for frac in (0.10, 0.20, 0.30):
        rules = replace(SETTINGS.rules, top_fraction=frac, bottom_fraction=frac)
        log = bt.decision_log(Horizon.MEDIUM, "Balanced", rules=rules)
        sens[f"top {frac:.0%}"] = {
            "BUY decisions": log.hit_rates()["BUY decisions"],
            "BUY hit rate": log.hit_rates()["BUY hit rate"],
            "avg excess BUY": log.hit_rates()["Avg excess BUY"],
            "CAGR": performance(bt.portfolio(log))["CAGR"],
        }
    save(pd.DataFrame(sens).T, "sensitivity_top_fraction", "Sensitivity of the top-fraction rule")

    # 6. Risk flags
    save(main_log.flag_effect(), "flags_buy", "BUY decisions with / without flags")
    save(main_log.flag_rules_effect(bt.flags.panels), "flags_rules", "Every flag rule, all decisions")

    # 7. Data cleaning
    save(pd.DataFrame([a.__dict__ for a in adjustments]), "corporate_actions",
         "Corporate-action corrections")


if __name__ == "__main__":
    main()
