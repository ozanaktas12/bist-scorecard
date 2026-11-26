"""Valuation ratios and the plain-language reading of a scorecard."""
import pandas as pd
import pytest

from scorecard.config import Horizon
from scorecard.data.valuation import compute_ratios, relative_to_sector, verdict
from scorecard.scoring import ScoreEngine
from scorecard.scoring.narrative import describe


def test_dollar_reporters_are_converted_to_lira():
    raw = pd.DataFrame({
        "financialCurrency": ["TRY", "USD"],
        "currentPrice": [100.0, 300.0],
        "marketCap": [1_000.0, 3_000.0],
        "netIncomeToCommon": [100.0, 2.5],       # USD profit: 2.5 x 40 = 100 TRY
        "bookValue": [50.0, 15.0],               # USD book per share: 15 x 40 = 600 TRY
        "dividendYield": [2.0, 0.0],
    }, index=["LIRA", "DOLLAR"])
    ratios = compute_ratios(raw, {"TRY": 1.0, "USD": 40.0})
    assert ratios.at["LIRA", "pe"] == pytest.approx(10.0)
    assert ratios.at["DOLLAR", "pe"] == pytest.approx(30.0)
    assert ratios.at["DOLLAR", "pb"] == pytest.approx(0.5)
    assert ratios.at["LIRA", "dividend_yield"] == pytest.approx(0.02)


def test_losses_and_impossible_values_give_no_ratio():
    raw = pd.DataFrame({"financialCurrency": ["TRY", "TRY"], "currentPrice": [10.0, 10.0],
                        "marketCap": [100.0, 100.0], "netIncomeToCommon": [-5.0, 0.01],
                        "bookValue": [5.0, 1000.0], "dividendYield": [0.0, 0.0]}, index=["L", "X"])
    ratios = compute_ratios(raw, {"TRY": 1.0})
    assert ratios.at["L", "loss_making"] and pd.isna(ratios.at["L", "pe"])
    assert pd.isna(ratios.at["X", "pe"]) and pd.isna(ratios.at["X", "pb"])   # P/E 10000, P/B 0.01


def test_peer_comparison_and_verdict():
    ratios = pd.DataFrame({"pe": [5.0, 10, 20, 8], "pb": [0.5, 1.0, 2.0, 1.0]}, index=list("ABCD"))
    sectors = pd.Series(["Bank", "Bank", "Bank", "Sport"], index=list("ABCD"))
    rel = relative_to_sector(ratios, sectors)
    assert rel.at["A", "pb_vs_peer"] == pytest.approx(0.5)     # bank median 1.0
    assert rel.at["D", "pb_peer"] == pytest.approx(1.0)        # lone stock: all-stock median
    assert list(rel.peer_group) == ["sector", "sector", "sector", "BIST100"]
    assert verdict(0.5) == "cheaper than"
    assert verdict(1.0) == "in line with"
    assert verdict(2.0) == "more expensive than"


def test_every_scorecard_gets_a_reading(market):
    engine = ScoreEngine(market, Horizon.MEDIUM)
    for ticker in market.tickers:
        card = engine.scorecard(ticker, "Balanced")
        sentences = describe(card, {"loss_making": True})
        assert any("trend" in s for s in sentences)
        assert sentences[-1].startswith(("The company made a loss", "Warning"))
