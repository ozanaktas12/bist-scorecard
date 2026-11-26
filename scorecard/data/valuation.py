"""P/E, P/B and dividend yield, shown next to the score but not part of it.

Ratios are computed from raw Yahoo fields because Yahoo's own ratios are
wrong for companies that report in USD. Checked against Fintables
(26 Nov 2026): P/B within 20 % for 95 of 100 stocks, P/E within 30 % for
57 of 74, so the cheap/expensive verdict uses P/B.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
FIELDS = ["financialCurrency", "currentPrice", "marketCap", "netIncomeToCommon",
          "bookValue", "dividendYield"]


def _fetch_one(ticker: str) -> dict:
    import yfinance as yf

    for _ in range(3):
        try:
            info = yf.Ticker(ticker + ".IS").info
            return {"ticker": ticker, **{f: info.get(f) for f in FIELDS}}
        except Exception as error:   # network hiccup: try again
            last = error
    log.warning("No valuation data for %s (%s)", ticker, last)
    return {"ticker": ticker}


def _fx_rates(currencies: set[str]) -> dict[str, float]:
    import yfinance as yf

    rates = {"TRY": 1.0}
    for cur in currencies - {"TRY"}:
        rates[cur] = float(yf.Ticker(f"{cur}TRY=X").history(period="5d")["Close"].iloc[-1])
    return rates


def compute_ratios(raw: pd.DataFrame, fx: dict[str, float]) -> pd.DataFrame:
    """P/E and P/B in lira terms; P/E is empty for loss-making companies."""
    rate = raw["financialCurrency"].map(fx).fillna(1.0)
    profit = raw["netIncomeToCommon"] * rate
    book_per_share = raw["bookValue"] * rate
    pe = (raw["marketCap"] / profit).where(profit > 0)
    pb = (raw["currentPrice"] / book_per_share).where(book_per_share > 0)
    # Values far outside what any BIST100 company has are data errors, not information.
    pe = pe.where(pe < 300)
    pb = pb.where(pb.between(0.05, 50))
    return pd.DataFrame({"pe": pe, "pb": pb, "dividend_yield": raw["dividendYield"] / 100,
                         "loss_making": profit <= 0}, index=raw.index)


def load_valuation(tickers: list[str], cache_file: str | Path, refresh: bool = False) -> pd.DataFrame:
    """Ratios of `tickers`, from the cached snapshot if there is one."""
    path = Path(cache_file)
    if path.exists() and not refresh:
        raw = pd.read_csv(path, index_col="ticker")
    else:
        with ThreadPoolExecutor(8) as pool:
            raw = pd.DataFrame(list(pool.map(_fetch_one, tickers))).set_index("ticker")
        raw["as_of"] = pd.Timestamp.today().normalize().date()
        fx = _fx_rates(set(raw["financialCurrency"].dropna()))
        for cur, value in fx.items():
            raw[f"fx_{cur}"] = value
        raw.to_csv(path)
    fx = {c.removeprefix("fx_"): raw[c].iloc[0] for c in raw.columns if c.startswith("fx_")}
    ratios = compute_ratios(raw, fx)
    ratios.attrs["as_of"] = str(raw["as_of"].iloc[0])
    return ratios.reindex(tickers)


def relative_to_sector(ratios: pd.DataFrame, sectors: pd.Series, min_peers: int = 3) -> pd.DataFrame:
    """Each ratio divided by its sector median (index median for small sectors)."""
    out = {}
    for col in ("pe", "pb"):
        values = ratios[col]
        counts = values.groupby(sectors).count()
        own_sector = sectors.map(counts).fillna(0) >= min_peers
        sector_median = sectors.map(values.groupby(sectors).median())
        peer = sector_median.where(own_sector, values.median())
        out[f"{col}_peer"] = peer
        out[f"{col}_vs_peer"] = values / peer
    out["peer_group"] = own_sector.map({True: "sector", False: "BIST100"})
    return pd.DataFrame(out, index=ratios.index)


def verdict(vs_peer: float) -> str:
    """Plain words for a ratio relative to its peers (lower = cheaper)."""
    if np.isnan(vs_peer):
        return "not comparable with"
    if vs_peer < 0.75:
        return "cheaper than"
    if vs_peer > 1.33:
        return "more expensive than"
    return "in line with"
