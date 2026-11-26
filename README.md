# BIST ScoreCard

An explainable scoring system for the 100 stocks of the BIST100 index: a 0-100
score, a BUY / HOLD / SELL label, risk flags, a plain-language reading of each
stock and its valuation next to its sector. Three investment horizons, four
risk profiles, and a backtest that checks the system's own past decisions.

SEN4015 Advanced Programming with Python — Ozan Aktaş, Demir Bigali, Arda Özköse

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app/streamlit_app.py                              # the dashboard
python -m scorecard ASELS --horizon medium --profile Balanced   # scorecard of one stock
python scripts/run_backtest.py                                  # every table in results/
pytest                                                          # run the tests
```

## Structure

| Path | Content |
|---|---|
| `scorecard/config.py` | Horizons, risk profiles, decision rules — every tunable number |
| `scorecard/data/` | Data providers (Yahoo Finance + CSV cache), corporate-action cleaning, point-in-time index membership, valuation ratios |
| `scorecard/indicators.py` | 15 technical indicators written with NumPy / pandas |
| `scorecard/scoring/` | 16 metrics → 5 pillars → score, decision rules, risk flags, scorecard, plain-language reading |
| `scorecard/backtest/` | Backtest of past decisions, significance tests, weight calibration |
| `app/` | Streamlit dashboard (`ui.py` holds colours, chart style and HTML parts) |
| `scripts/run_backtest.py` | Writes every result table of the report to `results/` |
| `tests/` | Unit tests, including the backtest arithmetic and a test that proves the system never uses future data |

## Data

* **Prices and valuation:** Yahoo Finance. A snapshot (prices until 25 Nov 2026, valuation
  of 26 Nov 2026) is stored in `data/cache/`, so the results in the report can be
  reproduced exactly; delete that folder to download fresh data.
* **Index membership:** `data/universe/bist100.csv` is the official list of the current
  quarter. `bist100_changes.csv` holds every quarterly BIST100 change since 2021 from
  Borsa Istanbul's own announcements (source link on every row); `ticker_renames.csv`
  maps old codes to today's. The backtest only scores the stocks that were in the index
  on each date, which avoids survivorship bias.
* **Sectors:** sector classification of all listed stocks, retrieved from Fintables on 26 Nov 2026 (`bist_all.csv`).
* **Inflation:** TCMB monthly CPI (`data/macro/`).

This project is for educational purposes only and is not investment advice.
