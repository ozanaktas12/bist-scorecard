# BIST ScoreCard

An explainable BUY / HOLD / SELL scoring system for the 100 stocks of the BIST100
index, with three investment horizons, three risk profiles, risk flags and a
backtest that checks the system's own past decisions.

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
| `scorecard/data/` | Data providers (Yahoo Finance + CSV cache) and corporate-action cleaning |
| `scorecard/indicators.py` | 15 technical indicators written with NumPy / pandas |
| `scorecard/scoring/` | 16 metrics → 5 pillars → score, decision rules, risk flags, scorecard |
| `scorecard/backtest/` | Backtest of past decisions, significance tests, weight calibration |
| `app/` | Streamlit dashboard (`ui.py` holds colours, chart style and HTML parts) |
| `scripts/run_backtest.py` | Writes every result table of the report to `results/` |
| `tests/` | Unit tests, including the backtest arithmetic and a test that proves the system never uses future data |

Data: Yahoo Finance (prices), Fintables (BIST100 constituents and sectors, 25 Nov 2026),
TCMB (monthly CPI). A snapshot of the downloaded prices (until 25 Nov 2026) is stored in
`data/cache/`, so the results in the report can be reproduced exactly; delete that folder
to download fresh prices.
This project is for educational purposes only and is not investment advice.
