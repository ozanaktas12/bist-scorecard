"""The most important property of the backtest: a decision made on day t must
not change when everything after day t is deleted from the data."""
import numpy as np
import pytest

from scorecard.config import Horizon
from scorecard.scoring import FlagEngine, ScoreEngine


@pytest.mark.parametrize("horizon", list(Horizon))
def test_scores_do_not_use_future_data(market, horizon):
    full = ScoreEngine(market, horizon)
    for cut in (650, 800):
        date = market.dates[cut]
        truncated = ScoreEngine(market.until(date), horizon)
        for key, panel in full.features.raw.items():
            np.testing.assert_allclose(panel.loc[date], truncated.features.raw[key].loc[date],
                                       rtol=1e-9, err_msg=f"{key} looks ahead")
        np.testing.assert_allclose(full.total_score("Balanced").loc[date],
                                   truncated.total_score("Balanced").loc[date], rtol=1e-9)


def test_flags_do_not_use_future_data(market):
    date = market.dates[700]
    full, truncated = FlagEngine(market), FlagEngine(market.until(date))
    for key in full.panels:
        assert full.panels[key].loc[date].equals(truncated.panels[key].loc[date]), key
