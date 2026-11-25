import pytest

from scorecard.config import PROFILES, Horizon, Pillar, RiskProfile
from scorecard.scoring import ScoreEngine


def test_profile_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        RiskProfile("Broken", {Pillar.TREND: 0.5, Pillar.RISK: 0.6})


def test_scores_are_between_0_and_100(market):
    engine = ScoreEngine(market, Horizon.MEDIUM)
    for profile in PROFILES:
        total = engine.total_score(profile).stack().dropna()
        assert len(total) > 0 and total.between(0, 100).all()


def test_scorecard_contains_all_metrics_and_pillars(market):
    card = ScoreEngine(market, Horizon.SHORT).scorecard("S05", "Balanced")
    assert len(card.metrics) == 16
    assert set(card.pillars) == set(Pillar)
    assert 1 <= card.rank <= card.universe_size
