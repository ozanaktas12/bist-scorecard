"""Command line scorecard:  python -m scorecard ASELS --horizon medium --profile Balanced"""
from __future__ import annotations

import argparse

from .config import PROFILE_NAMES, Horizon
from .data import load_market_data
from .scoring import ScoreEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="BIST ScoreCard for one stock")
    parser.add_argument("ticker")
    parser.add_argument("--horizon", choices=[h.value for h in Horizon], default="medium")
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="Balanced")
    args = parser.parse_args()

    market, _ = load_market_data()
    card = ScoreEngine(market, Horizon(args.horizon)).scorecard(args.ticker.upper(), args.profile)

    print(f"\n{card.ticker} ({card.sector}) - {card.date.date()} - {args.horizon} term, {args.profile}")
    print(f"Decision: {card.decision.value}   Score: {card.total:.1f}   "
          f"Rank: {card.rank}/{card.universe_size}\n")
    for pillar, score in card.pillars.items():
        print(f"  {pillar.value:18s} {score:5.1f}  {'#' * int(score / 5)}")
    print("\nStrengths:")
    for m in card.strengths():
        print(f"  + {m.explanation}")
    print("Weaknesses:")
    for m in card.weaknesses():
        print(f"  - {m.explanation}")
    if card.flags:
        print("\nFlags:")
        for f in card.flags:
            print(f"  [{f.level.value.upper()}] {f.title}: {f.message}")


if __name__ == "__main__":
    main()
