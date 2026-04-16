"""
Phase 5 — Full evaluation tournament with all metrics.

Usage:
    python scripts/phase5_evaluate.py --games 100
    python scripts/phase5_evaluate.py --games 50 --players 5
"""

import sys
import argparse
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    GrokAgent, TheSaint, ComebackCloser, GameTheorist,
    MrPathological, TheAccountant, TheCollector, RandomAgent
)
from evaluation.tournament import EvaluationTournament

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Full evaluation")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--no-grok", action="store_true", help="Skip Grok (archetypes only)")
    args = parser.parse_args()

    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    agents = [
        TheSaint(name="TheSaint"),
        ComebackCloser(name="ComebackCloser"),
        GameTheorist(name="GameTheorist"),
        MrPathological(name="MrPathological"),
        TheAccountant(name="TheAccountant"),
        TheCollector(name="TheCollector"),
        RandomAgent(name="Random"),
    ]

    if not args.no_grok:
        try:
            grok = GrokAgent(name="Grok")
            agents.append(grok)
        except ValueError as e:
            print(f"Warning: Could not load GrokAgent: {e}")
            print("Running evaluation without Grok. Set XAI_API_KEY to include it.")

    tournament = EvaluationTournament(output_dir=Path("data/results"))
    results = tournament.run(
        agents=agents,
        n_games=args.games,
        n_players_per_game=min(args.players, len(agents)),
    )


if __name__ == "__main__":
    main()
