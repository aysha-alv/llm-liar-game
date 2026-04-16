"""
Phase 4 — Train CFR strategy and export as soft targets for Grok.

Usage:
    python scripts/phase4_cfr.py --iterations 50000
    python scripts/phase4_cfr.py --load data/cfr/strategy.json --show
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.cfr import LiarCFR
from game.card import Rank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: CFR strategy training")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--save", type=str, default="data/cfr/strategy.json")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Print top strategies")
    args = parser.parse_args()

    cfr = LiarCFR(n_players=args.players)

    if args.load:
        cfr.load(Path(args.load))
        print(f"Loaded CFR strategy: {len(cfr.nodes)} nodes, {cfr.iterations_run} iterations")
    else:
        print(f"Training CFR for {args.iterations} iterations...")
        cfr.train(n_iterations=args.iterations)

        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cfr.save(save_path)
        print(f"\n✓ CFR strategy saved → {save_path}")

    if args.show:
        cfr.print_key_strategies(top_n=15)

    # Example: look up strategy for a specific state
    print("\n--- Example Strategy Lookups ---")
    example_states = [
        (Rank.ACE, 8, 0, 5, 0, 0.1, "play"),
        (Rank.KING, 3, 2, 20, 3, 0.6, "play"),
        (Rank.JACK, 10, 4, 8, 2, 0.3, "challenge"),
    ]
    for rank, hs, nm, dp, tc, lf, phase in example_states:
        strat = cfr.get_strategy_for_state(rank, hs, nm, dp, tc, lf, phase)
        print(f"\n  {phase.upper()} | Rank={rank.label()} | hand={hs} | matching={nm} | pile={dp}")
        for action, prob in sorted(strat.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"    {action:<15} {prob:.3f}  {bar}")


if __name__ == "__main__":
    main()
