"""
Phase 3 — Self-play training with bluff-aware reward shaping.

Usage:
    python scripts/phase3_selfplay.py --iterations 10 --games 20
    python scripts/phase3_selfplay.py --curriculum   # Full 3-level curriculum
"""

import sys
import argparse
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_agent import GrokAgent
from training.self_play import SelfPlayTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Self-play training")
    parser.add_argument("--iterations", type=int, default=5, help="Training iterations")
    parser.add_argument("--games", type=int, default=10, help="Games per iteration")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--curriculum", action="store_true", help="Run full 3-level curriculum")
    parser.add_argument("--bluff-coeff", type=float, default=2.0, help="Bluff reward coefficient")
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

    try:
        grok = GrokAgent(name="Grok")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    trainer = SelfPlayTrainer(
        agent=grok,
        bluff_reward_coeff=args.bluff_coeff,
        output_dir=Path("data/self_play"),
    )

    if args.curriculum:
        print("Running full 3-level curriculum...")
        all_stats = trainer.run_curriculum(
            iterations_per_level=args.iterations,
            games_per_iteration=args.games,
            n_players=args.players,
        )
        print(f"\nCurriculum complete. {len(all_stats)} iterations total.")
    else:
        print(f"Running {args.iterations} self-play iterations ({args.games} games each)...")
        for i in range(args.iterations):
            stats = trainer.run_iteration(
                n_games=args.games,
                n_players=args.players,
                curriculum_level=1,
            )
            print(f"Iter {i+1}: win_rate={stats['win_rate']:.2%} | avg_reward={stats['avg_total_reward']:.3f}")

    print(f"\nTraining data saved to data/self_play/")
    print("Use these JSONL files for fine-tuning when xAI fine-tuning API is available.")


if __name__ == "__main__":
    main()
