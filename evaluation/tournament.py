"""
Phase 5 — Evaluation Tournament

Runs a comprehensive evaluation of all agents using:
  - Round-robin tournament format
  - TrueSkill rating updates after each game
  - Information-theoretic metric accumulation
  - Per-agent report generation
"""

from __future__ import annotations
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from game.liar_game import LiarGame
from agents.base_agent import BaseAgent
from agents import (
    GrokAgent, TheSaint, ComebackCloser, GameTheorist,
    MrPathological, TheAccountant, TheCollector, RandomAgent
)
from evaluation.metrics import InformationTheoreticMetrics, TrueSkillRating

logger = logging.getLogger(__name__)


class EvaluationTournament:
    """
    Runs a round-robin tournament and produces a full evaluation report.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("data/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.it_metrics = InformationTheoreticMetrics()
        self.trueskill = TrueSkillRating()

    def run(
        self,
        agents: List[BaseAgent],
        n_games: int = 100,
        n_players_per_game: int = 4,
    ) -> dict:
        """
        Run tournament and return full results dict.
        """
        logger.info(f"Starting evaluation tournament: {len(agents)} agents, {n_games} games")

        per_agent_metrics = defaultdict(lambda: {
            "wins": 0, "games": 0,
            "mi_scores": [], "kl_scores": [], "belief_misalign": [],
            "bluff_rates": [], "catch_rates": [], "challenge_accs": [],
        })

        for game_num in range(n_games):
            # Sample n_players_per_game agents
            selected = random.sample(agents, min(n_players_per_game, len(agents)))

            seed = random.randint(0, 999999)
            game = LiarGame(agents=selected, seed=seed, verbose=False)
            winner_id, final_state = game.play()

            # Rankings for TrueSkill (winner first, then by hand size)
            rankings = sorted(
                range(len(selected)),
                key=lambda i: (0 if i == winner_id else 1, final_state.players[i].hand_size)
            )
            ranked_names = [selected[i].name for i in rankings]
            self.trueskill.update(ranked_names)

            # Collect IT metrics per agent
            for idx, agent in enumerate(selected):
                analysis = self.it_metrics.analyze_game(final_state, idx)
                m = per_agent_metrics[agent.name]
                m["games"] += 1
                if idx == winner_id:
                    m["wins"] += 1
                m["mi_scores"].append(analysis["mutual_information"])
                m["kl_scores"].append(analysis["kl_divergence"])
                m["belief_misalign"].append(analysis["belief_misalignment"])
                m["bluff_rates"].append(analysis["bluff_rate"])
                m["catch_rates"].append(analysis["bluff_catch_rate"])
                m["challenge_accs"].append(analysis["challenge_accuracy"])

            if (game_num + 1) % 25 == 0:
                logger.info(f"Game {game_num + 1}/{n_games} complete")

        # Compile results
        results = {}
        for agent_name, m in per_agent_metrics.items():
            g = m["games"]

            def avg(lst):
                return sum(lst) / len(lst) if lst else 0.0

            results[agent_name] = {
                "win_rate": m["wins"] / max(g, 1),
                "games": g,
                "wins": m["wins"],
                "avg_mutual_information": avg(m["mi_scores"]),
                "avg_kl_divergence": avg(m["kl_scores"]),
                "avg_belief_misalignment": avg(m["belief_misalign"]),
                "avg_bluff_rate": avg(m["bluff_rates"]),
                "avg_catch_rate": avg(m["catch_rates"]),
                "avg_challenge_accuracy": avg(m["challenge_accs"]),
                "trueskill_rating": self.trueskill.players.get(agent_name, None) and
                                    self.trueskill.players[agent_name].rating,
            }

        self._print_report(results)
        self._save_report(results)
        self.trueskill.print_leaderboard()

        return results

    def _print_report(self, results: dict) -> None:
        print("\n" + "=" * 80)
        print("PHASE 5 EVALUATION REPORT")
        print("=" * 80)
        header = f"{'Agent':<20} {'WR%':>6} {'MI↓':>7} {'KL↑':>7} {'BRate':>7} {'CatchR':>7} {'ChalAcc':>8}"
        print(header)
        print("-" * 80)

        for name, r in sorted(results.items(), key=lambda x: -x[1]["win_rate"]):
            marker = " ◄" if "Grok" in name else ""
            print(
                f"{name:<20} "
                f"{r['win_rate']*100:>5.1f}% "
                f"{r['avg_mutual_information']:>7.3f} "
                f"{r['avg_kl_divergence']:>7.3f} "
                f"{r['avg_bluff_rate']:>7.2f} "
                f"{r['avg_catch_rate']:>7.2f} "
                f"{r['avg_challenge_accuracy']:>8.2f}"
                f"{marker}"
            )

        print("\nMetric guide:")
        print("  MI↓  = Mutual Information (lower = better bluffer — actions don't reveal hand)")
        print("  KL↑  = KL Divergence (higher = more effectively deceived opponents)")
        print("  BRate = Bluff rate per turn")
        print("  CatchR = Fraction of bluffs caught (lower = better)")
        print("  ChalAcc = Challenge accuracy (won challenges / total challenges)")
        print("=" * 80)

    def _save_report(self, results: dict) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"phase5_evaluation_{ts}.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation report saved → {out}")
