"""
Phase 4 — Counterfactual Regret Minimization (CFR)

Implements External-Sampling Monte Carlo CFR for the card game Liar.

CFR finds a Nash equilibrium strategy by iteratively minimizing
"regret" — the amount a player wishes they had taken a different action.

The strategy σ_i^T(I, a) for player i at information set I for action a
is updated using regret matching:

    σ_i^{T+1}(I, a) = R_i^T(I, a)^+ / Σ_{a'} R_i^T(I, a')^+
                     = 1/|A(I)|  if all cumulative regrets ≤ 0

State abstraction (required for tractability):
  - Information set key: (current_rank, hand_count, discard_pile_size_bucket, n_cards_of_rank)
  - Buckets discard pile into: [0-5], [6-15], [16+]
  - This reduces the state space dramatically

The CFR-derived strategy can be used as:
  1. Soft targets for Grok's output distribution
  2. Synthetic training data (high-quality action probabilities)
  3. Direct strategy lookup (tabular CFR agent)
"""

from __future__ import annotations
import json
import random
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from game.card import Rank, Card, Deck
from game.game_state import GameState, PlayerState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Information Set abstraction
# ---------------------------------------------------------------------------

def _pile_bucket(size: int) -> str:
    if size <= 5:
        return "small"
    elif size <= 15:
        return "medium"
    return "large"


def _hand_bucket(size: int) -> str:
    if size <= 3:
        return "tiny"
    elif size <= 8:
        return "small"
    elif size <= 15:
        return "medium"
    return "large"


def get_info_set_key(
    current_rank: Rank,
    hand_size: int,
    n_matching: int,
    discard_pile_size: int,
    total_claimed_this_rank: int,
    phase: str,  # "play" or "challenge"
    lie_freq_bucket: str = "low",  # "low", "medium", "high"
) -> str:
    """
    Compact information set key for state abstraction.
    Dramatically reduces state space while preserving strategic structure.
    """
    return (
        f"{phase}|{current_rank.label()}|h{_hand_bucket(hand_size)}|"
        f"m{min(n_matching, 4)}|p{_pile_bucket(discard_pile_size)}|"
        f"claimed{min(total_claimed_this_rank, 4)}|lf{lie_freq_bucket}"
    )


def lie_freq_bucket(freq: float) -> str:
    if freq < 0.25:
        return "low"
    elif freq < 0.55:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

PLAY_ACTIONS = [
    "play_honest",       # Play matching cards truthfully
    "bluff_1",           # Bluff 1 card
    "bluff_2",           # Bluff 2 cards
    "bluff_3",           # Bluff 3 cards
    "overclaim",         # Play matching but claim more
]

CHALLENGE_ACTIONS = ["challenge", "pass"]


# ---------------------------------------------------------------------------
# CFR Node
# ---------------------------------------------------------------------------

@dataclass
class CFRNode:
    """
    Stores cumulative regrets and strategy sum for one information set.
    """
    actions: List[str]
    regret_sum: Dict[str, float] = field(default_factory=dict)
    strategy_sum: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for a in self.actions:
            self.regret_sum.setdefault(a, 0.0)
            self.strategy_sum.setdefault(a, 0.0)

    def get_strategy(self, reach_prob: float = 1.0) -> Dict[str, float]:
        """Current strategy via regret matching."""
        positive_regrets = {a: max(0.0, self.regret_sum[a]) for a in self.actions}
        total = sum(positive_regrets.values())
        if total > 0:
            strategy = {a: positive_regrets[a] / total for a in self.actions}
        else:
            strategy = {a: 1.0 / len(self.actions) for a in self.actions}

        # Accumulate strategy sum (for average strategy computation)
        for a in self.actions:
            self.strategy_sum[a] += reach_prob * strategy[a]

        return strategy

    def get_average_strategy(self) -> Dict[str, float]:
        """Average strategy over all iterations (converges to Nash equilibrium)."""
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {a: self.strategy_sum[a] / total for a in self.actions}
        return {a: 1.0 / len(self.actions) for a in self.actions}

    def update_regret(self, action: str, regret: float) -> None:
        self.regret_sum[action] += regret


# ---------------------------------------------------------------------------
# CFR Solver
# ---------------------------------------------------------------------------

class LiarCFR:
    """
    Monte Carlo CFR solver for the Liar card game.

    Runs iterations of external-sampling MCCFR to approximate
    Nash equilibrium strategies. Trained strategies are saved as
    lookup tables that the LLM agent can consult.
    """

    def __init__(self, n_players: int = 4):
        self.n_players = n_players
        self.nodes: Dict[str, CFRNode] = {}
        self.iterations_run = 0

    def get_or_create_node(self, info_set_key: str, actions: List[str]) -> CFRNode:
        if info_set_key not in self.nodes:
            self.nodes[info_set_key] = CFRNode(actions=actions)
        return self.nodes[info_set_key]

    def train(self, n_iterations: int = 10000) -> None:
        """
        Run n_iterations of MCCFR.
        Each iteration simulates a game and updates regrets.
        """
        logger.info(f"Starting CFR training: {n_iterations} iterations, {self.n_players} players")

        for i in range(n_iterations):
            self._run_cfr_iteration()
            self.iterations_run += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"  CFR iteration {i + 1}/{n_iterations} | nodes: {len(self.nodes)}")

        logger.info(f"CFR training complete. {len(self.nodes)} information sets learned.")

    def _run_cfr_iteration(self) -> None:
        """Single MCCFR iteration via external sampling."""
        # Sample a random game state
        rank = random.choice(list(Rank))
        hand_size = random.randint(1, 13)
        n_matching = random.randint(0, min(4, hand_size))
        discard_size = random.randint(0, 30)
        total_claimed = random.randint(0, 4)
        lf = random.choice(["low", "medium", "high"])

        # --- PLAY node update ---
        play_key = get_info_set_key(
            rank, hand_size, n_matching, discard_size, total_claimed, "play", lf
        )
        play_node = self.get_or_create_node(play_key, PLAY_ACTIONS)
        self._update_play_node(play_node, n_matching, hand_size, discard_size)

        # --- CHALLENGE node update ---
        challenge_key = get_info_set_key(
            rank, hand_size, n_matching, discard_size, total_claimed, "challenge", lf
        )
        chal_node = self.get_or_create_node(challenge_key, CHALLENGE_ACTIONS)
        self._update_challenge_node(chal_node, total_claimed, discard_size, lf)

    def _update_play_node(
        self,
        node: CFRNode,
        n_matching: int,
        hand_size: int,
        discard_size: int,
    ) -> None:
        """Compute counterfactual regrets for play actions."""
        strategy = node.get_strategy()
        action_values = self._simulate_play_values(n_matching, hand_size, discard_size)
        ev = sum(strategy[a] * action_values.get(a, 0.0) for a in node.actions)

        for action in node.actions:
            regret = action_values.get(action, 0.0) - ev
            node.update_regret(action, regret)

    def _update_challenge_node(
        self,
        node: CFRNode,
        total_claimed: int,
        discard_size: int,
        lf: str,
    ) -> None:
        """Compute counterfactual regrets for challenge actions."""
        strategy = node.get_strategy()

        # Estimate lie probability from total_claimed and lie frequency
        lie_prob_base = {"low": 0.15, "medium": 0.40, "high": 0.70}[lf]
        if total_claimed >= 4:
            lie_prob = min(lie_prob_base + 0.3, 0.99)
        else:
            lie_prob = lie_prob_base

        challenge_ev = lie_prob * discard_size - (1 - lie_prob) * discard_size
        pass_ev = 0.0

        action_values = {
            "challenge": challenge_ev,
            "pass": pass_ev,
        }
        ev = sum(strategy[a] * action_values[a] for a in node.actions)
        for action in node.actions:
            node.update_regret(action, action_values[action] - ev)

    def _simulate_play_values(
        self,
        n_matching: int,
        hand_size: int,
        discard_size: int,
    ) -> Dict[str, float]:
        """
        Estimate the value of each play action via simple simulation.
        Returns expected card reduction (positive = good for agent).
        """
        values = {}

        # Honest play: shed n_matching cards
        if n_matching > 0:
            values["play_honest"] = n_matching / max(hand_size, 1) * 10
        else:
            values["play_honest"] = -1.0  # Forced to bluff anyway

        # Bluff N: shed N cards but risk picking up pile
        for n_bluff in [1, 2, 3]:
            if hand_size >= n_bluff:
                detect_prob = 0.25 + (n_bluff * 0.1)
                shed_value = n_bluff / max(hand_size, 1) * 10
                catch_cost = -discard_size / max(hand_size, 1) * 8
                values[f"bluff_{n_bluff}"] = (1 - detect_prob) * shed_value + detect_prob * catch_cost
            else:
                values[f"bluff_{n_bluff}"] = -99  # Can't bluff more than hand size

        # Overclaim: play matching but claim more to deceive
        if n_matching > 0:
            values["overclaim"] = values["play_honest"] * 0.9  # Slight risk
        else:
            values["overclaim"] = -1.0

        return values

    def get_strategy_for_state(
        self,
        current_rank: Rank,
        hand_size: int,
        n_matching: int,
        discard_pile_size: int,
        total_claimed_this_rank: int,
        lf: float,
        phase: str,
    ) -> Dict[str, float]:
        """
        Look up the CFR-trained average strategy for a game state.
        Returns action probabilities.
        """
        lf_b = lie_freq_bucket(lf)
        key = get_info_set_key(
            current_rank, hand_size, n_matching, discard_pile_size,
            total_claimed_this_rank, phase, lf_b
        )
        actions = PLAY_ACTIONS if phase == "play" else CHALLENGE_ACTIONS

        if key in self.nodes:
            return self.nodes[key].get_average_strategy()
        return {a: 1.0 / len(actions) for a in actions}

    def save(self, path: Path) -> None:
        """Save trained CFR nodes to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "iterations": self.iterations_run,
            "n_players": self.n_players,
            "nodes": {
                key: {
                    "actions": node.actions,
                    "regret_sum": node.regret_sum,
                    "strategy_sum": node.strategy_sum,
                    "avg_strategy": node.get_average_strategy(),
                }
                for key, node in self.nodes.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"CFR strategy saved → {path} ({len(self.nodes)} nodes)")

    def load(self, path: Path) -> None:
        """Load previously trained CFR nodes."""
        with open(path) as f:
            data = json.load(f)
        self.iterations_run = data["iterations"]
        self.n_players = data["n_players"]
        self.nodes = {}
        for key, node_data in data["nodes"].items():
            node = CFRNode(actions=node_data["actions"])
            node.regret_sum = node_data["regret_sum"]
            node.strategy_sum = node_data["strategy_sum"]
            self.nodes[key] = node
        logger.info(f"CFR strategy loaded from {path} ({len(self.nodes)} nodes, {self.iterations_run} iters)")

    def print_key_strategies(self, top_n: int = 10) -> None:
        """Print the most-visited information sets and their strategies."""
        print("\n=== CFR Average Strategies (Top Information Sets) ===")
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: sum(x[1].strategy_sum.values()),
            reverse=True
        )
        for key, node in sorted_nodes[:top_n]:
            avg = node.get_average_strategy()
            best_action = max(avg, key=avg.get)
            print(f"\n  {key}")
            for a, p in sorted(avg.items(), key=lambda x: -x[1]):
                bar = "█" * int(p * 20)
                print(f"    {a:<15} {p:.3f} {bar}")
