"""
Phase 5 — Information-Theoretic Evaluation Metrics

From the paper, the key evaluation metrics are:

1. Mutual Information I(H; A)
   Measures how much an agent's actions reveal about their actual hand.
   Lower = better bluffer (actions don't leak hand contents).
   I(H; A) = H(A) - H(A|H)

2. KL Divergence D_KL(P_true || P_believed)
   Measures how far the opponent's posterior beliefs have shifted
   from the true hand distribution (attacker's perspective).
   Higher = more effective deception.

3. Belief Misalignment
   Average absolute error between what opponents believe
   and what is actually in the agent's hand.

4. TrueSkill Rating
   Microsoft's Bayesian skill rating system (like Elo but for multiplayer).
   Tracks agent improvement over time across diverse opponents.
"""

from __future__ import annotations
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from game.card import Rank
from game.game_state import GameState, ClaimRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Mutual Information I(H; A)
# ---------------------------------------------------------------------------

class InformationTheoreticMetrics:
    """
    Computes information-theoretic metrics from a completed game's history.
    """

    def compute_mutual_information(
        self,
        claims: List[ClaimRecord],
    ) -> float:
        """
        Approximate I(H; A) from claim records.

        H = actual hand composition (measured by whether claim was honest/bluff)
        A = claimed count (the observable action)

        Higher I(H;A) = actions are more correlated with hand contents = worse bluffer
        Lower I(H;A) = actions are decoupled from hand = better bluffer
        """
        if not claims:
            return 0.0

        # Joint distribution P(honest, claimed_count)
        joint: Dict[Tuple[bool, int], int] = defaultdict(int)
        for c in claims:
            joint[(not c.was_lying, min(c.claimed_count, 4))] += 1

        n = len(claims)
        joint_probs = {k: v / n for k, v in joint.items()}

        # Marginals
        honest_counts: Dict[bool, float] = defaultdict(float)
        action_counts: Dict[int, float] = defaultdict(float)
        for (h, a), p in joint_probs.items():
            honest_counts[h] += p
            action_counts[a] += p

        def entropy(dist: Dict) -> float:
            return -sum(p * math.log2(p + 1e-10) for p in dist.values() if p > 0)

        h_a = entropy(action_counts)           # H(A)
        # H(A|H) = Σ_h P(h) * H(A|H=h)
        h_a_given_h = 0.0
        for h_val in [True, False]:
            ph = honest_counts.get(h_val, 0)
            if ph == 0:
                continue
            cond_dist: Dict[int, float] = defaultdict(float)
            for (h, a), p in joint_probs.items():
                if h == h_val:
                    cond_dist[a] += p / ph
            h_a_given_h += ph * entropy(cond_dist)

        mi = max(0.0, h_a - h_a_given_h)
        return mi

    def compute_kl_divergence(
        self,
        true_hand_counts: Dict[str, int],    # rank_label -> actual count in hand
        believed_hand_counts: Dict[str, int],  # rank_label -> opponent belief
        total_cards: int = 13,
    ) -> float:
        """
        D_KL(P_true || P_believed)

        Measures how far opponent beliefs are from reality.
        Higher KL = more successfully deceived opponents.

        P_true = true hand distribution over ranks
        P_believed = opponent's posterior belief about hand distribution
        """
        all_ranks = [r.label() for r in Rank]

        # Normalize to distributions
        true_total = sum(true_hand_counts.values()) or 1
        believed_total = sum(believed_hand_counts.values()) or 1

        kl = 0.0
        for rank in all_ranks:
            p_true = true_hand_counts.get(rank, 0) / true_total
            p_believed = believed_hand_counts.get(rank, 0.01) / believed_total  # Laplace smoothing

            if p_true > 0:
                kl += p_true * math.log2(p_true / p_believed)

        return max(0.0, kl)

    def compute_belief_misalignment(
        self,
        claims: List[ClaimRecord],
        player_id: int,
    ) -> float:
        """
        Measures how well an agent hid their hand by comparing:
        - What they claimed to have (cumulative)
        - What they actually played (cumulative)

        Higher misalignment = better at decoupling actions from reality.
        """
        if not claims:
            return 0.0

        player_claims = [c for c in claims if c.player_id == player_id]
        if not player_claims:
            return 0.0

        total_misalignment = 0.0
        for claim in player_claims:
            actual_matching = sum(1 for c in claim.actual_cards if c.rank == claim.claimed_rank)
            misalignment = abs(claim.claimed_count - actual_matching)
            total_misalignment += misalignment

        return total_misalignment / len(player_claims)

    def analyze_game(self, state: GameState, agent_id: int) -> dict:
        """Full information-theoretic analysis of one game for one agent."""
        agent_claims = [c for c in state.claim_history if c.player_id == agent_id]

        mi = self.compute_mutual_information(agent_claims)
        belief_misalign = self.compute_belief_misalignment(state.claim_history, agent_id)

        # Estimate believed hand from claim history (simplified)
        believed = defaultdict(int)
        for claim in agent_claims:
            believed[claim.claimed_rank.label()] += claim.claimed_count

        true_hand = {r.label(): state.players[agent_id].count_rank(r) for r in Rank}
        kl = self.compute_kl_divergence(true_hand, dict(believed))

        agent = state.players[agent_id]
        bluff_rate = agent.bluffs_attempted / max(agent.turns_played, 1)
        catch_rate = agent.bluffs_caught / max(agent.bluffs_attempted, 1)

        return {
            "mutual_information": mi,
            "kl_divergence": kl,
            "belief_misalignment": belief_misalign,
            "bluff_rate": bluff_rate,
            "bluff_catch_rate": catch_rate,
            "challenge_accuracy": agent.challenges_won / max(agent.challenges_issued, 1),
            "win": state.winner_id == agent_id,
        }


# ---------------------------------------------------------------------------
# 4. TrueSkill Rating
# ---------------------------------------------------------------------------

@dataclass
class TrueSkillPlayer:
    """
    Simplified TrueSkill rating.
    Each player has a mean (mu) and standard deviation (sigma).
    Rating = mu - 3*sigma (conservative estimate)
    """
    name: str
    mu: float = 25.0          # Initial mean skill
    sigma: float = 25.0 / 3   # Initial uncertainty
    games_played: int = 0

    @property
    def rating(self) -> float:
        """Conservative skill estimate (TrueSkill convention)."""
        return self.mu - 3 * self.sigma

    @property
    def display(self) -> str:
        return f"μ={self.mu:.1f} σ={self.sigma:.1f} → {self.rating:.1f}"


class TrueSkillRating:
    """
    Simplified TrueSkill implementation for multiplayer ranking.

    Uses a Gaussian approximation of the factor graph update.
    For production use, install the `trueskill` package.
    """

    BETA = 25.0 / 6     # Performance noise
    TAU = 25.0 / 300    # Dynamic factor (skill drift)
    DRAW_PROB = 0.0     # No draws in Liar

    def __init__(self):
        self.players: Dict[str, TrueSkillPlayer] = {}

    def get_or_create(self, name: str) -> TrueSkillPlayer:
        if name not in self.players:
            self.players[name] = TrueSkillPlayer(name=name)
        return self.players[name]

    def update(self, ranked_names: List[str]) -> None:
        """
        Update ratings given a ranked list of player names
        (index 0 = winner, last = last place).
        """
        players = [self.get_or_create(n) for n in ranked_names]
        n = len(players)

        # Apply dynamics (increase uncertainty each game)
        for p in players:
            p.sigma = math.sqrt(p.sigma ** 2 + self.TAU ** 2)
            p.games_played += 1

        # Pairwise updates (winner beats everyone else)
        winner = players[0]
        for loser in players[1:]:
            self._update_pair(winner, loser)

    def _update_pair(self, winner: TrueSkillPlayer, loser: TrueSkillPlayer) -> None:
        """Update winner and loser ratings from a single head-to-head outcome."""
        c = math.sqrt(2 * self.BETA ** 2 + winner.sigma ** 2 + loser.sigma ** 2)
        t = (winner.mu - loser.mu) / c
        e = math.exp(-t ** 2 / 2) / (math.sqrt(2 * math.pi) * (0.5 * (1 + math.erf(t / math.sqrt(2)))))

        # Skill update
        winner_sigma2 = winner.sigma ** 2
        loser_sigma2 = loser.sigma ** 2

        winner.mu += (winner_sigma2 / c) * e
        loser.mu -= (loser_sigma2 / c) * e
        winner.sigma = math.sqrt(winner_sigma2 * (1 - (winner_sigma2 / c ** 2) * e * (e + t)))
        loser.sigma = math.sqrt(loser_sigma2 * (1 - (loser_sigma2 / c ** 2) * e * (e + t)))

        # Ensure sigma doesn't collapse
        winner.sigma = max(winner.sigma, 0.1)
        loser.sigma = max(loser.sigma, 0.1)

    def leaderboard(self) -> List[TrueSkillPlayer]:
        """Return players sorted by conservative rating."""
        return sorted(self.players.values(), key=lambda p: -p.rating)

    def print_leaderboard(self) -> None:
        print("\n=== TrueSkill Leaderboard ===")
        print(f"{'Rank':<5} {'Player':<20} {'Rating':>8} {'μ':>8} {'σ':>8} {'Games':>7}")
        print("-" * 60)
        for i, p in enumerate(self.leaderboard(), 1):
            marker = " ◄ GROK" if "Grok" in p.name else ""
            print(f"{i:<5} {p.name:<20} {p.rating:>8.1f} {p.mu:>8.1f} {p.sigma:>8.2f} {p.games_played:>7}{marker}")
