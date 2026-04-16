"""
Phase 3 — Bluff-Aware Reward Shaping

Implements the paper's recommended reward function:
  - Sparse win/loss reward (+1 / -1) at game end
  - Dense bluff score for each play (calculated via Monte Carlo simulation)
  - Optimal bluff reward coefficient ≈ 2.0 (per paper)

The BluffAwareRewardShaper:
  1. Takes a ClaimRecord + GameState
  2. Runs N Monte Carlo rollouts to estimate the bluff's expected value
  3. Returns an auxiliary reward in [-1, +1]
"""

from __future__ import annotations
import random
import math
import logging
from typing import List, Tuple
from dataclasses import dataclass

from game.card import Card, Rank
from game.game_state import GameState, ClaimRecord

logger = logging.getLogger(__name__)


@dataclass
class BluffScore:
    """Result of evaluating a single bluff."""
    raw_score: float           # Monte Carlo EV estimate [-1, +1]
    shaped_reward: float       # Scaled by bluff_reward_coeff
    is_bluff: bool
    hand_strength: float       # Fraction of hand that matched current rank
    pile_pressure: float       # Pile size as fraction of total cards
    detection_risk: float      # Estimated probability of being challenged


class BluffAwareRewardShaper:
    """
    Computes dense per-move rewards to supplement sparse win/loss signals.

    Parameters
    ----------
    bluff_reward_coeff : float
        Scaling factor for bluff rewards. Paper reports 2.0 is optimal.
        Too high → over-bluffing pathology
        Too low  → passive "honesty" strategy, easily exploited
    mc_rollouts : int
        Number of Monte Carlo simulations per bluff evaluation.
    """

    def __init__(self, bluff_reward_coeff: float = 2.0, mc_rollouts: int = 50):
        self.bluff_reward_coeff = bluff_reward_coeff
        self.mc_rollouts = mc_rollouts

    def compute_reward(
        self,
        state: GameState,
        player_id: int,
        claim: ClaimRecord,
        was_challenged: bool,
        challenge_result: str | None,
    ) -> float:
        """
        Returns a combined reward for a single play.

        Terminal reward:
            +1.0 if agent won the game
            -1.0 if agent lost
             0.0 otherwise (game ongoing)

        Dense reward (added when game is ongoing):
            bluff_reward_coeff * bluff_score  if successful aggressive bluff
           -bluff_reward_coeff * 0.5          if caught reckless bluff
            0.0                               if honest play
        """
        terminal = self._terminal_reward(state, player_id)
        if terminal != 0:
            return terminal

        bluff_score = self._compute_bluff_score(state, player_id, claim)
        return bluff_score.shaped_reward

    def _terminal_reward(self, state: GameState, player_id: int) -> float:
        if not state.game_over:
            return 0.0
        return 1.0 if state.winner_id == player_id else -1.0

    def _compute_bluff_score(
        self,
        state: GameState,
        player_id: int,
        claim: ClaimRecord,
    ) -> BluffScore:
        rank = claim.claimed_rank
        is_bluff = claim.was_lying

        # Hand strength: fraction of hand that matched before play
        player = state.players[player_id]
        hand_size_before = player.hand_size + len(claim.actual_cards)
        matching_before = sum(1 for c in claim.actual_cards if c.rank == rank) + player.count_rank(rank)
        hand_strength = matching_before / max(hand_size_before, 1)

        # Pile pressure: picking up pile would hurt by this much
        pile_pressure = state.discard_pile_size / max(52, 1)

        # Detection risk: how suspicious is this claim?
        total_claimed = sum(
            r.claimed_count for r in state.claim_history if r.claimed_rank == rank
        )
        detection_risk = min(1.0, (total_claimed + claim.claimed_count) / 4.0)

        # Monte Carlo evaluation
        mc_ev = self._monte_carlo_bluff_ev(
            is_bluff, detection_risk, pile_pressure, claim.claimed_count
        )

        # Shape reward
        if is_bluff:
            if mc_ev > 0:
                # Successful aggressive bluff
                raw_reward = mc_ev * self.bluff_reward_coeff
            else:
                # Reckless, high-risk bluff
                raw_reward = mc_ev * self.bluff_reward_coeff * 0.5
        else:
            raw_reward = 0.0

        return BluffScore(
            raw_score=mc_ev,
            shaped_reward=raw_reward,
            is_bluff=is_bluff,
            hand_strength=hand_strength,
            pile_pressure=pile_pressure,
            detection_risk=detection_risk,
        )

    def _monte_carlo_bluff_ev(
        self,
        is_bluff: bool,
        detection_risk: float,
        pile_pressure: float,
        claimed_count: int,
    ) -> float:
        """
        Simulate N rollouts to estimate the expected value of a bluff.

        Returns a value in [-1, +1]:
          +1 = excellent bluff (shed cards without penalty)
          -1 = terrible bluff (high chance of being caught)
        """
        if not is_bluff:
            return 0.0

        wins = 0
        for _ in range(self.mc_rollouts):
            # Simulate: does the bluff succeed?
            challenge_prob = detection_risk * 0.7 + random.gauss(0, 0.05)
            challenge_prob = max(0.0, min(1.0, challenge_prob))

            if random.random() > challenge_prob:
                # Bluff succeeds — positive value proportional to cards shed
                ev = min(1.0, claimed_count / 13.0)
                wins += ev
            else:
                # Caught — penalized by pile size
                ev = -pile_pressure
                wins += ev

        return wins / self.mc_rollouts

    def batch_compute(
        self,
        episode: List[Tuple[GameState, int, ClaimRecord, bool, str]],
    ) -> List[float]:
        """Compute rewards for an entire episode at once."""
        rewards = []
        for state, pid, claim, challenged, result in episode:
            r = self.compute_reward(state, pid, claim, challenged, result)
            rewards.append(r)
        return rewards

    def explained_summary(self, bluff_score: BluffScore) -> str:
        """Human-readable explanation of a bluff evaluation."""
        lines = [
            f"Bluff: {'YES' if bluff_score.is_bluff else 'NO'}",
            f"  Hand strength: {bluff_score.hand_strength:.2f}",
            f"  Pile pressure: {bluff_score.pile_pressure:.2f}",
            f"  Detection risk: {bluff_score.detection_risk:.2f}",
            f"  MC Expected Value: {bluff_score.raw_score:+.3f}",
            f"  Shaped reward: {bluff_score.shaped_reward:+.3f}",
        ]
        return "\n".join(lines)
