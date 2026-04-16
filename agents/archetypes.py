"""
Rule-based strategy archetypes from the paper.
Used as baselines against which the LLM agent is benchmarked.

Archetypes (with paper-reported win rates in 6-player games):
  TheSaint         - Never lies unless forced; rarely challenges         (51%)
  ComebackCloser   - Lies when losing, honest when winning              (43%)
  GameTheorist     - EIG-based discarding + Bayesian challenge calls    (variable)
  MrPathological   - Always lies; random challenges at 50%             (2%)
  TheAccountant    - Tracks lie frequencies; adaptive strategy         (variable)
  TheCollector     - Builds four-of-a-kinds; strategic challenges      (moderate)
  RandomAgent      - Baseline random play
"""

from __future__ import annotations
import random
import math
from typing import List, Tuple, TYPE_CHECKING

from .base_agent import BaseAgent
from game.card import Card, Rank

if TYPE_CHECKING:
    from game.game_state import GameState, ClaimRecord


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _worst_cards(hand: List[Card], rank: Rank, n: int) -> List[Card]:
    """
    Select n cards to play when forced to bluff.
    Prefer cards of ranks far from the current rank (least useful future cards).
    """
    non_matching = [c for c in hand if c.rank != rank]
    if not non_matching:
        return hand[:n]
    # Sort by distance from current rank (play the ones that cycle furthest away)
    non_matching.sort(key=lambda c: min(
        abs(c.rank.value - rank.value),
        13 - abs(c.rank.value - rank.value)
    ))
    return non_matching[:n]


def _select_honest_cards(hand: List[Card], rank: Rank, max_play: int = 4) -> List[Card]:
    """Return matching cards (up to max_play)."""
    matching = [c for c in hand if c.rank == rank]
    return matching[:max_play] if matching else []


# ---------------------------------------------------------------------------
# The Saint  (never lies unless literally has no matching cards)
# ---------------------------------------------------------------------------

class TheSaint(BaseAgent):
    """
    Play honestly whenever possible.
    If forced to bluff, play 1 card and claim 1.
    Never challenges unless it knows for certain the player is lying
    (more than 4 cards of a rank have been claimed/played this cycle).
    """

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        rank = state.current_rank
        matching = _select_honest_cards(player.hand, rank)
        if matching:
            return matching, len(matching)
        # Forced bluff — play one worst card, claim 1
        worst = _worst_cards(player.hand, rank, 1)
        return worst, 1

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        # Challenge only if total claimed for this rank cycle exceeds 4
        rank = claim.claimed_rank
        cycle = claim.rank_cycle
        total_claimed_this_rank = sum(
            r.claimed_count for r in state.claim_history
            if r.claimed_rank == rank
            and r.rank_cycle == cycle
            and not r.was_challenged
        ) + claim.claimed_count
        return total_claimed_this_rank > 4


# ---------------------------------------------------------------------------
# Comeback Closer  (bluffs when losing)
# ---------------------------------------------------------------------------

class ComebackCloser(BaseAgent):
    """
    Lies aggressively when losing (has more cards than average).
    Plays honestly and selectively challenges leaders.
    """

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        rank = state.current_rank
        matching = _select_honest_cards(player.hand, rank)

        avg_hand_size = sum(p.hand_size for p in state.players) / state.num_players
        is_losing = player.hand_size > avg_hand_size

        if matching and not is_losing:
            return matching, len(matching)

        # Bluff: play 1–3 worst cards, claim 2
        n_bluff = random.randint(1, min(3, player.hand_size))
        bluff_cards = _worst_cards(player.hand, rank, n_bluff)
        claimed = random.randint(1, min(3, player.hand_size))
        return bluff_cards, claimed

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        # Challenge the leader (player with fewest cards)
        leader = min(state.players, key=lambda p: p.hand_size if p.is_active else 999)
        if claim.player_id == leader.player_id:
            return random.random() < 0.5
        return False


# ---------------------------------------------------------------------------
# Game Theorist  (EIG-based)
# ---------------------------------------------------------------------------

class GameTheorist(BaseAgent):
    """
    Calculates a simplified Expected Information Gain (EIG) to decide challenges.
    Uses Bayesian updates on player history to estimate lie probability.
    """

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        rank = state.current_rank
        matching = _select_honest_cards(player.hand, rank)
        if matching:
            return matching, len(matching)
        worst = _worst_cards(player.hand, rank, 1)
        return worst, 1

    def _estimate_lie_prob(self, state: "GameState", claim: "ClaimRecord") -> float:
        """Bayesian estimate: P(lying | observable caught-lie history)."""
        pid = claim.player_id
        lie_count = state.player_caught_lie_counts.get(pid, 0)
        turn_count = state.player_turn_counts.get(pid, 1)
        prior = lie_count / max(turn_count, 1)

        # Adjust for claim plausibility (same rank cycle only)
        rank = claim.claimed_rank
        cycle = claim.rank_cycle
        total_that_rank_claimed = sum(
            r.claimed_count for r in state.claim_history
            if r.claimed_rank == rank and r.rank_cycle == cycle
        )
        remaining_plausible = max(0, 4 - total_that_rank_claimed)
        if claim.claimed_count > remaining_plausible:
            prior = min(prior + 0.3, 0.99)

        return prior

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        lie_prob = self._estimate_lie_prob(state, claim)
        pile_size = state.discard_pile_size + len(claim.actual_cards)
        my_hand = state.players[player_id].hand_size

        # EIG: expected cards gained/lost
        eig = lie_prob * pile_size - (1 - lie_prob) * pile_size
        return eig > 0


# ---------------------------------------------------------------------------
# Mr. Pathological  (always lies)
# ---------------------------------------------------------------------------

class MrPathological(BaseAgent):
    """
    Lies 100% of the time. Challenges randomly at 50% frequency.
    """

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        rank = state.current_rank
        # Always play non-matching cards
        non_matching = [c for c in player.hand if c.rank != rank]
        if non_matching:
            n = random.randint(1, min(3, len(non_matching)))
            return non_matching[:n], random.randint(1, 4)
        # All cards match — play them anyway but overclaim
        n = random.randint(1, min(2, len(player.hand)))
        return player.hand[:n], random.randint(n + 1, n + 3)

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        return random.random() < 0.5


# ---------------------------------------------------------------------------
# The Accountant  (tracks lie frequencies)
# ---------------------------------------------------------------------------

class TheAccountant(BaseAgent):
    """
    Tracks each player's lie frequency and doubles challenge rate for known liars.
    Adjusts discarding strategy based on which players are aggressive bluffers.
    """

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        rank = state.current_rank
        matching = _select_honest_cards(player.hand, rank)
        if matching:
            return matching, len(matching)
        worst = _worst_cards(player.hand, rank, 1)
        return worst, 1

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        pid = claim.player_id
        lie_freq = (
            state.player_caught_lie_counts.get(pid, 0) /
            max(state.player_turn_counts.get(pid, 1), 1)
        )
        # Base 20% challenge, doubled for each 10% lie frequency above 30%
        base_prob = 0.2
        if lie_freq > 0.3:
            multiplier = 2 ** ((lie_freq - 0.3) / 0.1)
            challenge_prob = min(base_prob * multiplier, 0.9)
        else:
            challenge_prob = base_prob
        return random.random() < challenge_prob


# ---------------------------------------------------------------------------
# The Collector  (builds four-of-a-kinds)
# ---------------------------------------------------------------------------

class TheCollector(BaseAgent):
    """
    Tries to accumulate four-of-a-kind sets by selectively picking up piles.
    Challenges when (hand_count + pile_play_count) == 4 to complete a set.
    """

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        rank = state.current_rank
        matching = _select_honest_cards(player.hand, rank)
        if matching:
            return matching, len(matching)
        worst = _worst_cards(player.hand, rank, 1)
        return worst, 1

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        # Challenge if gaining the pile would complete (or approach) a four-of-a-kind
        player = state.players[player_id]
        rank = claim.claimed_rank
        my_count = player.count_rank(rank)
        play_count = claim.claimed_count
        # Challenge if we'd get 4 of a kind
        if my_count + play_count >= 4:
            return True
        # Also challenge if pile is large and we have space
        if state.discard_pile_size > 8 and random.random() < 0.3:
            return True
        return False


# ---------------------------------------------------------------------------
# Random Agent  (baseline)
# ---------------------------------------------------------------------------

class RandomAgent(BaseAgent):
    """Completely random — plays 1 random card, claims a random count."""

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        n = random.randint(1, min(3, player.hand_size))
        cards = random.sample(player.hand, n)
        claimed = random.randint(1, min(4, player.hand_size))
        return cards, claimed

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        return random.random() < 0.33
