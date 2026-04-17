"""
Scripted baseline agents.

HonestAgent    — always plays honestly; prefers real rank matches then Jokers; never challenges
BluffAgent     — prefers off-rank bluffs; treats Jokers as honest fallback; low-probability challenge
HeuristicAgent — practical deterministic baseline: truthful when possible, minimal bluff,
                 simple additive challenge heuristic, structured belief reporting
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from engine.card import Card, JOKER_RANK
from .base import BaseAgent


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _partition(hand: List[Card], rank: str) -> Tuple[List[Card], List[Card], List[Card]]:
    """Split hand into (real_matching, jokers, off_rank) for the current required rank."""
    real   = [c for c in hand if c.rank == rank]
    jokers = [c for c in hand if c.rank == JOKER_RANK]
    off    = [c for c in hand if c.rank != rank and c.rank != JOKER_RANK]
    return real, jokers, off


def _assess_bluff(
    last_claim: Dict[str, Any],
    pile_size: int,
    opp_hand_size: int,
) -> Dict[str, Any]:
    """
    Estimate how likely the opponent's last claim was a bluff.

    Additive scoring over three signals; no randomness.
    Returns {"opponent_bluff_prob": float, "confidence": float, "reason_tag": str}.
    """
    n_claimed  = last_claim["n_cards"]
    prob       = 0.25
    confidence = 0.40
    reason_tag = "base"

    # Claiming a large fraction of a small hand is hard to do honestly
    if opp_hand_size > 0 and n_claimed / opp_hand_size >= 0.5:
        prob       += 0.30
        confidence += 0.20
        reason_tag  = "high_claim_ratio"
    elif n_claimed >= 3:
        prob       += 0.20
        confidence += 0.15
        reason_tag  = "high_claim_count"

    # Opponent is short on cards — likely bluffing to dump fast
    if opp_hand_size <= 3:
        prob       += 0.20
        confidence += 0.15
        if reason_tag == "base":
            reason_tag = "low_opp_hand"

    # Large pile raises the incentive to bluff (force the opponent to pick up)
    if pile_size >= 15:
        prob       += 0.10
        confidence += 0.05
        if reason_tag == "base":
            reason_tag = "large_pile"

    return {
        "opponent_bluff_prob": round(min(1.0, prob), 3),
        "confidence":          round(min(1.0, confidence), 3),
        "reason_tag":          reason_tag,
    }


# ── HonestAgent ────────────────────────────────────────────────────────────────

class HonestAgent(BaseAgent):
    """
    Always plays honestly; never challenges.

    Card selection priority:
      1. Real rank-matching cards (played first).
      2. Jokers as supplementary honest cards (only if no real match).
      3. Off-rank card as forced bluff (only when hand has neither).
    """

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank            = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        real, jokers, off = _partition(hand, rank)

        # Real matches first, then Joker wildcard, then forced bluff
        if real:
            return {"type": "play", "cards": [real[0]], "claimed_rank": rank}
        if jokers:
            return {"type": "play", "cards": [jokers[0]], "claimed_rank": rank}
        return {"type": "play", "cards": [off[0]], "claimed_rank": rank}


# ── BluffAgent ─────────────────────────────────────────────────────────────────

class BluffAgent(BaseAgent):
    """
    Prefers to bluff by playing off-rank cards.

    Jokers are NOT used for bluffing — they would only produce honest plays,
    which defeats the purpose.  They're kept as a last resort when no off-rank
    cards remain.  Challenge probability is fixed and low.
    """

    CHALLENGE_PROB = 0.15

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        last = obs["last_claim"]
        if last and last["player_id"] != self.player_id:
            if random.random() < self.CHALLENGE_PROB:
                return {"type": "challenge"}

        rank            = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        real, jokers, off = _partition(hand, rank)

        if off:
            # Primary bluff: dump one off-rank card
            return {"type": "play", "cards": [off[0]], "claimed_rank": rank}
        # Fallback: only honest options remain (real match or Joker)
        honest = real + jokers
        return {"type": "play", "cards": [honest[0]], "claimed_rank": rank}


# ── HeuristicAgent ─────────────────────────────────────────────────────────────

_CHALLENGE_THRESHOLD = 0.55


class HeuristicAgent(BaseAgent):
    """
    Deterministic practical baseline.

    Play policy (in priority order):
      1. Real rank-matching card  → honest, no Joker spent.
      2. Joker wildcard           → honest, used only when no real match.
      3. Off-rank card            → minimal forced bluff.

    Challenge policy:
      Compute opponent_bluff_prob via _assess_bluff(); challenge when it
      exceeds _CHALLENGE_THRESHOLD (0.55).  No randomness.

    Belief:
      report_belief(obs) returns the belief dict computed during the most
      recent choose_action call, or None if no challenge decision was made
      this turn (i.e. the agent played without seeing a pending claim).
    """

    def __init__(self, player_id: int, name: str = "heuristic"):
        super().__init__(player_id, name)
        self._last_belief: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._last_belief = None

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._last_belief = None
        last = obs["last_claim"]

        if last and last["player_id"] != self.player_id:
            belief = _assess_bluff(last, obs["pile_size"], obs["opponent_hand_size"])
            self._last_belief = belief
            if belief["opponent_bluff_prob"] >= _CHALLENGE_THRESHOLD:
                return {"type": "challenge"}

        rank            = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        real, jokers, off = _partition(hand, rank)

        if real:
            return {"type": "play", "cards": [real[0]], "claimed_rank": rank}
        if jokers:
            return {"type": "play", "cards": [jokers[0]], "claimed_rank": rank}
        return {"type": "play", "cards": [off[0]], "claimed_rank": rank}

    def report_belief(self, obs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._last_belief
