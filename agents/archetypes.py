"""
Rule-based strategy archetypes — adapted for the new engine interface.

Each archetype implements BaseAgent (from agents/base.py):
  choose_action(obs) -> {"type": "play", "cards": [...], "claimed_rank": str}
                     or {"type": "challenge"}
  observe_event(event) — accumulates public history for stateful agents

Deck is 104 cards (2 standard decks, no Jokers), so max honest cards per rank
per cycle is 8. Challenge impossibility threshold: >8.

Archetypes (paper-reported win rates in 6-player 52-card games, for reference):
  TheSaint        51%   Never lies unless forced; challenges only when impossible
  ComebackCloser  43%   Bluffs when losing, honest when winning
  GameTheorist    var   EIG-based challenges using Bayesian caught-lie history
  MrPathological   2%   Always lies; 50% random challenges
  TheAccountant   var   Tracks lie frequency; adaptive challenge rate
  TheCollector    mod   Builds sets; challenges to complete four-of-a-kinds
"""
from __future__ import annotations
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from engine.card import Card, RANK_INDEX
from .base import BaseAgent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _partition(hand: List[Card], rank: str) -> Tuple[List[Card], List[Card]]:
    """Split hand into (real_matching, off_rank)."""
    real   = [c for c in hand if c.rank == rank]
    off    = [c for c in hand if c.rank != rank]
    return real, off


def _worst_cards(hand: List[Card], rank: str, n: int) -> List[Card]:
    """
    Pick n cards to bluff with — prefer ranks furthest from current rank
    (least useful in upcoming turns).
    """
    off = [c for c in hand if c.rank != rank]
    if not off:
        return hand[:n]
    off.sort(key=lambda c: min(
        abs(RANK_INDEX.get(c.rank, 0) - RANK_INDEX.get(rank, 0)),
        13 - abs(RANK_INDEX.get(c.rank, 0) - RANK_INDEX.get(rank, 0))
    ), reverse=True)
    return off[:n]


def _is_opponents_claim(obs: dict, self_id: int) -> bool:
    """True if there is a pending claim made by someone other than self."""
    lc = obs.get("last_claim")
    return lc is not None and lc["player_id"] != self_id


# ---------------------------------------------------------------------------
# The Saint — never lies unless forced; challenges only when impossible
# ---------------------------------------------------------------------------

class TheSaint(BaseAgent):
    """
    Plays honestly whenever possible.
    Challenges only when the total claimed for this rank this cycle exceeds
    8 (impossible with 2 decks).
    """

    MAX_HONEST_PER_RANK = 8  # 2 decks × 4 suits

    def __init__(self, player_id: int, name: str = "TheSaint"):
        super().__init__(player_id, name)
        self._rank_claimed: Dict[str, int] = defaultdict(int)  # rank -> total claimed this cycle
        self._current_rank_seen: Optional[str] = None

    def reset(self) -> None:
        self._rank_claimed = defaultdict(int)
        self._current_rank_seen = None

    def observe_event(self, event: Dict[str, Any]) -> None:
        # Track claims per rank; reset counter when a challenge resolves the pile
        if "claimed_rank" in event:
            self._rank_claimed[event["claimed_rank"]] += event.get("n_cards", 0)
        if event.get("action", {}).get("type") == "challenge":
            # Pile was cleared — reset all rank counts
            self._rank_claimed = defaultdict(int)

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]

        # Challenge decision (if opponent just played)
        if _is_opponents_claim(obs, self.player_id):
            lc = obs["last_claim"]
            total_claimed = self._rank_claimed.get(rank, 0) + lc["n_cards"]
            if total_claimed > self.MAX_HONEST_PER_RANK:
                return {"type": "challenge"}

        # Play decision
        real, off = _partition(hand, rank)
        if real:
            return {"type": "play", "cards": real[:4], "claimed_rank": rank}
        # Forced bluff — play 1 worst card
        worst = _worst_cards(hand, rank, 1)
        return {"type": "play", "cards": worst, "claimed_rank": rank}


# ---------------------------------------------------------------------------
# Comeback Closer — bluffs when losing, honest when winning
# ---------------------------------------------------------------------------

class ComebackCloser(BaseAgent):
    """
    Compares own hand size to average opponent hand size.
    Bluffs aggressively when losing; plays honestly when ahead.
    Challenges the current leader (opponent with fewest cards).
    """

    def reset(self) -> None:
        pass

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        opp_sizes: Dict[int, int] = obs["opponent_hand_sizes"]

        # Challenge decision
        if _is_opponents_claim(obs, self.player_id):
            lc = obs["last_claim"]
            # Challenge the leader (opponent with fewest cards)
            if opp_sizes:
                leader_id = min(opp_sizes, key=opp_sizes.get)
                if lc["player_id"] == leader_id and random.random() < 0.5:
                    return {"type": "challenge"}

        # Play decision
        avg_opp = sum(opp_sizes.values()) / max(len(opp_sizes), 1) if opp_sizes else 0
        is_losing = len(hand) > avg_opp
        real, off = _partition(hand, rank)

        if real and not is_losing:
            return {"type": "play", "cards": real[:4], "claimed_rank": rank}

        # Bluff: play 1–3 worst cards
        n_bluff = random.randint(1, min(3, len(hand)))
        bluff_cards = _worst_cards(hand, rank, n_bluff)
        if not bluff_cards:
            bluff_cards = hand[:1]
        claimed = random.randint(1, min(3, len(hand)))
        return {"type": "play", "cards": bluff_cards[:claimed], "claimed_rank": rank}


# ---------------------------------------------------------------------------
# Game Theorist — EIG-based challenges with Bayesian history
# ---------------------------------------------------------------------------

class GameTheorist(BaseAgent):
    """
    Tracks each player's caught-lie frequency and turn counts.
    Challenge decision: EIG = P(lying)*pile_size - P(honest)*pile_size > 0.
    Play: honest first, otherwise minimal bluff.
    """

    def __init__(self, player_id: int, name: str = "GameTheorist"):
        super().__init__(player_id, name)
        self._lie_counts: Dict[int, int] = defaultdict(int)
        self._turn_counts: Dict[int, int] = defaultdict(int)
        self._rank_claimed: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self._lie_counts = defaultdict(int)
        self._turn_counts = defaultdict(int)
        self._rank_claimed = defaultdict(int)

    def observe_event(self, event: Dict[str, Any]) -> None:
        if "claimed_rank" in event:
            pid = event.get("player")
            if pid is not None:
                self._turn_counts[pid] += 1
                self._rank_claimed[event["claimed_rank"]] += event.get("n_cards", 0)
        if event.get("action", {}).get("type") == "challenge":
            if event.get("challenge_result") == "caught_bluffing":
                # The player who made the claim (pile_goes_to) was caught
                caught_pid = event.get("pile_goes_to")
                if caught_pid is not None:
                    self._lie_counts[caught_pid] += 1
            self._rank_claimed = defaultdict(int)

    def _lie_prob(self, pid: int, claimed_rank: str, n_claimed: int) -> float:
        turns = max(self._turn_counts.get(pid, 1), 1)
        prior = self._lie_counts.get(pid, 0) / turns
        # Adjust if claim seems implausible given rank history
        total_claimed = self._rank_claimed.get(claimed_rank, 0) + n_claimed
        if total_claimed > 8:  # exceeds 2-deck limit
            prior = min(prior + 0.3, 0.99)
        return prior

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]

        if _is_opponents_claim(obs, self.player_id):
            lc = obs["last_claim"]
            lie_prob = self._lie_prob(lc["player_id"], lc["claimed_rank"], lc["n_cards"])
            pile = obs["pile_size"]
            eig = lie_prob * pile - (1 - lie_prob) * pile
            if eig > 0:
                return {"type": "challenge"}

        real, off = _partition(hand, rank)
        if real:
            return {"type": "play", "cards": real[:4], "claimed_rank": rank}
        worst = _worst_cards(hand, rank, 1)
        return {"type": "play", "cards": worst, "claimed_rank": rank}


# ---------------------------------------------------------------------------
# Mr. Pathological — always lies; 50% random challenges
# ---------------------------------------------------------------------------

class MrPathological(BaseAgent):
    """Always plays off-rank cards. Challenges randomly at 50%."""

    def reset(self) -> None:
        pass

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]

        if _is_opponents_claim(obs, self.player_id) and random.random() < 0.5:
            return {"type": "challenge"}

        _, off = _partition(hand, rank)
        if off:
            n = random.randint(1, min(3, len(off)))
            return {"type": "play", "cards": off[:n], "claimed_rank": rank}
        # Only matching cards left — play them but overclaim
        n = random.randint(1, min(2, len(hand)))
        return {"type": "play", "cards": hand[:n], "claimed_rank": rank}


# ---------------------------------------------------------------------------
# The Accountant — tracks lie frequencies; adaptive challenge rate
# ---------------------------------------------------------------------------

class TheAccountant(BaseAgent):
    """
    Tracks each player's caught-lie rate. Base 20% challenge;
    doubles per 10% lie rate above 30% threshold.
    """

    def __init__(self, player_id: int, name: str = "TheAccountant"):
        super().__init__(player_id, name)
        self._lie_counts: Dict[int, int] = defaultdict(int)
        self._turn_counts: Dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        self._lie_counts = defaultdict(int)
        self._turn_counts = defaultdict(int)

    def observe_event(self, event: Dict[str, Any]) -> None:
        if "claimed_rank" in event:
            pid = event.get("player")
            if pid is not None:
                self._turn_counts[pid] += 1
        if event.get("action", {}).get("type") == "challenge":
            if event.get("challenge_result") == "caught_bluffing":
                caught_pid = event.get("pile_goes_to")
                if caught_pid is not None:
                    self._lie_counts[caught_pid] += 1

    def _challenge_prob(self, pid: int) -> float:
        turns = max(self._turn_counts.get(pid, 1), 1)
        lie_freq = self._lie_counts.get(pid, 0) / turns
        base = 0.2
        if lie_freq > 0.3:
            multiplier = 2 ** ((lie_freq - 0.3) / 0.1)
            return min(base * multiplier, 0.9)
        return base

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]

        if _is_opponents_claim(obs, self.player_id):
            lc = obs["last_claim"]
            prob = self._challenge_prob(lc["player_id"])
            if random.random() < prob:
                return {"type": "challenge"}

        real, off = _partition(hand, rank)
        if real:
            return {"type": "play", "cards": real[:4], "claimed_rank": rank}
        worst = _worst_cards(hand, rank, 1)
        return {"type": "play", "cards": worst, "claimed_rank": rank}


# ---------------------------------------------------------------------------
# Balanced Player — calibrated symmetric baseline for controlled experiments
# ---------------------------------------------------------------------------

class BalancedPlayer(BaseAgent):
    """
    Calibrated symmetric baseline for controlled/symmetric opponent pools.

    Design goals:
      1. Symmetric: 3 clones in a 4-player game each win ~25% by construction.
         Any LLM deviation from 25% is a clean skill signal.
      2. Games terminate naturally: random challenge rates cause wrong challenges
         that redistribute large piles back into hands, so hands never empty and
         games hit the turn cap. Accurate (impossibility-based) challenges avoid
         this — wrong challenges are essentially zero, so hands deplete at full
         rate and games end in ~60-100 turns.
      3. Symmetric state tracking: all clones observe the same public events and
         run the same algorithm, so symmetry holds even with state.

    Play     : all honest cards (up to 4) when available; 1 forced bluff otherwise.
    Challenge: only when total claimed for this rank this cycle exceeds 8
               (mathematically impossible with 2 decks of 4 suits each).
               Mirrors TheSaint's challenge logic exactly.
    """

    MAX_HONEST_PER_RANK: int = 8  # 2 decks × 4 suits

    def __init__(self, player_id: int, name: str = "BalancedPlayer"):
        super().__init__(player_id, name)
        self._rank_claimed: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self._rank_claimed = defaultdict(int)

    def observe_event(self, event: Dict[str, Any]) -> None:
        if "claimed_rank" in event:
            self._rank_claimed[event["claimed_rank"]] += event.get("n_cards", 0)
        if event.get("action", {}).get("type") == "challenge":
            self._rank_claimed = defaultdict(int)

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]

        # Challenge only when the total cards claimed for this rank this cycle
        # exceeds what's physically possible (>8 with 2 decks).
        if _is_opponents_claim(obs, self.player_id):
            lc = obs["last_claim"]
            total_claimed = self._rank_claimed.get(rank, 0) + lc["n_cards"]
            if total_claimed > self.MAX_HONEST_PER_RANK:
                return {"type": "challenge"}

        # Play: honest cards first (up to 4), forced bluff when none available.
        # Cap of 4 matches the engine's maximum play size and ensures BP's
        # throughput matches the LLM's (which can also play up to 4 cards).
        real, _ = _partition(hand, rank)
        if real:
            return {"type": "play", "cards": real[:4], "claimed_rank": rank}
        return {"type": "play", "cards": [hand[0]], "claimed_rank": rank}


# ---------------------------------------------------------------------------
# The Collector — builds four-of-a-kinds; challenges to complete sets
# ---------------------------------------------------------------------------

class TheCollector(BaseAgent):
    """
    Challenges when gaining the pile would complete (or approach) a four-of-a-kind.
    Also challenges on large piles with some probability.
    """

    def reset(self) -> None:
        pass

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]

        if _is_opponents_claim(obs, self.player_id):
            lc = obs["last_claim"]
            my_count = sum(1 for c in hand if c.rank == lc["claimed_rank"])
            if my_count + lc["n_cards"] >= 4:
                return {"type": "challenge"}
            if obs["pile_size"] > 8 and random.random() < 0.3:
                return {"type": "challenge"}

        real, off = _partition(hand, rank)
        if real:
            return {"type": "play", "cards": real[:4], "claimed_rank": rank}
        worst = _worst_cards(hand, rank, 1)
        return {"type": "play", "cards": worst, "claimed_rank": rank}
