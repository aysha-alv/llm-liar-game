"""
Game state tracking for the Liar card game.
Tracks hands, discard pile, claim history, and challenge records.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .card import Card, Rank


@dataclass
class ClaimRecord:
    """A single play/claim made during the game."""
    player_id: int
    claimed_rank: Rank
    claimed_count: int       # How many cards the player claims to be playing
    actual_cards: List[Card]  # The cards they actually played (face-down)
    rank_cycle: int = 0      # Which full Ace→King cycle this play belongs to
    was_challenged: bool = False
    challenge_result: Optional[str] = None   # "liar_caught" | "honest_vindicated"
    challenger_id: Optional[int] = None

    @property
    def was_lying(self) -> bool:
        ranks_wrong = any(c.rank != self.claimed_rank for c in self.actual_cards)
        count_wrong = len(self.actual_cards) != self.claimed_count
        return ranks_wrong or count_wrong

    @property
    def actual_count(self) -> int:
        return len(self.actual_cards)


@dataclass
class PlayerState:
    """State for a single player."""
    player_id: int
    name: str
    hand: List[Card] = field(default_factory=list)
    cards_picked_up: int = 0    # Total cards picked up over the game
    challenges_issued: int = 0
    challenges_won: int = 0
    bluffs_attempted: int = 0
    bluffs_caught: int = 0
    turns_played: int = 0
    is_active: bool = True       # False when they've won (emptied hand)
    # Cards picked up after being caught lying (liar takes discard pile)
    cards_picked_up_bluff_caught: int = 0
    # Cards picked up after issuing a failed challenge (honest play vindicated)
    cards_picked_up_lost_challenge: int = 0

    @property
    def hand_size(self) -> int:
        return len(self.hand)

    def count_rank(self, rank: Rank) -> int:
        return sum(1 for c in self.hand if c.rank == rank)

    def cards_of_rank(self, rank: Rank) -> List[Card]:
        return [c for c in self.hand if c.rank == rank]

    def remove_cards(self, cards: List[Card]) -> None:
        for card in cards:
            self.hand.remove(card)

    def add_cards(self, cards: List[Card]) -> None:
        self.hand.extend(cards)


@dataclass
class GameState:
    """
    Full observable (and hidden) state of a Liar game.

    Public info:  discard pile SIZE, claim history, hand sizes per player
    Private info: actual card contents of each hand and the discard pile
    """
    num_players: int
    players: List[PlayerState] = field(default_factory=list)
    discard_pile: List[Card] = field(default_factory=list)
    claim_history: List[ClaimRecord] = field(default_factory=list)
    current_player_idx: int = 0
    current_rank: Rank = Rank.ACE
    turn_number: int = 0
    winner_id: Optional[int] = None
    game_over: bool = False
    # Increments each time rank wraps King → Ace (full deck cycle through ranks)
    current_rank_cycle: int = 0

    # Ground-truth bluff attempts (internal / post-hoc only; not exposed to agents)
    player_lie_counts: Dict[int, int] = field(default_factory=dict)
    player_turn_counts: Dict[int, int] = field(default_factory=dict)
    # Observable: caught lies only (increment when challenge proves liar)
    player_caught_lie_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    @property
    def discard_pile_size(self) -> int:
        return len(self.discard_pile)

    def hand_sizes(self) -> Dict[int, int]:
        return {p.player_id: p.hand_size for p in self.players}

    def last_claim(self) -> Optional[ClaimRecord]:
        return self.claim_history[-1] if self.claim_history else None

    def get_public_observation(self, observer_id: int) -> dict:
        """
        Returns what a player can legitimately observe:
        - Their own hand
        - Everyone's hand size
        - Discard pile size (not contents)
        - Full claim history (rank + count claimed, plus challenge outcomes)
        - Current rank
        - Whose turn it is
        """
        observer = self.players[observer_id]
        return {
            "my_hand": [str(c) for c in observer.hand],
            "my_hand_counts": {r.label(): observer.count_rank(r) for r in Rank},
            "hand_sizes": self.hand_sizes(),
            "discard_pile_size": self.discard_pile_size,
            "current_rank": self.current_rank.label(),
            "current_player_id": self.current_player_idx,
            "turn_number": self.turn_number,
            "claim_history": [
                {
                    "player": r.player_id,
                    "claimed_rank": r.claimed_rank.label(),
                    "claimed_count": r.claimed_count,
                    "actual_cards_placed": len(r.actual_cards),
                    "rank_cycle": r.rank_cycle,
                    "was_challenged": r.was_challenged,
                    "challenge_result": r.challenge_result,
                }
                for r in self.claim_history
            ],
            "current_rank_cycle": self.current_rank_cycle,
            # Observable: estimated from caught lies only (not ground-truth bluffs)
            "lie_frequencies": {
                pid: (self.player_caught_lie_counts.get(pid, 0) /
                      max(self.player_turn_counts.get(pid, 1), 1))
                for pid in range(self.num_players)
            },
        }

    def total_cards_in_play(self) -> int:
        return sum(p.hand_size for p in self.players) + self.discard_pile_size

    def cards_of_rank_remaining_in_hands(self, rank: Rank) -> int:
        """How many cards of a rank are currently in players' hands."""
        return sum(p.count_rank(rank) for p in self.players)

    def advance_player(self) -> None:
        """Move to next active player in clockwise order."""
        n = self.num_players
        for _ in range(n):
            self.current_player_idx = (self.current_player_idx + 1) % n
            if self.players[self.current_player_idx].is_active:
                break

    def advance_rank(self) -> None:
        if self.current_rank == Rank.KING:
            self.current_rank_cycle += 1
        self.current_rank = Rank.next_rank(self.current_rank)
