"""GameState: the single source of truth for all hidden and public state.

Adapted from teammate's original 2-player version to support N players.
Changes: public_observation returns opponent_hand_sizes dict (all opponents)
instead of a single opponent_hand_size int.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .card import Card, RANKS


@dataclass
class Claim:
    """A player's declared play: N cards claimed to be `rank`."""
    player_id: int
    claimed_rank: str
    n_cards: int                          # number of cards claimed
    actual_cards: List[Card] = field(repr=False)  # hidden from opponents

    @property
    def is_honest(self) -> bool:
        return all(c.rank == self.claimed_rank for c in self.actual_cards)


@dataclass
class GameState:
    hands: List[List[Card]]         # hands[player_id] — hidden from opponents
    pile: List[Card]                 # face-down discard pile
    current_player: int              # whose turn it is
    current_rank: str                # rank that must be claimed this turn
    last_claim: Optional[Claim]      # the most recent claim (challengeable)
    turn: int                        # turn counter
    winner: Optional[int]            # set when game ends

    @classmethod
    def new(cls, hands: List[List[Card]], seed_rank: str = "A") -> "GameState":
        return cls(
            hands=hands,
            pile=[],
            current_player=0,
            current_rank=seed_rank,
            last_claim=None,
            turn=0,
            winner=None,
        )

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None

    def n_players(self) -> int:
        return len(self.hands)

    def hand_sizes(self) -> List[int]:
        return [len(h) for h in self.hands]

    def public_observation(self, for_player: int) -> dict:
        """Everything a player can legally see."""
        return {
            "my_hand": list(self.hands[for_player]),
            "my_hand_size": len(self.hands[for_player]),
            # All opponents' hand sizes keyed by player_id
            "opponent_hand_sizes": {
                pid: len(hand)
                for pid, hand in enumerate(self.hands)
                if pid != for_player
            },
            "pile_size": len(self.pile),
            "current_rank": self.current_rank,
            "current_player": self.current_player,
            "n_players": len(self.hands),
            "turn": self.turn,
            "last_claim": (
                {
                    "player_id": self.last_claim.player_id,
                    "claimed_rank": self.last_claim.claimed_rank,
                    "n_cards": self.last_claim.n_cards,
                }
                if self.last_claim
                else None
            ),
        }
