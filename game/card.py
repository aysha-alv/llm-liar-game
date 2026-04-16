"""
Card, Deck, Rank, and Suit definitions for the Liar card game.
"""

from __future__ import annotations
from enum import Enum, IntEnum
from dataclasses import dataclass
import random
from typing import List


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class Rank(IntEnum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13

    def label(self) -> str:
        names = {
            1: "Ace", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
            6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
            11: "Jack", 12: "Queen", 13: "King"
        }
        return names[self.value]

    @staticmethod
    def next_rank(current: "Rank") -> "Rank":
        """Return the next rank in cycle (King → Ace)."""
        nxt = (current.value % 13) + 1
        return Rank(nxt)


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.label()}{self.suit.value}"

    def __repr__(self) -> str:
        return str(self)


class Deck:
    """Standard 52-card deck."""

    def __init__(self):
        self.cards: List[Card] = [
            Card(rank, suit)
            for suit in Suit
            for rank in Rank
        ]

    def shuffle(self, seed: int = None) -> None:
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, num_players: int) -> List[List[Card]]:
        """Deal all cards as evenly as possible among players."""
        hands: List[List[Card]] = [[] for _ in range(num_players)]
        for i, card in enumerate(self.cards):
            hands[i % num_players].append(card)
        return hands

    def __len__(self) -> int:
        return len(self.cards)
