"""Card primitives: rank, suit, deck, hand."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["♠", "♥", "♦", "♣"]
RANK_INDEX = {r: i for i, r in enumerate(RANKS)}

JOKER_RANK = "Joker"
JOKER_SUIT = "★"


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return str(self)


def full_deck() -> List[Card]:
    """Two standard 52-card decks plus 4 Jokers = 108 cards."""
    standard = [Card(r, s) for r in RANKS for s in SUITS]
    jokers = [Card(JOKER_RANK, JOKER_SUIT)] * 4
    return standard * 2 + jokers


def deal(deck: List[Card], n_players: int) -> List[List[Card]]:
    """Deal all cards as evenly as possible among n_players."""
    hands: List[List[Card]] = [[] for _ in range(n_players)]
    for i, card in enumerate(deck):
        hands[i % n_players].append(card)
    return hands


def next_rank(rank: str) -> str:
    return RANKS[(RANK_INDEX[rank] + 1) % len(RANKS)]


def card_from_str(s: str) -> Card:
    """Parse "A♠", "10♦", "Joker★" etc. back into a Card. Raises ValueError on bad input."""
    if s == f"{JOKER_RANK}{JOKER_SUIT}":
        return Card(JOKER_RANK, JOKER_SUIT)
    for suit in SUITS:
        if s.endswith(suit):
            rank = s[: -len(suit)]
            if rank in RANK_INDEX:
                return Card(rank, suit)
    raise ValueError(f"Cannot parse card string: {s!r}")
