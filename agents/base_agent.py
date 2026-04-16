"""
Abstract base class for all Liar game agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from game.game_state import GameState, ClaimRecord
    from game.card import Card


class BaseAgent(ABC):
    """
    Every agent — rule-based or LLM — must implement two methods:

    choose_play:
        Given the full game state and the agent's player_id,
        return (cards_to_play, claimed_count).
        - cards_to_play: actual Card objects from the agent's hand
        - claimed_count: the number the agent CLAIMS to be playing
          (may differ from actual for bluffing)

    choose_challenge:
        Given the full game state, the agent's player_id, and the
        most recent ClaimRecord, return True to challenge or False to pass.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_play(
        self,
        state: "GameState",
        player_id: int,
    ) -> Tuple[List["Card"], int]:
        """Return (actual_cards_played, claimed_count)."""
        ...

    @abstractmethod
    def choose_challenge(
        self,
        state: "GameState",
        player_id: int,
        claim: "ClaimRecord",
    ) -> bool:
        """Return True to challenge the most recent play."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
