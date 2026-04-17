"""Base agent interface. All agents — scripted, LLM, human — implement this."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    def __init__(self, player_id: int, name: str = ""):
        self.player_id = player_id
        self.name = name or type(self).__name__

    @abstractmethod
    def choose_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive a public observation dict and return an action dict.

        Observation keys (see GameState.public_observation):
            my_hand, my_hand_size, opponent_hand_size,
            pile_size, current_rank, current_player, turn, last_claim

        Action must be one of:
            {"type": "play", "cards": [...], "claimed_rank": str}
            {"type": "challenge"}
        """

    def observe_event(self, public_event: Dict[str, Any]) -> None:
        """
        Called by the runner for BOTH players after every action, with a
        player-appropriate filtered view (opponents never see actual_cards
        from a play — only from a resolved challenge).

        Stateless agents ignore this. History-aware agents accumulate it.
        """

    def report_belief(self, obs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Called by the runner immediately after choose_action(obs).
        Return a belief dict or None if the agent does not report beliefs.

        Structured belief format:
            {"opponent_bluff_prob": float, "confidence": float, "reason_tag": str}

        obs is passed so stateless agents can compute beliefs on demand;
        history-aware agents may ignore it and return a cached value.
        """
        return None

    def reset(self) -> None:
        """Called at the start of each game episode."""
