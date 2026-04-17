"""
Game logger: records every turn as a structured JSON-serialisable dict.

Episode log schema
------------------
{
  "episode_id": str,
  "seed": int,
  "agents": [{"player_id": int, "name": str, "type": str}, ...],
  "turns": [
    {
      "turn": int,
      "player": int,
      "action": {...},
      "observation_before": {...},   # what the acting player saw
      "event": {...},                # engine event (includes actual_cards, honest, etc.)
      "hand_sizes_after": [int, int]
    },
    ...
  ],
  "outcome": {"winner": int, "total_turns": int, "reason": str}
}
"""
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from engine.card import Card


def _serialise(obj: Any) -> Any:
    if isinstance(obj, Card):
        return str(obj)
    if isinstance(obj, list):
        return [_serialise(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


@dataclass
class EpisodeLogger:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    seed: int = 0
    agents: List[Dict] = field(default_factory=list)
    turns: List[Dict] = field(default_factory=list)
    outcome: Optional[Dict] = None

    def log_turn(
        self,
        turn: int,
        player: int,
        action: Dict,
        observation_before: Dict,
        event: Dict,
        hand_sizes_after: List[int],
        beliefs: Optional[Dict] = None,
    ) -> None:
        record: Dict = {
            "turn": turn,
            "player": player,
            "action": action,
            "observation_before": observation_before,
            "event": event,
            "hand_sizes_after": hand_sizes_after,
        }
        if beliefs is not None:
            record["beliefs"] = beliefs
        self.turns.append(_serialise(record))

    def log_outcome(self, winner: int, total_turns: int, reason: str = "empty_hand") -> None:
        self.outcome = {"winner": winner, "total_turns": total_turns, "reason": reason}

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "agents": self.agents,
            "turns": self.turns,
            "outcome": self.outcome,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
