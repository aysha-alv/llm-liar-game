"""Game engine: applies actions and advances state.

Adapted from teammate's original 2-player version to support N players.
Changes:
  - _apply_play: current_player advances via (player + 1) % n_players
  - _apply_play: removed hardcoded `1 - player` turn validation
  - _apply_challenge: current_player after challenge = pile_goes_to (already player-agnostic)
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from .card import Card, next_rank
from .state import GameState, Claim

# ── Action types ──────────────────────────────────────────────────────────────
# play      : {"type": "play", "cards": [Card, ...], "claimed_rank": str}
# challenge : {"type": "challenge"}
# ─────────────────────────────────────────────────────────────────────────────

MAX_PLAY = 4   # max cards per play


class IllegalAction(Exception):
    pass


def apply_action(state: GameState, action: Dict[str, Any]) -> Tuple[GameState, Dict[str, Any]]:
    """
    Apply `action` for `state.current_player`.
    Returns (new_state, event_dict) — engine never mutates state in-place.
    """
    state = _copy_state(state)
    player = state.current_player
    event: Dict[str, Any] = {"turn": state.turn, "player": player, "action": action}

    if action["type"] == "play":
        state, event = _apply_play(state, player, action, event)
    elif action["type"] == "challenge":
        state, event = _apply_challenge(state, player, event)
    else:
        raise IllegalAction(f"Unknown action type: {action['type']}")

    # Check win condition after every action
    for pid, hand in enumerate(state.hands):
        if len(hand) == 0:
            state.winner = pid
            event["winner"] = pid
            break

    return state, event


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_play(state: GameState, player: int, action: dict, event: dict):
    n = len(state.hands)

    cards: List[Card] = action["cards"]
    claimed_rank: str = action.get("claimed_rank", state.current_rank)

    if not (1 <= len(cards) <= MAX_PLAY):
        raise IllegalAction(f"Must play 1–{MAX_PLAY} cards.")
    if claimed_rank != state.current_rank:
        raise IllegalAction(f"Must claim current rank {state.current_rank!r}.")

    # Remove cards from hand
    hand = list(state.hands[player])
    for card in cards:
        if card not in hand:
            raise IllegalAction(f"Card {card} not in hand.")
        hand.remove(card)
    state.hands[player] = hand

    claim = Claim(
        player_id=player,
        claimed_rank=claimed_rank,
        n_cards=len(cards),
        actual_cards=cards,
    )
    state.pile.extend(cards)
    state.last_claim = claim

    event["claimed_rank"] = claimed_rank
    event["n_cards"] = len(cards)
    event["actual_cards"] = [str(c) for c in cards]
    event["honest"] = claim.is_honest

    # Advance to next player (N-player safe)
    state.current_player = (player + 1) % n
    state.current_rank = next_rank(claimed_rank)
    state.turn += 1
    return state, event


def _apply_challenge(state: GameState, challenger: int, event: dict):
    if state.last_claim is None:
        raise IllegalAction("Nothing to challenge.")
    if challenger == state.last_claim.player_id:
        raise IllegalAction("Cannot challenge your own claim.")

    claim = state.last_claim
    honest = claim.is_honest

    event["challenge_result"] = "honest" if honest else "caught_bluffing"
    event["actual_cards"] = [str(c) for c in claim.actual_cards]

    pile = list(state.pile)
    state.pile = []
    state.last_claim = None

    if honest:
        # Challenger was wrong — they pick up the pile
        state.hands[challenger].extend(pile)
        event["pile_goes_to"] = challenger
    else:
        # Bluffer was caught — they pick up the pile
        state.hands[claim.player_id].extend(pile)
        event["pile_goes_to"] = claim.player_id

    # Loser of the challenge plays next
    state.current_player = event["pile_goes_to"]
    state.turn += 1
    return state, event


def _copy_state(state: GameState) -> GameState:
    return GameState(
        hands=[list(h) for h in state.hands],
        pile=list(state.pile),
        current_player=state.current_player,
        current_rank=state.current_rank,
        last_claim=state.last_claim,
        turn=state.turn,
        winner=state.winner,
    )
