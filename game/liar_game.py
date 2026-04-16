"""
Core Liar game engine.

Rules implemented:
- 52 cards dealt among 3–10 players
- Ranks cycle Ace → 2 → 3 → ... → King → Ace
- On your turn: play 1+ cards face-down and claim they are the current rank
- Any other player may challenge ("Bullshit!")
  - If lying  → liar picks up entire discard pile
  - If honest → challenger picks up entire discard pile
- After a challenge (regardless of outcome) the rank advances and play continues
- If no challenge: rank advances, next player goes
- First player to empty their hand wins
"""

from __future__ import annotations
import random
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

from .card import Card, Deck, Rank
from .game_state import GameState, PlayerState, ClaimRecord

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LiarGame:
    """
    Orchestrates a full game of Liar between agent objects.

    Each agent must implement:
        choose_play(state, player_id) -> (cards_to_play, claimed_count)
        choose_challenge(state, player_id, claim) -> bool
    """

    MAX_TURNS = 500  # safety cap to prevent infinite loops

    def __init__(self, agents: List["BaseAgent"], seed: int = None, verbose: bool = False):
        assert 3 <= len(agents) <= 10, "Liar requires 3–10 players"
        self.agents = agents
        self.seed = seed
        self.verbose = verbose
        self.state: Optional[GameState] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> GameState:
        deck = Deck()
        deck.shuffle(seed=self.seed)
        hands = deck.deal(len(self.agents))

        players = [
            PlayerState(
                player_id=i,
                name=self.agents[i].name,
                hand=hands[i],
            )
            for i in range(len(self.agents))
        ]

        self.state = GameState(
            num_players=len(self.agents),
            players=players,
            current_rank=Rank.ACE,
            current_player_idx=0,
        )
        return self.state

    # ------------------------------------------------------------------
    # Main game loop
    # ------------------------------------------------------------------

    def play(self) -> Tuple[int, GameState]:
        """
        Run a full game. Returns (winner_id, final_state).
        """
        if self.state is None:
            self.setup()

        state = self.state

        for _ in range(self.MAX_TURNS):
            if state.game_over:
                break

            cp = state.current_player
            agent = self.agents[cp.player_id]

            if self.verbose:
                logger.info(f"Turn {state.turn_number}: {cp.name} (hand={cp.hand_size} cards), rank={state.current_rank.label()}")

            # --- PLAY PHASE ---
            obs = state.get_public_observation(cp.player_id)
            cards_played, claimed_count = agent.choose_play(state, cp.player_id)

            # Validate: must play at least 1 card that's in their hand
            cards_played = self._validate_play(cp, cards_played, state.current_rank)

            claim = ClaimRecord(
                player_id=cp.player_id,
                claimed_rank=state.current_rank,
                claimed_count=claimed_count,
                actual_cards=cards_played,
                rank_cycle=state.current_rank_cycle,
            )

            # Remove played cards from hand
            cp.remove_cards(cards_played)
            state.discard_pile.extend(cards_played)
            cp.turns_played += 1

            # Track lie frequency
            state.player_turn_counts[cp.player_id] = state.player_turn_counts.get(cp.player_id, 0) + 1
            if claim.was_lying:
                cp.bluffs_attempted += 1
                state.player_lie_counts[cp.player_id] = state.player_lie_counts.get(cp.player_id, 0) + 1

            if self.verbose:
                logger.info(f"  {cp.name} plays {len(cards_played)} card(s), claims {claimed_count}x {state.current_rank.label()}")

            # --- CHALLENGE PHASE ---
            challenged = False
            for other in state.players:
                if other.player_id == cp.player_id or not other.is_active:
                    continue
                other_agent = self.agents[other.player_id]
                if other_agent.choose_challenge(state, other.player_id, claim):
                    # Process challenge
                    claim.was_challenged = True
                    claim.challenger_id = other.player_id
                    other.challenges_issued += 1

                    if claim.was_lying:
                        # Liar caught — liar picks up pile
                        claim.challenge_result = "liar_caught"
                        cp.bluffs_caught += 1
                        n_pile = len(state.discard_pile)
                        state.player_caught_lie_counts[cp.player_id] = (
                            state.player_caught_lie_counts.get(cp.player_id, 0) + 1
                        )
                        cp.add_cards(state.discard_pile.copy())
                        cp.cards_picked_up += n_pile
                        cp.cards_picked_up_bluff_caught += n_pile
                        other.challenges_won += 1
                        if self.verbose:
                            logger.info(f"  CHALLENGE by {other.name}: LIAR CAUGHT! {cp.name} picks up {len(state.discard_pile)} cards.")
                    else:
                        # Honest play — challenger picks up pile
                        claim.challenge_result = "honest_vindicated"
                        n_pile = len(state.discard_pile)
                        other.add_cards(state.discard_pile.copy())
                        other.cards_picked_up += n_pile
                        other.cards_picked_up_lost_challenge += n_pile
                        if self.verbose:
                            logger.info(f"  CHALLENGE by {other.name}: HONEST PLAY! {other.name} picks up {len(state.discard_pile)} cards.")

                    state.discard_pile.clear()
                    challenged = True
                    break  # Only one challenge per play

            state.claim_history.append(claim)

            # --- CHECK WIN ---
            if cp.hand_size == 0:
                state.game_over = True
                state.winner_id = cp.player_id
                cp.is_active = False
                if self.verbose:
                    logger.info(f"  {cp.name} wins!")
                break

            # --- ADVANCE ---
            state.advance_rank()
            state.advance_player()
            state.turn_number += 1

        if not state.game_over:
            # Hit turn cap — pick player with fewest cards
            min_hand = min(state.players, key=lambda p: p.hand_size)
            state.winner_id = min_hand.player_id
            state.game_over = True
            logger.warning(f"Turn cap reached. Winner by fewest cards: {min_hand.name}")

        return state.winner_id, state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_play(self, player: PlayerState, cards: List[Card], rank: Rank) -> List[Card]:
        """
        Ensure the agent returned valid cards that exist in the player's hand.
        Falls back to playing a single random card if invalid.
        """
        valid = []
        hand_copy = list(player.hand)
        for c in cards:
            if c in hand_copy:
                valid.append(c)
                hand_copy.remove(c)

        if not valid:
            # Fallback: play one random card
            logger.warning(f"Agent {player.name} returned invalid cards. Playing random card.")
            valid = [random.choice(player.hand)]

        return valid
