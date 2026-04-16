"""
Phase 2 — Synthetic Reasoning Trace Generator

Generates high-quality rule-based reasoning traces for supervised fine-tuning.
Each trace follows the paper's format:
  1. Observation  (game state)
  2. Logic        (what's deducible)
  3. Risk         (probability analysis)
  4. Action       (final decision with rationale)

Traces are saved as JSONL in OpenAI fine-tune format:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    python scripts/phase2_finetune.py --traces 1000 --output data/traces/synthetic.jsonl
"""

from __future__ import annotations
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from game.card import Card, Deck, Rank, Suit
from game.game_state import GameState, PlayerState, ClaimRecord
from game.liar_game import LiarGame
from agents.archetypes import TheSaint, ComebackCloser, TheAccountant

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTrace:
    """A single (state, action, reasoning) example."""
    observation: str
    logic: str
    risk_assessment: str
    action_type: str          # "play" or "challenge" or "pass"
    action_detail: dict       # cards played, claimed count, etc.
    full_reasoning: str
    game_state_snapshot: dict


# ---------------------------------------------------------------------------
# Trace generator
# ---------------------------------------------------------------------------

class SyntheticTraceGenerator:
    """
    Simulates games using rule-based agents and extracts
    expert-quality reasoning traces from each decision point.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def generate(self, n_traces: int = 1000, output_path: Path = None) -> List[dict]:
        """
        Generate n_traces reasoning trace examples.
        Returns list of OpenAI fine-tune format messages.
        """
        traces = []
        games_needed = max(n_traces // 20, 50)  # ~20 decisions per game

        logger.info(f"Generating {n_traces} traces from ~{games_needed} simulated games...")

        for game_idx in range(games_needed):
            if len(traces) >= n_traces:
                break

            game_traces = self._simulate_game_with_traces(game_idx)
            traces.extend(game_traces)

            if (game_idx + 1) % 50 == 0:
                logger.info(f"Games simulated: {game_idx + 1}, traces collected: {len(traces)}")

        traces = traces[:n_traces]
        logger.info(f"Generated {len(traces)} traces.")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for trace in traces:
                    f.write(json.dumps(trace) + "\n")
            logger.info(f"Saved traces → {output_path}")

        return traces

    def _simulate_game_with_traces(self, game_idx: int) -> List[dict]:
        """Run one game and capture decision traces."""
        from agents.archetypes import (
            TheSaint, ComebackCloser, GameTheorist, TheAccountant, TheCollector
        )

        archetype_pool = [TheSaint, ComebackCloser, GameTheorist, TheAccountant, TheCollector]
        n_players = random.randint(3, 5)
        chosen = random.sample(archetype_pool, n_players)
        agents = [cls(name=f"{cls.__name__}_{i}") for i, cls in enumerate(chosen)]

        seed = game_idx * 13 + 7
        game = LiarGame(agents=agents, seed=seed, verbose=False)
        state = game.setup()

        traces = []
        for turn in range(LiarGame.MAX_TURNS):
            if state.game_over:
                break

            cp_idx = state.current_player_idx
            cp = state.current_player
            agent = agents[cp_idx]

            # Capture state BEFORE play
            obs = state.get_public_observation(cp_idx)

            # Generate play trace
            cards_played, claimed_count = agent.choose_play(state, cp_idx)
            cards_played = game._validate_play(cp, cards_played, state.current_rank)

            play_trace = self._build_play_trace(state, cp_idx, cards_played, claimed_count, obs)
            traces.append(play_trace)

            # Apply play to state
            claim = ClaimRecord(
                player_id=cp_idx,
                claimed_rank=state.current_rank,
                claimed_count=claimed_count,
                actual_cards=cards_played,
                rank_cycle=state.current_rank_cycle,
            )
            cp.remove_cards(cards_played)
            state.discard_pile.extend(cards_played)
            cp.turns_played += 1
            state.player_turn_counts[cp_idx] = state.player_turn_counts.get(cp_idx, 0) + 1
            if claim.was_lying:
                cp.bluffs_attempted += 1
                state.player_lie_counts[cp_idx] = state.player_lie_counts.get(cp_idx, 0) + 1

            # Generate challenge traces for other players
            challenged = False
            for other_idx, other_agent in enumerate(agents):
                if other_idx == cp_idx or not state.players[other_idx].is_active:
                    continue
                other_obs = state.get_public_observation(other_idx)
                will_challenge = other_agent.choose_challenge(state, other_idx, claim)

                challenge_trace = self._build_challenge_trace(
                    state, other_idx, claim, will_challenge, other_obs
                )
                traces.append(challenge_trace)

                if will_challenge:
                    claim.was_challenged = True
                    claim.challenger_id = other_idx
                    state.players[other_idx].challenges_issued += 1
                    if claim.was_lying:
                        claim.challenge_result = "liar_caught"
                        cp.bluffs_caught += 1
                        n_pile = len(state.discard_pile)
                        state.player_caught_lie_counts[cp_idx] = (
                            state.player_caught_lie_counts.get(cp_idx, 0) + 1
                        )
                        cp.add_cards(state.discard_pile.copy())
                        cp.cards_picked_up += n_pile
                        cp.cards_picked_up_bluff_caught += n_pile
                        state.players[other_idx].challenges_won += 1
                    else:
                        claim.challenge_result = "honest_vindicated"
                        n_pile = len(state.discard_pile)
                        oth = state.players[other_idx]
                        oth.add_cards(state.discard_pile.copy())
                        oth.cards_picked_up += n_pile
                        oth.cards_picked_up_lost_challenge += n_pile
                    state.discard_pile.clear()
                    challenged = True
                    break

            state.claim_history.append(claim)

            if cp.hand_size == 0:
                state.game_over = True
                state.winner_id = cp_idx
                break

            state.advance_rank()
            state.advance_player()
            state.turn_number += 1

        return traces

    def _build_play_trace(
        self,
        state: GameState,
        player_id: int,
        cards_played: List[Card],
        claimed_count: int,
        obs: dict,
    ) -> dict:
        player = state.players[player_id]
        rank = state.current_rank
        matching = [c for c in player.hand if c.rank == rank]
        is_bluff = any(c.rank != rank for c in cards_played) or len(cards_played) != claimed_count

        # Build natural language observation
        observation = (
            f"Current rank is {rank.label()}. "
            f"Discard pile has {state.discard_pile_size} cards. "
            f"My hand has {player.hand_size} cards total: "
            f"{player.count_rank(rank)}x {rank.label()} (matching). "
            f"Hand sizes — " +
            ", ".join(f"P{p.player_id}: {p.hand_size}" for p in state.players if p.player_id != player_id) +
            "."
        )

        # Build logic trace (current rank cycle only)
        cycle = state.current_rank_cycle
        total_claimed_this_rank = sum(
            r.claimed_count for r in state.claim_history
            if r.claimed_rank == rank and r.rank_cycle == cycle
        )
        remaining_plausible = max(0, 4 - total_claimed_this_rank)

        logic = (
            f"There are {len(matching)} {rank.label()}s in my hand. "
            f"So far this rank cycle, {total_claimed_this_rank} {rank.label()}s have been claimed, "
            f"leaving {remaining_plausible} plausible to claim. "
        )
        if is_bluff:
            logic += (
                f"I don't have enough matching cards, so I must bluff. "
                f"I'll play {len(cards_played)} non-matching card(s) to minimize hand size."
            )
        else:
            logic += (
                f"I have honest cards to play. Playing {len(cards_played)} {rank.label()}(s) truthfully."
            )

        risk = (
            f"{'Bluff' if is_bluff else 'Honest'} play. "
            f"Pile size is {state.discard_pile_size}. "
            f"Risk of challenge: {'moderate' if state.discard_pile_size > 5 else 'low'}."
        )

        action_str = json.dumps({
            "action": "play",
            "cards": [str(c) for c in cards_played],
            "claimed_count": claimed_count,
            "reasoning": f"{logic} {risk}",
        }, indent=2)

        user_msg = self._build_play_user_msg(obs, rank, player)
        return self._format_finetune_example(user_msg, action_str)

    def _build_challenge_trace(
        self,
        state: GameState,
        player_id: int,
        claim: ClaimRecord,
        decision: bool,
        obs: dict,
    ) -> dict:
        player = state.players[player_id]
        rank = state.current_rank

        claimer_lie_freq = obs["lie_frequencies"].get(claim.player_id, 0.0)
        cycle = claim.rank_cycle
        total_rank_claimed = sum(
            r["claimed_count"] for r in obs["claim_history"]
            if r["claimed_rank"] == rank.label() and r.get("rank_cycle", cycle) == cycle
        ) + claim.claimed_count

        logic = (
            f"P{claim.player_id} claims {claim.claimed_count}x {rank.label()}. "
            f"Their lie frequency: {claimer_lie_freq:.2f}. "
            f"Total {rank.label()}s claimed this cycle: {total_rank_claimed}/4. "
        )
        if total_rank_claimed > 4:
            logic += "IMPOSSIBLE — more than 4 claimed. Must be lying."
        elif claimer_lie_freq > 0.5:
            logic += "High lie frequency — likely bluffing."
        else:
            logic += "Plausible claim."

        pile_risk = state.discard_pile_size
        risk = (
            f"If wrong challenge: pick up {pile_risk} cards. "
            f"Confidence in challenge: {'high' if total_rank_claimed > 4 or claimer_lie_freq > 0.6 else 'low'}."
        )

        action_str = json.dumps({
            "action": "challenge" if decision else "pass_challenge",
            "reasoning": f"{logic} {risk}",
        }, indent=2)

        user_msg = self._build_challenge_user_msg(obs, rank, player, claim, state)
        return self._format_finetune_example(user_msg, action_str)

    def _build_play_user_msg(self, obs: dict, rank: Rank, player: PlayerState) -> str:
        hand = ", ".join(obs["my_hand"])
        sizes = "; ".join(f"P{k}: {v}" for k, v in obs["hand_sizes"].items())
        return (
            f"Required rank: {rank.label()}\n"
            f"My hand ({player.hand_size}): {hand}\n"
            f"Hand sizes: {sizes}\n"
            f"Discard pile: {obs['discard_pile_size']} cards\n"
            f"YOUR TURN TO PLAY."
        )

    def _build_challenge_user_msg(
        self, obs: dict, rank: Rank, player: PlayerState,
        claim: ClaimRecord, state: GameState
    ) -> str:
        hand = ", ".join(obs["my_hand"])
        sizes = "; ".join(f"P{k}: {v}" for k, v in obs["hand_sizes"].items())
        return (
            f"Required rank: {rank.label()}\n"
            f"My hand ({player.hand_size}): {hand}\n"
            f"Hand sizes: {sizes}\n"
            f"Discard pile: {obs['discard_pile_size']} cards\n"
            f"P{claim.player_id} placed {len(claim.actual_cards)} card(s), "
            f"claims {claim.claimed_count}x {rank.label()}.\n"
            f"CHALLENGE DECISION."
        )

    def _format_finetune_example(self, user_msg: str, assistant_msg: str) -> dict:
        from agents.llm_agent import SYSTEM_PROMPT
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }
