"""
Phase 3 — Self-Play Training Framework (SPIRAL-inspired)

The agent plays against copies of itself (or a pool of past versions).
Each iteration:
  1. Run N games between current agent and previous versions
  2. Collect (state, action, reward) tuples
  3. Compute bluff-aware rewards
  4. Save episode data for fine-tuning
  5. (Optionally) update agent via fine-tuning API when available

Since Grok's xAI fine-tuning API is not yet public, this module:
  - Runs the self-play games
  - Accumulates high-quality experience data
  - Exports training data in OpenAI fine-tune format
  - Provides a hook for future fine-tuning calls
"""

from __future__ import annotations
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from game.liar_game import LiarGame
from game.game_state import GameState, ClaimRecord
from agents.base_agent import BaseAgent
from agents.archetypes import TheSaint, ComebackCloser, GameTheorist, TheAccountant
from training.reward_shaping import BluffAwareRewardShaper

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """One complete game worth of (state, action, reward) tuples."""
    game_id: int
    winner_id: int
    turns: int
    rewards: Dict[int, List[float]] = field(default_factory=dict)
    traces: List[dict] = field(default_factory=list)
    cumulative_reward: Dict[int, float] = field(default_factory=dict)


class SelfPlayTrainer:
    """
    Runs self-play iterations and accumulates training data.

    Opponent pool strategy:
    - Start with rule-based archetypes (curriculum level 1)
    - Then mix in previous Grok checkpoints (curriculum level 2)
    - Then play only against itself (curriculum level 3)
    """

    CURRICULUM_LEVELS = {
        1: "rule_based",        # vs archetypes only
        2: "mixed",             # vs archetypes + past self
        3: "self_play",         # vs self only
    }

    def __init__(
        self,
        agent: BaseAgent,
        bluff_reward_coeff: float = 2.0,
        mc_rollouts: int = 50,
        output_dir: Path = None,
    ):
        self.agent = agent
        self.rewarder = BluffAwareRewardShaper(
            bluff_reward_coeff=bluff_reward_coeff,
            mc_rollouts=mc_rollouts,
        )
        self.output_dir = output_dir or Path("data/self_play")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes: List[Episode] = []
        self.iteration = 0

    def run_iteration(
        self,
        n_games: int = 20,
        n_players: int = 4,
        curriculum_level: int = 1,
    ) -> dict:
        """
        Run one self-play iteration.
        Returns iteration statistics.
        """
        self.iteration += 1
        logger.info(f"Self-play iteration {self.iteration} | {n_games} games | curriculum={curriculum_level}")

        stats = defaultdict(float)
        all_training_data = []

        for game_idx in range(n_games):
            opponents = self._sample_opponents(n_players - 1, curriculum_level)
            agent_pos = random.randint(0, n_players - 1)
            agents = opponents[:agent_pos] + [self.agent] + opponents[agent_pos:]
            agents = agents[:n_players]
            agent_game_id = agents.index(self.agent)

            seed = random.randint(0, 9999999)
            episode = self._run_one_game(agents, agent_game_id, seed, game_idx)
            self.episodes.append(episode)

            # Stats
            won = episode.winner_id == agent_game_id
            stats["wins"] += int(won)
            stats["total_reward"] += sum(episode.rewards.get(agent_game_id, []))
            stats["avg_turns"] += episode.turns

            # Convert episode to training data
            training_examples = self._episode_to_training_data(episode, agent_game_id)
            all_training_data.extend(training_examples)

        # Save training data
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.output_dir / f"iter_{self.iteration:04d}_{ts}.jsonl"
        with open(out_file, "w") as f:
            for ex in all_training_data:
                f.write(json.dumps(ex) + "\n")

        stats["win_rate"] = stats["wins"] / n_games
        stats["avg_turns"] /= n_games
        stats["avg_total_reward"] = stats["total_reward"] / n_games
        stats["training_examples"] = len(all_training_data)

        logger.info(
            f"  Win rate: {stats['win_rate']:.2%} | "
            f"Avg reward: {stats['avg_total_reward']:.3f} | "
            f"Examples: {stats['training_examples']}"
        )

        self._save_iteration_stats(stats)
        return dict(stats)

    def run_curriculum(
        self,
        iterations_per_level: int = 5,
        games_per_iteration: int = 20,
        n_players: int = 4,
    ) -> List[dict]:
        """
        Full 3-level curriculum training run.
        Returns list of per-iteration stats.
        """
        all_stats = []
        for level, level_name in self.CURRICULUM_LEVELS.items():
            logger.info(f"\n=== Curriculum Level {level}: {level_name} ===")
            for _ in range(iterations_per_level):
                stats = self.run_iteration(
                    n_games=games_per_iteration,
                    n_players=n_players,
                    curriculum_level=level,
                )
                all_stats.append({"level": level, **stats})
        return all_stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_opponents(self, n: int, curriculum_level: int) -> List[BaseAgent]:
        archetype_classes = [TheSaint, ComebackCloser, GameTheorist, TheAccountant]
        opponents = []
        for i in range(n):
            cls = random.choice(archetype_classes)
            opponents.append(cls(name=f"{cls.__name__}_opp{i}"))
        return opponents

    def _run_one_game(
        self,
        agents: List[BaseAgent],
        agent_id: int,
        seed: int,
        game_idx: int,
    ) -> Episode:
        """Run a game and collect rewards at each step."""
        game = LiarGame(agents=agents, seed=seed, verbose=False)
        state = game.setup()

        rewards_per_player: Dict[int, List[float]] = defaultdict(list)
        traces = []

        # Run game manually to hook into each step for reward shaping
        for _ in range(LiarGame.MAX_TURNS):
            if state.game_over:
                break

            cp_idx = state.current_player_idx
            cp = state.current_player
            cur_agent = agents[cp_idx]

            cards_played, claimed_count = cur_agent.choose_play(state, cp_idx)
            cards_played = game._validate_play(cp, cards_played, state.current_rank)

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

            challenged = False
            challenge_result = None
            for other_idx, other_agent in enumerate(agents):
                if other_idx == cp_idx or not state.players[other_idx].is_active:
                    continue
                if other_agent.choose_challenge(state, other_idx, claim):
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
                        challenge_result = "liar_caught"
                    else:
                        claim.challenge_result = "honest_vindicated"
                        n_pile = len(state.discard_pile)
                        oth = state.players[other_idx]
                        oth.add_cards(state.discard_pile.copy())
                        oth.cards_picked_up += n_pile
                        oth.cards_picked_up_lost_challenge += n_pile
                        challenge_result = "honest_vindicated"
                    state.discard_pile.clear()
                    challenged = True
                    break

            # Compute reward for this step
            reward = self.rewarder.compute_reward(
                state, cp_idx, claim, challenged, challenge_result
            )
            rewards_per_player[cp_idx].append(reward)

            state.claim_history.append(claim)

            if cp.hand_size == 0:
                state.game_over = True
                state.winner_id = cp_idx
                # Terminal rewards
                for pid in range(len(agents)):
                    tr = 1.0 if pid == cp_idx else -0.5
                    rewards_per_player[pid].append(tr)
                break

            state.advance_rank()
            state.advance_player()
            state.turn_number += 1

        if not state.game_over:
            min_player = min(state.players, key=lambda p: p.hand_size)
            state.winner_id = min_player.player_id
            state.game_over = True

        # Collect LLM agent traces
        if hasattr(self.agent, "reasoning_traces"):
            traces = list(self.agent.reasoning_traces)
            self.agent.reasoning_traces.clear()

        return Episode(
            game_id=game_idx,
            winner_id=state.winner_id,
            turns=state.turn_number,
            rewards=dict(rewards_per_player),
            traces=traces,
            cumulative_reward={
                pid: sum(rs) for pid, rs in rewards_per_player.items()
            },
        )

    def _episode_to_training_data(self, episode: Episode, agent_id: int) -> List[dict]:
        """
        Convert an episode's reasoning traces to fine-tune format.
        Only include traces where the agent performed well (positive cumulative reward).
        """
        from agents.llm_agent import SYSTEM_PROMPT
        if episode.cumulative_reward.get(agent_id, 0) < 0:
            return []  # Skip bad episodes

        training_examples = []
        for trace in episode.traces:
            if trace.get("player_id") != agent_id:
                continue
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Turn {trace['turn']}, rank {trace['current_rank']}, hand_size {trace['hand_size']}"},
                    {"role": "assistant", "content": json.dumps({
                        "action": trace.get("action"),
                        "claimed_count": trace.get("claimed_count"),
                        "reasoning": trace.get("reasoning", ""),
                    })},
                ]
            }
            training_examples.append(example)
        return training_examples

    def _save_iteration_stats(self, stats: dict) -> None:
        stats_file = self.output_dir / "training_stats.jsonl"
        with open(stats_file, "a") as f:
            f.write(json.dumps({"iteration": self.iteration, **stats}) + "\n")
