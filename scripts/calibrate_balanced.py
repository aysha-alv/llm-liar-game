"""
Calibration check for BalancedPlayer symmetry.

Runs N games with 4 BalancedPlayer clones (no LLM, no API calls).
Expected result: each seat wins ~25% in expectation.

If symmetry holds:
  - No seat wins significantly more than the others
  - Aggregate win rate across all seats ≈ 25%
  - p-value from chi-squared test > 0.05

A failed calibration (any seat winning >35% or <15% consistently, or
p < 0.05) means BalancedPlayer has a seat-position bias or implementation
bug that must be fixed before running real LLM experiments.

Usage:
    python scripts/calibrate_balanced.py          # 500 games (default)
    python scripts/calibrate_balanced.py --games 1000
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.card import full_deck, deal
from engine.state import GameState
from engine.game import apply_action, _copy_state
from agents.archetypes import BalancedPlayer

MAX_TURNS = 600


def run_calibration_episode(seed: int) -> int:
    """Run one 4-player BalancedPlayer game. Returns winner seat index (0-3)."""
    n = 4
    rng = random.Random(seed)
    deck = full_deck()
    rng.shuffle(deck)
    hands = deal(deck, n)
    state = GameState.new(hands)

    agents = [BalancedPlayer(player_id=i, name=f"BP{i}") for i in range(n)]
    for agent in agents:
        agent.reset()

    def _public_event_for(event: dict, for_player: int) -> dict:
        if event.get("action", {}).get("type") == "challenge":
            return event
        actor = event["player"]
        if for_player == actor:
            return event
        return {
            "turn":         event["turn"],
            "player":       actor,
            "action":       {"type": "play"},
            "claimed_rank": event.get("claimed_rank"),
            "n_cards":      event.get("n_cards"),
        }

    while not state.is_terminal and state.turn < MAX_TURNS:
        player = state.current_player
        agent  = agents[player]
        obs    = state.public_observation(player)

        action = agent.choose_action(obs)
        if action.get("type") == "challenge":
            action = {"type": "play", "cards": [obs["my_hand"][0]], "claimed_rank": obs["current_rank"]}
        if action.get("type") == "play" and not (1 <= len(action.get("cards", [])) <= 4):
            action = {"type": "play", "cards": [obs["my_hand"][0]], "claimed_rank": obs["current_rank"]}

        new_state, play_event = apply_action(state, action)
        for pid, ag in enumerate(agents):
            ag.observe_event(_public_event_for(play_event, pid))
        state = new_state

        if state.is_terminal:
            break

        for i in range(1, n):
            challenger_id = (player + i) % n
            challenger    = agents[challenger_id]
            c_obs = state.public_observation(challenger_id)
            c_action = challenger.choose_action(c_obs)

            if c_action.get("type") == "challenge":
                challenge_state = _copy_state(state)
                challenge_state.current_player = challenger_id
                new_state2, ch_event = apply_action(challenge_state, {"type": "challenge"})
                for pid, ag in enumerate(agents):
                    ag.observe_event(_public_event_for(ch_event, pid))
                state = new_state2
                break

        if state.is_terminal:
            break

    return state.winner if state.is_terminal else min(range(n), key=lambda i: len(state.hands[i]))


def chi_squared_uniform(observed: list[int]) -> float:
    """Chi-squared p-value testing whether observed counts are uniformly distributed."""
    import math
    n = sum(observed)
    k = len(observed)
    expected = n / k
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    # Approximate p-value using chi-squared CDF (k-1 degrees of freedom)
    # Using Wilson-Hilferty cube-root approximation
    df = k - 1
    x = chi2 / df
    z = (x ** (1 / 3) - (1 - 2 / (9 * df))) / ((2 / (9 * df)) ** 0.5)
    # Convert z to p-value (one-tailed, upper): P(Z > z)
    p = 0.5 * math.erfc(z / 2 ** 0.5)
    return p


def main():
    parser = argparse.ArgumentParser(description="Calibrate BalancedPlayer symmetry")
    parser.add_argument("--games", type=int, default=500,
                        help="Number of calibration games (default: 500)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed (default: 0)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seat_wins = defaultdict(int)
    total_turns = []

    print(f"Running {args.games} calibration games (4 × BalancedPlayer, no LLM)…")
    for i in range(args.games):
        seed = rng.randint(0, 999_999)
        winner = run_calibration_episode(seed)
        seat_wins[winner] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.games} games done…")

    n_seats = 4
    print(f"\n{'='*55}")
    print(f"CALIBRATION RESULTS — BalancedPlayer (CHALLENGE_RATE={BalancedPlayer.CHALLENGE_RATE})")
    print(f"{'='*55}")
    print(f"  Total games: {args.games}")
    print(f"  Expected WR per seat: 25.0%")
    print()
    print(f"  {'Seat':<8} {'Wins':>6}  {'WR%':>6}  {'Δ from 25%':>10}")
    print(f"  {'-'*38}")

    observed = []
    max_delta = 0.0
    for seat in range(n_seats):
        wins = seat_wins[seat]
        wr   = wins / args.games * 100
        delta = wr - 25.0
        max_delta = max(max_delta, abs(delta))
        flag = "  ← ⚠ BIASED" if abs(delta) > 5.0 else ""
        print(f"  Seat {seat:<3}  {wins:>6}  {wr:>5.1f}%  {delta:>+8.1f}%{flag}")
        observed.append(wins)

    p_value = chi_squared_uniform(observed)
    print(f"\n  Chi-squared p-value: {p_value:.4f}  (>0.05 = symmetric  ✓)")
    print()

    PASS_THRESHOLD_DELTA = 5.0   # max allowed |seat WR - 25%|
    PASS_THRESHOLD_P     = 0.05  # min p-value to accept symmetry

    if max_delta <= PASS_THRESHOLD_DELTA and p_value >= PASS_THRESHOLD_P:
        print(f"  ✅  PASS — BalancedPlayer is symmetric.")
        print(f"      Safe to run LLM experiments with --opponent-pool symmetric.")
    else:
        reasons = []
        if max_delta > PASS_THRESHOLD_DELTA:
            reasons.append(f"max seat deviation {max_delta:.1f}% > {PASS_THRESHOLD_DELTA}%")
        if p_value < PASS_THRESHOLD_P:
            reasons.append(f"p-value {p_value:.4f} < {PASS_THRESHOLD_P}")
        print(f"  ❌  FAIL — Symmetry check failed: {'; '.join(reasons)}.")
        print(f"      DO NOT run LLM experiments until BalancedPlayer is fixed.")
        sys.exit(1)

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
