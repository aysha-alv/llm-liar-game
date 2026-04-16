"""
Phase 1 — Baseline Testing

Runs a tournament: LLM (multi-provider) vs rule-based archetypes.
Reports win rate, bluff/challenge stats, challenge rate, card pickup breakdown,
LLM fallback-play count, failure modes, and traces.

Usage:
    python scripts/phase1_baseline.py --games 50 --players 4
    python scripts/phase1_baseline.py --provider xai --model grok-3
    python scripts/phase1_baseline.py --models grok-3,openai:gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.game_state import GameState
from game.liar_game import LiarGame
from agents import (
    LLMAgent,
    MODEL_CONFIGS,
    TheSaint,
    ComebackCloser,
    GameTheorist,
    MrPathological,
    TheAccountant,
    TheCollector,
    parse_model_spec,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _default_agent_stats():
    return {
        "wins": 0,
        "games": 0,
        "bluffs": 0,
        "bluffs_caught": 0,
        "challenges": 0,
        "challenges_won": 0,
        "challenge_opportunities": 0,
        "cards_picked_up_bluff_caught": 0,
        "cards_picked_up_lost_challenge": 0,
        "fallback_plays": 0,
    }


# ---------------------------------------------------------------------------
# Failure mode detection
# ---------------------------------------------------------------------------


def detect_failure_modes(state: GameState, llm_id: int) -> dict:
    """
    Analyze a completed game for documented LLM failure modes.
    """
    failures = {
        "indexing_errors": 0,
        "computation_errors": 0,
        "control_flow_errors": 0,
        "hallucination": 0,
        "logic_following": 0,
    }

    llm_claims = [r for r in state.claim_history if r.player_id == llm_id]

    for claim in llm_claims:
        if claim.claimed_count > 4:
            failures["computation_errors"] += 1

        if claim.was_challenged and claim.challenge_result == "liar_caught":
            if claim.claimed_count > len(claim.actual_cards) + 2:
                failures["hallucination"] += 1

    return failures


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

ARCHETYPES = [
    ("TheSaint", TheSaint),
    ("ComebackCloser", ComebackCloser),
    ("GameTheorist", GameTheorist),
    ("MrPathological", MrPathological),
    ("TheAccountant", TheAccountant),
    ("TheCollector", TheCollector),
]


def run_tournament(
    num_games: int = 50,
    num_players: int = 4,
    verbose: bool = False,
    save_traces: bool = True,
    output_dir: Path | None = None,
    llm_provider: str = "xai",
    llm_model: str = "grok-3",
    llm_results_key: str = "LLM",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Run num_games with one LLM seat + (num_players-1) archetypes.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_dir = Path(__file__).parent.parent / "data" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    results: dict = defaultdict(_default_agent_stats)
    failure_totals: dict = defaultdict(int)
    all_game_summaries = []

    logger.info(
        "Starting Phase 1: %s games, %s players, LLM=%s/%s (%s)",
        num_games,
        num_players,
        llm_provider,
        llm_model,
        llm_results_key,
    )

    try:
        llm = LLMAgent(
            name=llm_results_key,
            provider=llm_provider,
            model=llm_model,
            base_url=base_url,
            api_key=api_key,
        )
    except ValueError as e:
        logger.error("Could not initialize LLMAgent: %s", e)
        sys.exit(1)

    for game_num in range(num_games):
        seed = random.randint(0, 999999)
        llm.fallback_play_count = 0

        opponent_classes = random.sample(ARCHETYPES, min(num_players - 1, len(ARCHETYPES)))
        opponents = [cls(name=f"{name}_{game_num}") for name, cls in opponent_classes]

        grok_pos = random.randint(0, num_players - 1)
        agents = opponents[:grok_pos] + [llm] + opponents[grok_pos:]
        agents = agents[:num_players]

        llm_id = agents.index(llm)

        game = LiarGame(agents=agents, seed=seed, verbose=verbose)
        winner_id, final_state = game.play()

        for agent in agents:
            pid = agents.index(agent)
            player = final_state.players[pid]
            aname = agent.name.split("_")[0]
            results[aname]["games"] += 1
            results[aname]["bluffs"] += player.bluffs_attempted
            results[aname]["bluffs_caught"] += player.bluffs_caught
            results[aname]["challenges"] += player.challenges_issued
            results[aname]["challenges_won"] += player.challenges_won
            results[aname]["cards_picked_up_bluff_caught"] += player.cards_picked_up_bluff_caught
            results[aname]["cards_picked_up_lost_challenge"] += player.cards_picked_up_lost_challenge
            opp = sum(1 for r in final_state.claim_history if r.player_id != pid)
            results[aname]["challenge_opportunities"] += opp
            if agent is llm:
                results[aname]["fallback_plays"] += llm.fallback_play_count

        if winner_id == llm_id:
            results[llm_results_key]["wins"] += 1
        else:
            winner_agent = agents[winner_id]
            wname = winner_agent.name.split("_")[0]
            results[wname]["wins"] += 1

        failures = detect_failure_modes(final_state, llm_id)
        for k, v in failures.items():
            failure_totals[k] += v

        llm_player = final_state.players[llm_id]
        ch_issued = llm_player.challenges_issued
        ch_won = llm_player.challenges_won
        ch_opp = sum(1 for r in final_state.claim_history if r.player_id != llm_id)

        game_summary = {
            "game_num": game_num,
            "seed": seed,
            "num_players": num_players,
            "llm_results_key": llm_results_key,
            "winner": agents[winner_id].name.split("_")[0],
            "llm_won": winner_id == llm_id,
            "turns": final_state.turn_number,
            "llm_hand_size": llm_player.hand_size,
            "llm_bluffs_attempted": llm_player.bluffs_attempted,
            "llm_bluffs_caught": llm_player.bluffs_caught,
            "llm_challenges_issued": ch_issued,
            "llm_challenges_won": ch_won,
            "llm_challenges_lost": ch_issued - ch_won,
            "llm_challenge_opportunities": ch_opp,
            "llm_cards_picked_up_bluff_caught": llm_player.cards_picked_up_bluff_caught,
            "llm_cards_picked_up_lost_challenge": llm_player.cards_picked_up_lost_challenge,
            "llm_fallback_plays": llm.fallback_play_count,
            "failure_modes": failures,
            "opponents": [a.name.split("_")[0] for a in agents if a is not llm],
        }
        all_game_summaries.append(game_summary)

        if (game_num + 1) % 10 == 0 or num_games <= 10:
            wr = results[llm_results_key]["wins"] / max(results[llm_results_key]["games"], 1) * 100
            logger.info("Game %s/%s | %s win rate: %.1f%%", game_num + 1, num_games, llm_results_key, wr)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_slug = llm_results_key.replace("/", "-").replace(" ", "_")

    if save_traces and llm.reasoning_traces:
        trace_file = traces_dir / f"llm_traces_{safe_slug}_{ts}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(llm.reasoning_traces, f, indent=2)
        logger.info("Saved %s reasoning traces → %s", len(llm.reasoning_traces), trace_file)

    results_file = output_dir / f"phase1_results_{safe_slug}_{ts}.json"
    final_results = {
        "config": {
            "num_games": num_games,
            "num_players": num_players,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_results_key": llm_results_key,
        },
        "agent_results": dict(results),
        "failure_modes": dict(failure_totals),
        "game_summaries": all_game_summaries,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    _print_summary(results, failure_totals, num_games, llm_results_key)

    return final_results


def _print_summary(results, failure_totals, num_games, llm_results_key: str):
    print("\n" + "=" * 72)
    print("PHASE 1 BASELINE RESULTS")
    print("=" * 72)
    print(
        f"\n{'Agent':<22} {'Win%':>7} {'BluffAcc':>9} {'ChalAcc':>8} "
        f"{'ChalRate':>9} {'FallPlays':>10}"
    )
    print("-" * 72)

    for agent_name, r in sorted(results.items(), key=lambda x: -x[1]["wins"]):
        g = r["games"]
        if g == 0:
            continue
        wr = r["wins"] / g * 100
        bluff_acc = (r["bluffs"] - r["bluffs_caught"]) / max(r["bluffs"], 1) * 100
        chal_acc = r["challenges_won"] / max(r["challenges"], 1) * 100
        chal_rate = r["challenges"] / max(r["challenge_opportunities"], 1) * 100
        fall = r.get("fallback_plays", 0)
        marker = f" ◄ {llm_results_key}" if agent_name == llm_results_key else ""
        print(
            f"{agent_name:<22} {wr:>6.1f}% {bluff_acc:>8.1f}% {chal_acc:>7.1f}% "
            f"{chal_rate:>8.1f}% {fall:>10}{marker}"
        )

    print(f"\n--- {llm_results_key} failure modes ---")
    for mode, count in failure_totals.items():
        per_game = count / max(num_games, 1)
        print(f"  {mode:<30}: {count:>4} total  ({per_game:.2f}/game)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline tournament")
    parser.add_argument("--games", type=int, default=20, help="Number of games to run")
    parser.add_argument("--players", type=int, default=4, help="Players per game (3-7)")
    parser.add_argument("--verbose", action="store_true", help="Log each turn")
    parser.add_argument("--no-save", action="store_true", help="Don't save traces")
    parser.add_argument(
        "--provider",
        type=str,
        default="xai",
        help="Default provider if model spec has no prefix (openai, xai, anthropic, google, local)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="grok-3",
        help="Single model id or preset name (see MODEL_CONFIGS in agents/llm_agent.py)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list; each item is a preset key or provider:model",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default=None,
        help="Label for results/traces (default: derived from model id)",
    )
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base URL (e.g. Ollama)")
    parser.add_argument("--api-key", type=str, default=None, help="Override API key for the LLM provider")
    args = parser.parse_args()

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    if args.models:
        specs = [s.strip() for s in args.models.split(",") if s.strip()]
    else:
        specs = [args.model.strip()]

    num_players = max(3, min(7, args.players))

    logger.info("MODEL_CONFIGS presets: %s", ", ".join(sorted(MODEL_CONFIGS.keys())))
    if len(specs) > 1:
        logger.info("Running %s model configurations sequentially.", len(specs))

    for run_idx, spec in enumerate(specs):
        prov, mod = parse_model_spec(spec, args.provider)
        label = args.llm_name or f"{prov}-{mod.replace('/', '-')}"
        if len(specs) > 1:
            label = f"{label}__run{run_idx}"

        run_tournament(
            num_games=args.games,
            num_players=num_players,
            verbose=args.verbose,
            save_traces=not args.no_save,
            llm_provider=prov,
            llm_model=mod,
            llm_results_key=label,
            base_url=args.base_url,
            api_key=args.api_key,
        )


if __name__ == "__main__":
    main()
