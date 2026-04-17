"""
Multi-Model Tournament

Runs all configured LLM models across all prompt modes in one command,
then writes a unified comparative results JSON.

Usage:
    # Full study: all 4 models × 3 prompt modes × 100 games
    python scripts/multi_model_tournament.py --games 100

    # Subset
    python scripts/multi_model_tournament.py --games 50 --models grok-3,gpt-4o --prompt-modes cot,few_shot

    # Smoke test
    python scripts/multi_model_tournament.py --games 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_agent import MODEL_CONFIGS, PROMPT_MODES, parse_model_spec
from scripts.phase1_baseline import run_tournament

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS       = ["grok-3", "gpt-4o", "claude-sonnet-4-5", "gemini-2.0-flash"]
DEFAULT_PROMPT_MODES = ["zero_shot", "cot", "few_shot"]


def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _extract_summary(result: dict, label: str) -> dict:
    summaries = result.get("game_summaries", [])
    wins   = sum(1 for s in summaries if s.get("llm_won"))
    games  = len(summaries)
    bluffs = sum(s.get("llm_bluffs_attempted", 0) for s in summaries)
    caught = sum(s.get("llm_bluffs_caught", 0) for s in summaries)
    chals  = sum(s.get("llm_challenges_issued", 0) for s in summaries)
    chal_w = sum(s.get("llm_challenges_won", 0) for s in summaries)
    mis = [s["mutual_information"] for s in summaries if "mutual_information" in s]
    kls = [s["kl_divergence"]      for s in summaries if "kl_divergence" in s]
    return {
        "win_rate":              wins / max(games, 1),
        "wins":                  wins,
        "games":                 games,
        "bluff_catch_rate":      caught / max(bluffs, 1),
        "challenge_accuracy":    chal_w / max(chals, 1),
        "avg_mutual_information": sum(mis) / len(mis) if mis else None,
        "avg_kl_divergence":     sum(kls) / len(kls) if kls else None,
    }


def _print_comparison_table(runs: list) -> None:
    print("\n" + "=" * 100)
    print("MULTI-MODEL TOURNAMENT RESULTS")
    print("=" * 100)
    print(f"\n{'Label':<42} {'Win%':>7} {'BluffCatch%':>12} {'ChalAcc%':>10} {'Avg MI':>8} {'Avg KL':>8}")
    print("-" * 100)
    for run in runs:
        if "error" in run:
            print(f"{run['label']:<42}  ERROR: {run['error']}")
            continue
        s = run["summary"]
        mi = f"{s['avg_mutual_information']:.3f}" if s["avg_mutual_information"] is not None else "  N/A"
        kl = f"{s['avg_kl_divergence']:.3f}"      if s["avg_kl_divergence"] is not None else "  N/A"
        print(f"{run['label']:<42} {s['win_rate']*100:>6.1f}% "
              f"{s['bluff_catch_rate']*100:>11.1f}% "
              f"{s['challenge_accuracy']*100:>9.1f}% "
              f"{mi:>8} {kl:>8}")
    print("=" * 100)


def run_multi_model_tournament(
    model_specs: list[str],
    prompt_modes: list[str],
    num_games: int = 100,
    num_players: int = 4,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> dict:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    total = len(model_specs) * len(prompt_modes)
    idx = 0

    for spec in model_specs:
        prov, mod = parse_model_spec(spec, "openai")
        for mode in prompt_modes:
            idx += 1
            label = f"{prov}-{mod.replace('/', '-')}_{mode}"
            logger.info("=== Run %s/%s: %s prompt_mode=%s ===", idx, total, label, mode)
            try:
                result = run_tournament(
                    num_games=num_games,
                    num_players=num_players,
                    verbose=verbose,
                    save_traces=True,
                    output_dir=output_dir,
                    llm_provider=prov,
                    llm_model=mod,
                    llm_results_key=label,
                    prompt_mode=mode,
                )
                all_runs.append({
                    "model_spec":  spec,
                    "provider":    prov,
                    "model":       mod,
                    "prompt_mode": mode,
                    "label":       label,
                    "summary":     _extract_summary(result, label),
                })
            except Exception as e:
                logger.error("Run failed %s / %s: %s", label, mode, e)
                all_runs.append({"model_spec": spec, "provider": prov, "model": mod,
                                  "prompt_mode": mode, "label": label, "error": str(e)})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined = {
        "study": "multi_model_pivot",
        "timestamp": ts,
        "config": {"num_games": num_games, "num_players": num_players,
                   "models": model_specs, "prompt_modes": prompt_modes},
        "runs": all_runs,
    }
    out = output_dir / f"multi_model_{ts}.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info("Combined results → %s", out)
    _print_comparison_table(all_runs)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",        type=int, default=100)
    parser.add_argument("--players",      type=int, default=4)
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--models",       type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--prompt-modes", type=str, default=",".join(DEFAULT_PROMPT_MODES))
    args = parser.parse_args()

    _load_env()

    model_specs = [s.strip() for s in args.models.split(",") if s.strip()]
    modes = [m.strip() for m in args.prompt_modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in PROMPT_MODES]
    if invalid:
        parser.error(f"Invalid prompt modes: {invalid}")

    logger.info("Study: %s models × %s modes × %s games = %s total games",
                len(model_specs), len(modes), args.games,
                len(model_specs) * len(modes) * args.games)
    run_multi_model_tournament(
        model_specs=model_specs,
        prompt_modes=modes,
        num_games=args.games,
        num_players=max(3, min(7, args.players)),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
