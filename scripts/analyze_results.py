"""
Results Analysis — Statistical Significance + Information-Theoretic Correlations

Loads one or more Phase 1 results JSON files (or a multi_model combined JSON)
and produces:

1. Bootstrap 95% confidence intervals on win rates per model/mode
2. Pairwise Mann-Whitney U tests for win rate comparisons
3. Pearson correlations:
     - mutual_information vs win (per game)
     - kl_divergence vs bluff_success (1 - bluff_catch_rate) per game
     - challenge_accuracy vs win per game
4. Summary table + optional JSON export

Usage:
    # Analyze a single Phase 1 results file
    python scripts/analyze_results.py data/results/phase1_results_*.json

    # Analyze a combined multi-model file
    python scripts/analyze_results.py data/results/multi_model_*.json

    # Save analysis to JSON
    python scripts/analyze_results.py data/results/multi_model_*.json --output analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Stats utilities (stdlib only — no scipy dependency)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _bootstrap_ci(
    values: List[float],
    stat_fn=_mean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lower_ci, upper_ci) via percentile bootstrap."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = stat_fn(values)
    if n == 1:
        return point, point, point
    boots = [stat_fn([rng.choice(values) for _ in range(n)]) for _ in range(n_resamples)]
    boots.sort()
    lo = boots[int(alpha / 2 * n_resamples)]
    hi = boots[int((1 - alpha / 2) * n_resamples)]
    return point, lo, hi


def _mann_whitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Two-sided Mann-Whitney U test (exact rank-sum method).
    Returns (U statistic, approximate two-tailed p-value).
    Uses normal approximation, valid for n > ~20.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, 1.0
    combined = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(ranks[k] for k, (_, grp) in enumerate(combined) if grp == 0)
    u1 = r1 - na * (na + 1) / 2
    u2 = na * nb - u1
    u = min(u1, u2)
    mean_u = na * nb / 2
    std_u = math.sqrt(na * nb * (na + nb + 1) / 12)
    if std_u == 0:
        return u, 1.0
    z = (u - mean_u) / std_u
    # Two-tailed p via normal CDF approximation
    p = 2 * _norm_cdf(-abs(z))
    return u, p


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _pearson(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Pearson r and approximate p-value (t-distribution)."""
    n = len(xs)
    if n < 3:
        return 0.0, 1.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0, 1.0
    r = num / (dx * dy)
    r = max(-1.0, min(1.0, r))
    t = r * math.sqrt(n - 2) / math.sqrt(max(1 - r ** 2, 1e-10))
    # Approximate p via normal (valid for n > 30)
    p = 2 * _norm_cdf(-abs(t))
    return r, p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_game_summaries(path: Path) -> Dict[str, List[dict]]:
    """
    Returns {label: [game_summary, ...]} from a Phase 1 or multi-model JSON.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    groups: Dict[str, List[dict]] = {}

    if "runs" in data:
        # multi_model combined format
        for run in data["runs"]:
            if "error" in run:
                continue
            label = run["label"]
            # Load the individual results file that was saved alongside
            # Fall back to reconstructing from the combined summary
            summaries = run.get("game_summaries")
            if summaries is None:
                # The combined file only has the summary; full game_summaries
                # are in the individual phase1_results_*.json files.
                # Try to find them.
                results_dir = path.parent
                slug = label.replace("/", "-").replace(" ", "_")
                candidates = sorted(results_dir.glob(f"phase1_results_{slug}_*.json"))
                if candidates:
                    with open(candidates[-1], encoding="utf-8") as f2:
                        individual = json.load(f2)
                    summaries = individual.get("game_summaries", [])
                else:
                    continue
            groups[label] = summaries
    else:
        # Single Phase 1 results file
        label = data["config"].get("llm_results_key", "LLM")
        groups[label] = data.get("game_summaries", [])

    return groups


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(groups: Dict[str, List[dict]]) -> dict:
    analysis = {}

    for label, summaries in groups.items():
        if not summaries:
            continue

        won = [1.0 if s["llm_won"] else 0.0 for s in summaries]
        mis = [s["mutual_information"] for s in summaries if "mutual_information" in s]
        kls = [s["kl_divergence"] for s in summaries if "kl_divergence" in s]

        bluff_success = []
        for s in summaries:
            attempted = s.get("llm_bluffs_attempted", 0)
            caught = s.get("llm_bluffs_caught", 0)
            if attempted > 0:
                bluff_success.append(1.0 - caught / attempted)

        chal_acc = []
        for s in summaries:
            issued = s.get("llm_challenges_issued", 0)
            won_c = s.get("llm_challenges_won", 0)
            if issued > 0:
                chal_acc.append(won_c / issued)

        wr_est, wr_lo, wr_hi = _bootstrap_ci(won)

        # Correlations (align arrays by index — all per-game)
        mi_vs_win = _pearson(mis, won[:len(mis)]) if len(mis) >= 3 else (None, None)
        kl_vs_bluff = _pearson(kls, bluff_success[:len(kls)]) if len(kls) >= 3 and bluff_success else (None, None)
        chal_vs_win = _pearson(chal_acc, won[:len(chal_acc)]) if len(chal_acc) >= 3 else (None, None)

        analysis[label] = {
            "n_games": len(summaries),
            "win_rate": wr_est,
            "win_rate_ci_95": [wr_lo, wr_hi],
            "avg_mi": _mean(mis) if mis else None,
            "avg_kl": _mean(kls) if kls else None,
            "avg_bluff_success_rate": _mean(bluff_success) if bluff_success else None,
            "avg_challenge_accuracy": _mean(chal_acc) if chal_acc else None,
            "correlations": {
                "mi_vs_win": {"r": mi_vs_win[0], "p": mi_vs_win[1]},
                "kl_vs_bluff_success": {"r": kl_vs_bluff[0], "p": kl_vs_bluff[1]},
                "challenge_acc_vs_win": {"r": chal_vs_win[0], "p": chal_vs_win[1]},
            },
        }

    # Pairwise Mann-Whitney on win rates
    pairwise = {}
    labels = list(groups.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            a_won = [1.0 if s["llm_won"] else 0.0 for s in groups[la]]
            b_won = [1.0 if s["llm_won"] else 0.0 for s in groups[lb]]
            u, p = _mann_whitney_u(a_won, b_won)
            pair_key = f"{la} vs {lb}"
            pairwise[pair_key] = {"U": u, "p_value": p, "significant_p05": p < 0.05}

    return {"per_model": analysis, "pairwise_comparisons": pairwise}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_analysis(result: dict) -> None:
    pm = result["per_model"]
    pw = result["pairwise_comparisons"]

    print("\n" + "=" * 110)
    print("RESULTS ANALYSIS")
    print("=" * 110)
    print(
        f"\n{'Label':<42} {'Win%':>7} {'95% CI':>16} {'Avg MI':>8} "
        f"{'Avg KL':>8} {'BluffSucc%':>11} {'ChalAcc%':>9}"
    )
    print("-" * 110)
    for label, a in sorted(pm.items(), key=lambda x: -x[1]["win_rate"]):
        ci = a["win_rate_ci_95"]
        ci_str = f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"
        mi_str = f"{a['avg_mi']:.3f}" if a["avg_mi"] is not None else "  N/A"
        kl_str = f"{a['avg_kl']:.3f}" if a["avg_kl"] is not None else "  N/A"
        bs_str = f"{a['avg_bluff_success_rate']*100:.1f}%" if a["avg_bluff_success_rate"] is not None else "  N/A"
        ca_str = f"{a['avg_challenge_accuracy']*100:.1f}%" if a["avg_challenge_accuracy"] is not None else "  N/A"
        print(
            f"{label:<42} {a['win_rate']*100:>6.1f}% {ci_str:>16} "
            f"{mi_str:>8} {kl_str:>8} {bs_str:>11} {ca_str:>9}"
        )

    print("\n--- Correlations (per-game, Pearson r / p-value) ---")
    print(f"\n{'Label':<42} {'MI→Win r':>10} {'KL→Bluff r':>12} {'Chal→Win r':>12}")
    print("-" * 80)
    for label, a in pm.items():
        corr = a["correlations"]
        def _fmt(d):
            if d["r"] is None:
                return "    N/A"
            sig = "*" if d["p"] is not None and d["p"] < 0.05 else " "
            return f"{d['r']:+.3f}{sig} (p={d['p']:.3f})"
        print(
            f"{label:<42} {_fmt(corr['mi_vs_win']):>18} "
            f"{_fmt(corr['kl_vs_bluff_success']):>18} "
            f"{_fmt(corr['challenge_acc_vs_win']):>18}"
        )

    if pw:
        print("\n--- Pairwise Mann-Whitney U (win rates) ---")
        print(f"\n{'Comparison':<60} {'U':>10} {'p-value':>10} {'Sig p<.05':>10}")
        print("-" * 95)
        for pair, res in pw.items():
            sig = "YES" if res["significant_p05"] else "no"
            print(f"{pair:<60} {res['U']:>10.1f} {res['p_value']:>10.4f} {sig:>10}")

    print("=" * 110)
    print("\nNotes:")
    print("  MI→Win:    lower MI = better bluffer; expect NEGATIVE correlation with win rate")
    print("  KL→Bluff:  higher KL = more deceptive; expect POSITIVE correlation with bluff success")
    print("  Chal→Win:  higher challenge accuracy = better reading of opponents; expect POSITIVE")
    print("  * = p < 0.05")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze tournament results: stats + correlations")
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more Phase 1 or multi_model results JSON files",
    )
    parser.add_argument("--output", type=Path, default=None, help="Save analysis JSON to this path")
    args = parser.parse_args()

    all_groups: Dict[str, list] = {}
    for path in args.files:
        if not path.exists():
            print(f"WARNING: file not found: {path}", file=sys.stderr)
            continue
        groups = _load_game_summaries(path)
        for label, summaries in groups.items():
            if label in all_groups:
                all_groups[label].extend(summaries)
            else:
                all_groups[label] = summaries

    if not all_groups:
        print("No game summaries found.", file=sys.stderr)
        sys.exit(1)

    result = analyze(all_groups)
    print_analysis(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nAnalysis saved → {args.output}")


if __name__ == "__main__":
    main()
