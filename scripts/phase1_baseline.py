"""
Phase 1 — Baseline Tournament

4-player games: 1 LLM seat + 3 randomly sampled archetypes.
After each play, all other players are polled for a challenge in clockwise order
(first challenger wins, rest skip — standard Liar rules).

Usage:
    python scripts/phase1_baseline.py --games 50 --players 4
    python scripts/phase1_baseline.py --provider xai --model grok-3
    python scripts/phase1_baseline.py --prompt-mode few_shot --games 100
    python scripts/phase1_baseline.py --models grok-3,gpt-4o
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.card import full_deck, deal
from engine.state import GameState
from engine.game import apply_action, _copy_state
from logging_.logger import EpisodeLogger
from agents import (
    LLMAgent, MODEL_CONFIGS, PROMPT_MODES, parse_model_spec,
    TheSaint, ComebackCloser, GameTheorist,
    MrPathological, TheAccountant, TheCollector,
    BalancedPlayer,
)
from evaluation.metrics import InformationTheoreticMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

MAX_TURNS = 600

ARCHETYPES = [
    ("TheSaint",       TheSaint),
    ("ComebackCloser", ComebackCloser),
    ("GameTheorist",   GameTheorist),
    ("MrPathological", MrPathological),
    ("TheAccountant",  TheAccountant),
    ("TheCollector",   TheCollector),
]

# Controlled pool: excludes TheSaint (48% WR) and GameTheorist (45% WR) — the two
# dominant agents whose presence suppresses every other player's win rate well below
# the 25% random baseline in a 4-player game. Used for the "Option A" controlled runs.
CONTROLLED_ARCHETYPES = [
    ("ComebackCloser", ComebackCloser),
    ("MrPathological", MrPathological),
    ("TheAccountant",  TheAccountant),
    ("TheCollector",   TheCollector),
]

# Symmetric pool: exactly 3 BalancedPlayer clones every game.
# Symmetry guarantees each clone wins 25% in expectation — any LLM deviation
# from 25% is a pure skill signal with no opponent-composition confound.
# Calibration (scripts/calibrate_balanced.py) verifies implementation symmetry
# before running LLM games.
SYMMETRIC_ARCHETYPES = [
    ("BalancedPlayer", BalancedPlayer),
    ("BalancedPlayer", BalancedPlayer),
    ("BalancedPlayer", BalancedPlayer),
]

OPPONENT_POOLS = {
    "full":       ARCHETYPES,
    "controlled": CONTROLLED_ARCHETYPES,
    "symmetric":  SYMMETRIC_ARCHETYPES,
}


# ---------------------------------------------------------------------------
# N-player episode runner
# ---------------------------------------------------------------------------

def run_episode(agents, seed: int, verbose: bool = False, max_turns: int = MAX_TURNS) -> EpisodeLogger:
    """
    Run one game with N agents.

    After each play, poll all OTHER players in clockwise order for a challenge.
    First player to challenge wins; rest skip.
    """
    n = len(agents)
    rng = random.Random(seed)
    deck = full_deck()
    rng.shuffle(deck)
    hands = deal(deck, n)
    state = GameState.new(hands)

    for agent in agents:
        agent.reset()

    ep_logger = EpisodeLogger(seed=seed)
    ep_logger.agents = [
        {"player_id": a.player_id, "name": a.name, "type": type(a).__name__}
        for a in agents
    ]

    def _public_event_for(event: dict, for_player: int) -> dict:
        """Filter event so observers can't see actual_cards from a play."""
        if event.get("action", {}).get("type") == "challenge":
            return event  # challenge reveal is public
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

    while not state.is_terminal and state.turn < max_turns:
        player = state.current_player
        agent  = agents[player]
        obs    = state.public_observation(player)

        # Current player PLAYS (no challenge option on your own turn)
        action = agent.choose_action(obs)
        # Ensure they play, not challenge (it's their turn to play)
        if action.get("type") == "challenge":
            logger.debug("Agent %s tried to challenge on their play turn — overriding", agent.name)
            action = {"type": "play", "cards": [obs["my_hand"][0]], "claimed_rank": obs["current_rank"]}
        # Guard: must play 1–4 valid cards
        if action.get("type") == "play" and not (1 <= len(action.get("cards", [])) <= 4):
            logger.warning("Agent %s returned invalid card count (%s) — using fallback",
                           agent.name, len(action.get("cards", [])))
            hand = obs["my_hand"]
            action = {"type": "play", "cards": [hand[0]], "claimed_rank": obs["current_rank"]}

        new_state, play_event = apply_action(state, action)
        belief = agent.report_belief(obs)

        # Broadcast play event to all agents
        for pid, ag in enumerate(agents):
            ag.observe_event(_public_event_for(play_event, pid))

        ep_logger.log_turn(
            turn=state.turn, player=player, action=action,
            observation_before=obs, event=play_event,
            hand_sizes_after=new_state.hand_sizes(),
            beliefs={str(player): belief} if belief else None,
        )

        if verbose:
            _print_turn(play_event, new_state)

        state = new_state
        if state.is_terminal:
            break

        # --- CHALLENGE PHASE: poll other players in clockwise order ---
        challenged = False
        for i in range(1, n):
            challenger_id = (player + i) % n
            challenger    = agents[challenger_id]
            c_obs = state.public_observation(challenger_id)

            c_action = challenger.choose_action(c_obs)

            if c_action.get("type") == "challenge":
                # Apply the challenge: temporarily set current_player to challenger
                challenge_state = _copy_state(state)
                challenge_state.current_player = challenger_id
                new_state2, ch_event = apply_action(challenge_state, {"type": "challenge"})
                ch_belief = challenger.report_belief(c_obs)

                for pid, ag in enumerate(agents):
                    ag.observe_event(_public_event_for(ch_event, pid))

                ep_logger.log_turn(
                    turn=state.turn, player=challenger_id,
                    action={"type": "challenge"},
                    observation_before=c_obs, event=ch_event,
                    hand_sizes_after=new_state2.hand_sizes(),
                    beliefs={str(challenger_id): ch_belief} if ch_belief else None,
                )

                if verbose:
                    _print_turn(ch_event, new_state2)

                state = new_state2
                challenged = True
                break
            # else: pass (they chose to play — we ignore the play action here,
            # it'll be their turn when the normal loop reaches them)

        if state.is_terminal:
            break

    winner = state.winner if state.is_terminal else min(range(n), key=lambda i: len(state.hands[i]))
    reason = "empty_hand" if state.is_terminal else "max_turns_reached"
    ep_logger.log_outcome(winner=winner, total_turns=state.turn, reason=reason)

    return ep_logger


def _print_turn(event: dict, state: GameState) -> None:
    t = event["turn"]
    p = event["player"]
    atype = event.get("action", {}).get("type")
    if atype == "play":
        flag = "(honest)" if event.get("honest") else "(BLUFF)"
        print(f"[T{t:03d}] P{p} plays {event['n_cards']}x {event['claimed_rank']} "
              f"{flag}  hands={state.hand_sizes()}")
    elif atype == "challenge":
        print(f"[T{t:03d}] P{p} CHALLENGES → {event.get('challenge_result')}  "
              f"pile→P{event.get('pile_goes_to')}  hands={state.hand_sizes()}")
    if "winner" in event:
        print(f"  *** P{event['winner']} wins! ***")


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

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
    prompt_mode: str = "cot",
    max_turns: int = MAX_TURNS,
    opponent_pool: str = "full",
) -> dict:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = Path(__file__).parent.parent / "data" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    if opponent_pool not in OPPONENT_POOLS:
        raise ValueError(f"opponent_pool must be one of {list(OPPONENT_POOLS)}, got {opponent_pool!r}")
    archetype_pool = OPPONENT_POOLS[opponent_pool]

    # Sanity check: need at least num_players-1 archetypes to fill the table
    n_opponents_needed = num_players - 1
    if len(archetype_pool) < n_opponents_needed:
        raise ValueError(
            f"opponent_pool={opponent_pool!r} has only {len(archetype_pool)} archetypes "
            f"but {n_opponents_needed} opponents are needed for a {num_players}-player game. "
            f"Lower --players or choose a larger pool."
        )

    it_metrics = InformationTheoreticMetrics()
    agent_stats: dict = defaultdict(lambda: defaultdict(int))
    all_game_summaries = []

    logger.info("Phase 1: %s games, %s players, model=%s/%s, prompt_mode=%s, opponent_pool=%s",
                num_games, num_players, llm_provider, llm_model, prompt_mode, opponent_pool)

    try:
        llm = LLMAgent(
            player_id=0,  # reassigned each game below
            name=llm_results_key,
            provider=llm_provider,
            model=llm_model,
            base_url=base_url,
            api_key=api_key,
            prompt_mode=prompt_mode,
        )
    except ValueError as e:
        logger.error("Could not initialize LLMAgent: %s", e)
        sys.exit(1)

    for game_num in range(num_games):
        seed = random.randint(0, 999_999)
        llm.reset()

        # Sample opponents from the chosen pool (without replacement per game)
        n_opponents = num_players - 1  # already validated above
        opp_specs = random.sample(archetype_pool, n_opponents)
        opponents = [cls(player_id=i + 1, name=f"{nm}") for i, (nm, cls) in enumerate(opp_specs)]

        # Randomise LLM seat
        llm_seat = random.randint(0, num_players - 1)
        agents_list = opponents[:]
        agents_list.insert(llm_seat, llm)
        agents_list = agents_list[:num_players]

        # Re-assign player IDs by position
        for idx, ag in enumerate(agents_list):
            ag.player_id = idx

        llm_id = llm_seat

        ep = run_episode(agents_list, seed=seed, verbose=verbose, max_turns=max_turns)
        log = ep.to_dict()
        winner_id = log["outcome"]["winner"]

        # Aggregate stats
        for ag in agents_list:
            aname = ag.name.split("_")[0]
            agent_stats[aname]["games"] += 1
        winner_name = agents_list[winner_id].name.split("_")[0]
        agent_stats[winner_name]["wins"] += 1
        logger.info("Game %s/%s done — winner: %s, turns: %s",
                    game_num + 1, num_games, agents_list[winner_id].name, log["outcome"]["total_turns"])

        # Per-game IT metrics for LLM
        it = it_metrics.analyze_episode(log, llm_id)

        # Count LLM bluffs / challenges from log
        llm_bluffs = sum(
            1 for t in log["turns"]
            if t["player"] == llm_id
            and t["action"]["type"] == "play"
            and not t["event"].get("honest", True)
        )
        llm_bluffs_caught = sum(
            1 for t in log["turns"]
            if t["action"]["type"] == "challenge"
            and t["event"].get("challenge_result") == "caught_bluffing"
            and t["event"].get("pile_goes_to") == llm_id
        )
        llm_challenges = sum(
            1 for t in log["turns"]
            if t["player"] == llm_id and t["action"]["type"] == "challenge"
        )
        llm_challenges_won = sum(
            1 for t in log["turns"]
            if t["player"] == llm_id
            and t["action"]["type"] == "challenge"
            and t["event"].get("challenge_result") == "caught_bluffing"
        )

        game_summary = {
            "game_num":                     game_num,
            "seed":                         seed,
            "num_players":                  num_players,
            "llm_results_key":              llm_results_key,
            "prompt_mode":                  prompt_mode,
            "winner":                       winner_name,
            "llm_won":                      winner_id == llm_id,
            "turns":                        log["outcome"]["total_turns"],
            "llm_bluffs_attempted":         llm_bluffs,
            "llm_bluffs_caught":            llm_bluffs_caught,
            "llm_challenges_issued":        llm_challenges,
            "llm_challenges_won":           llm_challenges_won,
            "llm_fallback_plays":           llm.fallback_play_count,
            "opponents":                    [a.name for a in agents_list if a is not llm],
            "mutual_information":           it["mutual_information"],
            "kl_divergence":                it["kl_divergence"],
            "belief_misalignment":          it["belief_misalignment"],
        }
        all_game_summaries.append(game_summary)

        if (game_num + 1) % 10 == 0 or num_games <= 10:
            wins = sum(1 for s in all_game_summaries if s["llm_won"])
            wr = wins / (game_num + 1) * 100
            logger.info("Game %s/%s | %s win rate: %.1f%%", game_num + 1, num_games, llm_results_key, wr)
            # Save partial results every 10 games so a crash loses at most 10 games
            _safe_slug = llm_results_key.replace("/", "-").replace(" ", "_")
            checkpoint_file = output_dir / f"phase1_checkpoint_{_safe_slug}.json"
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "config": {
                        "num_games": num_games, "num_players": num_players,
                        "llm_provider": llm_provider, "llm_model": llm_model,
                        "llm_results_key": llm_results_key, "prompt_mode": prompt_mode,
                        "opponent_pool": opponent_pool,
                        "opponent_pool_agents": [nm for nm, _ in archetype_pool],
                    },
                    "games_completed": game_num + 1,
                    "agent_stats": {k: dict(v) for k, v in agent_stats.items()},
                    "game_summaries": all_game_summaries,
                }, f, indent=2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_slug = llm_results_key.replace("/", "-").replace(" ", "_")
    # Remove checkpoint file now that the full run is saved
    checkpoint_file = output_dir / f"phase1_checkpoint_{safe_slug}.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Save traces
    if save_traces and llm.reasoning_traces:
        trace_file = traces_dir / f"llm_traces_{safe_slug}_{ts}.json"
        with open(trace_file, "w") as f:
            json.dump(llm.reasoning_traces, f, indent=2)
        logger.info("Saved %s traces → %s", len(llm.reasoning_traces), trace_file)

    final_results = {
        "config": {
            "num_games":       num_games,
            "num_players":     num_players,
            "llm_provider":    llm_provider,
            "llm_model":       llm_model,
            "llm_results_key": llm_results_key,
            "prompt_mode":     prompt_mode,
            "opponent_pool":   opponent_pool,
            "opponent_pool_agents": [nm for nm, _ in archetype_pool],
        },
        "agent_stats":    {k: dict(v) for k, v in agent_stats.items()},
        "game_summaries": all_game_summaries,
    }

    results_file = output_dir / f"phase1_results_{safe_slug}_{ts}.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    _print_summary(all_game_summaries, llm_results_key, num_games, opponent_pool)
    return final_results


def _print_summary(summaries: list, llm_key: str, num_games: int, opponent_pool: str = "full") -> None:
    wins    = sum(1 for s in summaries if s["llm_won"])
    bluffs  = sum(s["llm_bluffs_attempted"] for s in summaries)
    caught  = sum(s["llm_bluffs_caught"] for s in summaries)
    chals   = sum(s["llm_challenges_issued"] for s in summaries)
    chal_w  = sum(s["llm_challenges_won"] for s in summaries)
    mis     = [s["mutual_information"] for s in summaries]
    kls     = [s["kl_divergence"] for s in summaries]

    # Opponent win rates from summaries
    opp_wins: dict = defaultdict(lambda: {"wins": 0, "games": 0})
    for s in summaries:
        for opp in s["opponents"]:
            opp_wins[opp]["games"] += 1
        winner = s["winner"]
        if not s["llm_won"] and winner in opp_wins:
            opp_wins[winner]["wins"] += 1

    print(f"\n{'='*65}")
    print(f"PHASE 1 RESULTS — {llm_key}")
    print(f"  Opponent pool:    {opponent_pool}")
    print(f"{'='*65}")
    print(f"  Games:            {num_games}")
    print(f"  Win rate:         {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"  Bluff catch rate: {caught}/{bluffs} ({caught/max(bluffs,1)*100:.1f}%)")
    print(f"  Challenge acc:    {chal_w}/{chals} ({chal_w/max(chals,1)*100:.1f}%)")
    print(f"  Avg MI:           {sum(mis)/max(len(mis),1):.3f}")
    print(f"  Avg KL:           {sum(kls)/max(len(kls),1):.3f}")
    print(f"{'='*65}")
    print(f"  Opponent win rates this run:")
    for oname, s in sorted(opp_wins.items(), key=lambda x: -x[1]["wins"]/max(x[1]["games"],1)):
        wr = s["wins"] / s["games"] * 100 if s["games"] else 0
        print(f"    {oname:<22} {s['wins']:>2}/{s['games']:>2} ({wr:.1f}%)")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline tournament")
    parser.add_argument("--games",    type=int, default=20)
    parser.add_argument("--players",  type=int, default=4)
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--no-save",  action="store_true")
    parser.add_argument("--provider", type=str, default="xai")
    parser.add_argument("--model",    type=str, default="grok-3")
    parser.add_argument("--models",   type=str, default=None,
                        help="Comma-separated list of model specs")
    parser.add_argument("--llm-name", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key",  type=str, default=None)
    parser.add_argument("--prompt-mode", type=str, default="cot",
                        choices=list(PROMPT_MODES))
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS,
                        help="Turn cap per game (default 300; lower = faster/cheaper)")
    parser.add_argument("--opponent-pool", type=str, default="full",
                        choices=list(OPPONENT_POOLS),
                        help=(
                            "Which opponent pool to sample from. "
                            "'full' (default) uses all 6 archetypes including TheSaint and GameTheorist. "
                            "'controlled' excludes the two dominant agents (TheSaint WR=48%%, "
                            "GameTheorist WR=45%%) for a fair-field comparison."
                        ))
    args = parser.parse_args()

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    specs = [s.strip() for s in args.models.split(",")] if args.models else [args.model]
    num_players = max(3, min(7, args.players))

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
            prompt_mode=args.prompt_mode,
            max_turns=args.max_turns,
            opponent_pool=args.opponent_pool,
        )


if __name__ == "__main__":
    main()
