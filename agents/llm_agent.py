"""
Multi-provider LLM agent for Liar (Cheat/Bullshit).

Implements the new engine interface: choose_action(obs) -> action dict.
Supports OpenAI-compatible APIs (OpenAI, xAI/Grok, Ollama), Anthropic, and Google Gemini.

Uses structured SYSTEM_PROMPT + PLAY_PROMPT_TEMPLATE + CHALLENGE_PROMPT_TEMPLATE.
Deck is 108 cards (2 standard decks + 4 Jokers): max 8 honest cards per rank per cycle.

Prompt modes (all use the same structured templates — mode controls the SYSTEM_PROMPT):
  zero_shot  — rules only, no strategic guidance
  cot        — rules + strategy + step-by-step reasoning hints (default)
  few_shot   — rules + strategy + worked examples
"""
from __future__ import annotations
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .base import BaseAgent
from engine.card import Card, JOKER_RANK, RANKS

logger = logging.getLogger(__name__)

PROMPT_MODES = {"zero_shot", "cot", "few_shot"}

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "grok-3":            {"provider": "xai",       "model": "grok-3"},
    "gpt-4o":            {"provider": "openai",    "model": "gpt-4o"},
    "gpt-4o-mini":       {"provider": "openai",    "model": "gpt-4o-mini"},
    "claude-sonnet-4-5": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
    "gemini-2.0-flash":  {"provider": "google",    "model": "gemini-2.0-flash"},
}

# ---------------------------------------------------------------------------
# System prompts (one per mode)
# ---------------------------------------------------------------------------

_SYSTEM_ZERO_SHOT = """You are an expert player of the card game Liar (also called Cheat or Bullshit).
A 108-card deck (two standard decks + 4 Jokers) is dealt among all players. Your goal: be the FIRST to empty your hand.

═══ RULES ═══

1. TURNS: On your turn you MUST place 1–4 cards face-down and CLAIM they are the current required rank.
2. VISIBILITY: All players see HOW MANY cards you placed. Nobody sees the faces until a challenge.
3. LYING: You may lie about the rank of the cards you play. Your claimed_count must equal the number of cards you actually place.
4. RANK CYCLE: The required rank advances each turn: Ace → Two → Three → … → King → Ace → Two → …
   Each full Ace-through-King pass is one "rank cycle."
5. CHALLENGES: After any play, any OTHER player may call "Bullshit":
   • If the player LIED (wrong rank) → the liar picks up the ENTIRE discard pile.
   • If the player was HONEST → the challenger picks up the ENTIRE discard pile.
   After a challenge the rank still advances and play continues.
6. EIGHT-CARD RULE: Two decks means 8 cards of each rank exist.
   Within a SINGLE rank cycle, at most 8 of any rank can truthfully be played.
   If total claims for a rank EXCEED 8 within the same cycle, someone MUST be lying.
   (Jokers count as wildcards and are always honest.)

═══ CARD FORMAT ═══

Cards are written as RankSuit with Unicode suit symbols.
Examples: Ace♥  Two♦  Ten♣  King♠  Queen♥  Jack♦  Three♠  Seven♣  Joker★
ONLY use cards that appear in "My hand" below. Do NOT invent cards.

═══ OUTPUT FORMAT ═══

Respond with ONLY a single JSON object. No markdown, no commentary, no extra text.

When it is YOUR TURN TO PLAY:
{
  "action": "play",
  "cards": ["Ace♥", "Ace♣"],
  "claimed_count": 2,
  "reasoning": "I have 2 Aces, so I play them both honestly."
}

RULES for the play action:
- "cards" must list ONLY cards currently in your hand (exact RankSuit strings).
- "claimed_count" must EQUAL the number of cards in "cards" (1–4).
- You may play cards of ANY rank, but you are CLAIMING they match the required rank.

When you are DECIDING WHETHER TO CHALLENGE another player's play:
{
  "action": "challenge",
  "reasoning": "Total Aces claimed this cycle is 9, exceeding the 8-card maximum."
}
or
{
  "action": "pass_challenge",
  "reasoning": "Only 2 Aces claimed so far, plausible. Pile is large, too risky."
}

IMPORTANT: Output ONLY the JSON object. No other text before or after."""

_SYSTEM_COT = """You are an expert player of the card game Liar (also called Cheat or Bullshit).
A 108-card deck (two standard decks + 4 Jokers) is dealt among all players. Your goal: be the FIRST to empty your hand.

═══ RULES ═══

1. TURNS: On your turn you MUST place 1–4 cards face-down and CLAIM they are the current required rank.
2. VISIBILITY: All players see HOW MANY cards you placed. Nobody sees the faces until a challenge.
3. LYING: You may lie about the rank of the cards you play. Your claimed_count must equal the number of cards you actually place.
4. RANK CYCLE: The required rank advances each turn: Ace → Two → Three → … → King → Ace → Two → …
   Each full Ace-through-King pass is one "rank cycle."
5. CHALLENGES: After any play, any OTHER player may call "Bullshit":
   • If the player LIED (wrong rank) → the liar picks up the ENTIRE discard pile.
   • If the player was HONEST → the challenger picks up the ENTIRE discard pile.
   After a challenge the rank still advances and play continues.
6. EIGHT-CARD RULE: Two decks means 8 cards of each rank exist.
   Within a SINGLE rank cycle, at most 8 of any rank can truthfully be played.
   If total claims for a rank EXCEED 8 within the same cycle, someone MUST be lying.
   (Jokers count as wildcards and are always honest.)

═══ CARD FORMAT ═══

Cards are written as RankSuit with Unicode suit symbols.
Examples: Ace♥  Two♦  Ten♣  King♠  Queen♥  Jack♦  Three♠  Seven♣  Joker★
ONLY use cards that appear in "My hand" below. Do NOT invent cards.

═══ STRATEGY GUIDELINES ═══

• PLAY HONESTLY when you have matching cards — it is risk-free progress toward emptying your hand.
• When you MUST bluff, play as FEW cards as possible (1 is safest). Pick cards from ranks you have many of (expendable).
• CHALLENGE only when you are HIGHLY confident AND the discard pile is small enough that losing the challenge is survivable.
  Winning a challenge does NOT reduce YOUR hand. Losing means YOU pick up the entire pile.
• A player's "caught-lie frequency" is noisy — it only counts CAUGHT lies, not undetected ones.
  A low frequency does NOT mean they are honest; a moderate one does NOT mean this specific play is a lie.
• ALWAYS check: do the total claims for this rank in the current cycle exceed 8? If yes, someone is certainly lying.

═══ OUTPUT FORMAT ═══

Respond with ONLY a single JSON object. No markdown, no commentary, no extra text.

When it is YOUR TURN TO PLAY:
{
  "action": "play",
  "cards": ["Ace♥", "Ace♣"],
  "claimed_count": 2,
  "reasoning": "I have 2 Aces, so I play them both honestly."
}

RULES for the play action:
- "cards" must list ONLY cards currently in your hand (exact RankSuit strings).
- "claimed_count" must EQUAL the number of cards in "cards" (1–4).
- You may play cards of ANY rank, but you are CLAIMING they match the required rank.

When you are DECIDING WHETHER TO CHALLENGE another player's play:
{
  "action": "challenge",
  "reasoning": "Total Aces claimed this cycle is 9, exceeding the 8-card maximum."
}
or
{
  "action": "pass_challenge",
  "reasoning": "Only 2 Aces claimed so far, plausible. Pile is large, too risky."
}

IMPORTANT: Output ONLY the JSON object. No other text before or after."""

_SYSTEM_FEW_SHOT = """You are an expert player of the card game Liar (also called Cheat or Bullshit).
A 108-card deck (two standard decks + 4 Jokers) is dealt among all players. Your goal: be the FIRST to empty your hand.

═══ RULES ═══

1. TURNS: On your turn you MUST place 1–4 cards face-down and CLAIM they are the current required rank.
2. VISIBILITY: All players see HOW MANY cards you placed. Nobody sees the faces until a challenge.
3. LYING: You may lie about the rank of the cards you play. Your claimed_count must equal the number of cards you actually place.
4. RANK CYCLE: The required rank advances each turn: Ace → Two → Three → … → King → Ace → Two → …
   Each full Ace-through-King pass is one "rank cycle."
5. CHALLENGES: After any play, any OTHER player may call "Bullshit":
   • If the player LIED (wrong rank) → the liar picks up the ENTIRE discard pile.
   • If the player was HONEST → the challenger picks up the ENTIRE discard pile.
   After a challenge the rank still advances and play continues.
6. EIGHT-CARD RULE: Two decks means 8 cards of each rank exist.
   Within a SINGLE rank cycle, at most 8 of any rank can truthfully be played.
   If total claims for a rank EXCEED 8 within the same cycle, someone MUST be lying.
   (Jokers count as wildcards and are always honest.)

═══ CARD FORMAT ═══

Cards are written as RankSuit with Unicode suit symbols.
Examples: Ace♥  Two♦  Ten♣  King♠  Queen♥  Jack♦  Three♠  Seven♣  Joker★
ONLY use cards that appear in "My hand" below. Do NOT invent cards.

═══ STRATEGY GUIDELINES ═══

• PLAY HONESTLY when you have matching cards — it is risk-free progress toward emptying your hand.
• When you MUST bluff, play as FEW cards as possible (1 is safest). Pick cards from ranks you have many of (expendable).
• CHALLENGE only when you are HIGHLY confident AND the discard pile is small enough that losing the challenge is survivable.
  Winning a challenge does NOT reduce YOUR hand. Losing means YOU pick up the entire pile.
• A player's "caught-lie frequency" is noisy — it only counts CAUGHT lies, not undetected ones.
• ALWAYS check: do the total claims for this rank in the current cycle exceed 8? If yes, someone is certainly lying.

═══ EXAMPLES ═══

Example 1 — Honest play (have matching cards):
Rank=King. My hand has King♠, King♥, Three♦. → Play both Kings honestly.
{"action": "play", "cards": ["King♠","King♥"], "claimed_count": 2, "reasoning": "Have 2 Kings — play honestly, risk-free."}

Example 2 — Minimal bluff (no matching cards):
Rank=Seven. My hand has no Sevens. 18 cards in hand. → Bluff 1 expendable card.
{"action": "play", "cards": ["Two♣"], "claimed_count": 1, "reasoning": "No Sevens. Play 1 expendable card, claim 1 to minimise suspicion."}

Example 3 — Pass challenge (pile too large to risk):
Last claim: 2 Aces, pile=14 cards. Opponent lie freq=0.10. My hand=8 cards.
{"action": "pass_challenge", "reasoning": "Losing adds 14 cards. Low observed lie rate. Not worth the risk."}

Example 4 — Challenge (mathematically certain):
Total Aces claimed this cycle: 9 (exceeds 8 max for 2 decks). Pile=4.
{"action": "challenge", "reasoning": "9 Aces claimed this cycle but only 8 exist. Certain lie."}

═══ OUTPUT FORMAT ═══

Respond with ONLY a single JSON object. No markdown, no commentary, no extra text.

When it is YOUR TURN TO PLAY:
{"action": "play", "cards": ["Ace♥","Ace♣"], "claimed_count": 2, "reasoning": "..."}

When deciding whether to CHALLENGE:
{"action": "challenge", "reasoning": "..."} or {"action": "pass_challenge", "reasoning": "..."}

IMPORTANT: Output ONLY the JSON object. No other text before or after."""


def _build_system_prompt(mode: str) -> str:
    if mode == "zero_shot":
        return _SYSTEM_ZERO_SHOT
    if mode == "few_shot":
        return _SYSTEM_FEW_SHOT
    return _SYSTEM_COT  # cot (default)


# ---------------------------------------------------------------------------
# Prompt templates (adapted from prompt_1.md for 108-card deck)
# ---------------------------------------------------------------------------

PLAY_PROMPT_TEMPLATE = """
═══ GAME STATE (Turn {turn_number}) ═══

Required rank this turn: {current_rank}
Rank cycle: {rank_cycle} (Ace→King round #{rank_cycle})

My hand ({hand_size} cards): {hand_list}
  → Matching cards ({current_rank}): {matching_count}

Other players' hand sizes:
{hand_sizes}

Discard pile: {discard_pile_size} cards

Caught-lie frequencies (fraction of turns where each player was CAUGHT lying):
{lie_frequencies}

Recent history (last 8 turns):
{claim_history}

═══ YOUR TURN TO PLAY ═══
Step 1: Check how many {current_rank}s you have in your hand.
Step 2: If you have matching cards, play them honestly (safest).
Step 3: If you must bluff, pick 1 expendable card from a rank you hold many of.
Step 4: Set claimed_count equal to the number of cards you list.
Respond with your JSON action.
"""

CHALLENGE_PROMPT_TEMPLATE = """
═══ GAME STATE (Turn {turn_number}) ═══

Required rank this turn: {current_rank}
Rank cycle: {rank_cycle} (Ace→King round #{rank_cycle})

My hand ({hand_size} cards): {hand_list}
  → My {current_rank}s: {matching_count}

Other players' hand sizes:
{hand_sizes}

Discard pile: {discard_pile_size} cards

Caught-lie frequencies:
{lie_frequencies}

Recent history (last 8 turns):
{claim_history}

═══ CHALLENGE DECISION ═══
Player {claimer_name} (ID={claimer_id}) just placed {actual_cards_placed} card(s) face-down and claimed {claimed_count}x {current_rank}.

Key facts for your decision:
  • {claimer_name}'s caught-lie frequency: {claimer_lie_freq:.2f}
  • Total {current_rank}s claimed THIS rank cycle (including this play): {total_rank_claimed} / 8 max
  • You hold {matching_count}x {current_rank} yourself — subtract from 8 to see how many could be held by others
  • Discard pile if you lose the challenge: {discard_pile_size} cards added to your hand

Decision framework:
  1. If total_rank_claimed > 8 → someone is CERTAINLY lying. Challenge is safe.
  2. If you hold many {current_rank}s yourself, fewer are left for others → claim less plausible.
  3. If pile is large → losing the challenge is devastating. Be cautious.
  4. Caught-lie frequency is a weak signal. Use it alongside other evidence, not alone.

Respond with "challenge" or "pass_challenge" as your JSON action.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_model_spec(spec: str, default_provider: str) -> Tuple[str, str]:
    spec = spec.strip()
    if ":" in spec:
        prov, mod = spec.split(":", 1)
        return prov.strip().lower(), mod.strip()
    if spec in MODEL_CONFIGS:
        c = MODEL_CONFIGS[spec]
        return c["provider"], c["model"]
    return default_provider, spec


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------

class LLMAgent(BaseAgent):
    """LLM player with pluggable provider and prompt-mode ablation."""

    _ENV_KEYS = {
        "xai":       "XAI_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "local":     "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google":    "GOOGLE_API_KEY",
    }

    def __init__(
        self,
        player_id: int,
        name: str = "LLM",
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_mode: str = "cot",
    ):
        super().__init__(player_id, name)
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
        self.provider = provider.lower().strip()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_url = base_url
        self._api_key = api_key
        self.prompt_mode = prompt_mode
        self._system_prompt = _build_system_prompt(prompt_mode)
        self._openai_client: Optional[OpenAI] = None

        # Runtime state (reset each episode)
        self.reasoning_traces: List[dict] = []
        self.fallback_play_count: int = 0
        self._last_belief: Optional[dict] = None

        # History tracking for prompt context
        self._claim_history: List[dict] = []      # recent claims for prompt
        self._lie_counts: Dict[int, int] = defaultdict(int)
        self._turn_counts: Dict[int, int] = defaultdict(int)
        self._rank_claimed_this_cycle: Dict[str, int] = defaultdict(int)
        self._rank_cycle: int = 1
        self._last_rank_seen: Optional[str] = None
        self._agent_names: Dict[int, str] = {}    # populated from episode logger agents list

    def reset(self) -> None:
        self.reasoning_traces = []
        self.fallback_play_count = 0
        self._last_belief = None
        self._claim_history = []
        self._lie_counts = defaultdict(int)
        self._turn_counts = defaultdict(int)
        self._rank_claimed_this_cycle = defaultdict(int)
        self._rank_cycle = 1
        self._last_rank_seen = None
        self._agent_names = {}

    def set_agent_names(self, names: Dict[int, str]) -> None:
        """Optional: call this before a game to give the agent opponent names."""
        self._agent_names = names

    def observe_event(self, event: Dict[str, Any]) -> None:
        atype = event.get("action", {}).get("type")

        if atype == "play" and "claimed_rank" in event:
            rank = event["claimed_rank"]
            n    = event.get("n_cards", 1)
            pid  = event["player"]

            # Track rank cycle: new cycle when rank resets A after K
            if self._last_rank_seen == "K" and rank == "A":
                self._rank_cycle += 1
                self._rank_claimed_this_cycle = defaultdict(int)
            self._last_rank_seen = rank
            self._rank_claimed_this_cycle[rank] += n
            self._turn_counts[pid] += 1

            self._claim_history.append({
                "turn":         event.get("turn", 0),
                "player":       pid,
                "player_name":  self._agent_names.get(pid, f"P{pid}"),
                "claimed_rank": rank,
                "n_cards":      n,
                "challenged":   False,
                "result":       None,
            })
            # Keep last 20 entries
            self._claim_history = self._claim_history[-20:]

        elif atype == "challenge":
            result = event.get("challenge_result", "")
            if result == "caught_bluffing":
                caught_pid = event.get("pile_goes_to")
                if caught_pid is not None:
                    self._lie_counts[caught_pid] += 1
            # Mark the last claim as challenged
            if self._claim_history:
                self._claim_history[-1]["challenged"] = True
                self._claim_history[-1]["result"] = result
            # Pile cleared — reset rank claims this cycle
            self._rank_claimed_this_cycle = defaultdict(int)

    def report_belief(self, obs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._last_belief

    # ------------------------------------------------------------------
    # Main action method
    # ------------------------------------------------------------------

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        lc = obs.get("last_claim")
        is_challenge_decision = (lc is not None and lc["player_id"] != self.player_id)

        if is_challenge_decision:
            prompt = self._build_challenge_prompt(obs)
        else:
            prompt = self._build_play_prompt(obs)

        response = self._call_llm(prompt)

        if response is None:
            self.fallback_play_count += 1
            return self._fallback_action(obs)

        self._record_trace(obs, response, "challenge" if is_challenge_decision else "play")

        try:
            return self._parse_response(response, obs, is_challenge_decision)
        except Exception as e:
            logger.warning("Failed to parse LLM response: %s — fallback", e)
            self.fallback_play_count += 1
            return self._fallback_action(obs)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _fmt_hand_sizes(self, obs: Dict[str, Any]) -> str:
        sizes = obs.get("opponent_hand_sizes", {})
        if not sizes:
            return "  (no opponents)"
        return "\n".join(
            f"  Player {pid} ({self._agent_names.get(pid, f'P{pid}')}): {sz} cards"
            for pid, sz in sorted(sizes.items())
        )

    def _fmt_lie_frequencies(self, obs: Dict[str, Any]) -> str:
        sizes = obs.get("opponent_hand_sizes", {})
        lines = []
        for pid in sorted(sizes.keys()):
            turns = max(self._turn_counts.get(pid, 1), 1)
            freq  = self._lie_counts.get(pid, 0) / turns
            name  = self._agent_names.get(pid, f"P{pid}")
            lines.append(f"  Player {pid} ({name}): {freq:.2f}")
        return "\n".join(lines) if lines else "  (no data yet)"

    def _fmt_claim_history(self) -> str:
        recent = self._claim_history[-8:]
        if not recent:
            return "  (no claims yet)"
        lines = []
        for r in recent:
            name = r["player_name"]
            ch = ""
            if r["challenged"]:
                ch = f" → CHALLENGED ({r['result']})"
            lines.append(
                f"  Turn {r['turn']}: {name} placed {r['n_cards']} card(s), "
                f"claimed {r['n_cards']}x {r['claimed_rank']}{ch}"
            )
        return "\n".join(lines)

    def _build_play_prompt(self, obs: Dict[str, Any]) -> str:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        matching = sum(1 for c in hand if c.rank == rank or c.rank == JOKER_RANK)
        return PLAY_PROMPT_TEMPLATE.format(
            turn_number=obs["turn"],
            current_rank=rank,
            rank_cycle=self._rank_cycle,
            hand_size=obs["my_hand_size"],
            hand_list=", ".join(str(c) for c in hand) or "(empty)",
            matching_count=matching,
            hand_sizes=self._fmt_hand_sizes(obs),
            discard_pile_size=obs["pile_size"],
            lie_frequencies=self._fmt_lie_frequencies(obs),
            claim_history=self._fmt_claim_history(),
        )

    def _build_challenge_prompt(self, obs: Dict[str, Any]) -> str:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        lc = obs["last_claim"]
        matching = sum(1 for c in hand if c.rank == rank or c.rank == JOKER_RANK)
        claimer_id = lc["player_id"]
        claimer_name = self._agent_names.get(claimer_id, f"P{claimer_id}")
        claimer_turns = max(self._turn_counts.get(claimer_id, 1), 1)
        claimer_lie_freq = self._lie_counts.get(claimer_id, 0) / claimer_turns
        total_rank_claimed = self._rank_claimed_this_cycle.get(rank, 0) + lc["n_cards"]

        return CHALLENGE_PROMPT_TEMPLATE.format(
            turn_number=obs["turn"],
            current_rank=rank,
            rank_cycle=self._rank_cycle,
            hand_size=obs["my_hand_size"],
            hand_list=", ".join(str(c) for c in hand) or "(empty)",
            matching_count=matching,
            hand_sizes=self._fmt_hand_sizes(obs),
            discard_pile_size=obs["pile_size"],
            lie_frequencies=self._fmt_lie_frequencies(obs),
            claim_history=self._fmt_claim_history(),
            claimer_name=claimer_name,
            claimer_id=claimer_id,
            actual_cards_placed=lc["n_cards"],
            claimed_count=lc["n_cards"],
            claimer_lie_freq=claimer_lie_freq,
            total_rank_claimed=total_rank_claimed,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, response: dict, obs: Dict[str, Any], is_challenge_decision: bool
    ) -> Dict[str, Any]:
        action = response.get("action", "play")
        self._last_belief = {"reasoning": response.get("reasoning", "")}

        if is_challenge_decision:
            if action == "challenge":
                lc = obs.get("last_claim")
                if lc and lc["player_id"] != self.player_id:
                    return {"type": "challenge"}
            # pass_challenge or anything else → return play action (treated as "no challenge" by runner)
            return self._fallback_action(obs)

        # Play decision
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        card_strings: List[str] = response.get("cards", [])

        hand_lookup: Dict[str, List[Card]] = defaultdict(list)
        for c in hand:
            hand_lookup[str(c)].append(c)

        parsed: List[Card] = []
        for cs in card_strings:
            if cs in hand_lookup and hand_lookup[cs]:
                parsed.append(hand_lookup[cs].pop(0))

        if not parsed:
            self.fallback_play_count += 1
            return self._fallback_action(obs)

        return {"type": "play", "cards": parsed, "claimed_rank": rank}

    def _fallback_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        if not hand:
            # Hand is empty — game should have ended; raise so the runner can handle it
            raise ValueError("_fallback_action called with empty hand (game logic error)")
        matching = [c for c in hand if c.rank == rank]
        if matching:
            return {"type": "play", "cards": [matching[0]], "claimed_rank": rank}
        jokers = [c for c in hand if c.rank == JOKER_RANK]
        if jokers:
            return {"type": "play", "cards": [jokers[0]], "claimed_rank": rank}
        return {"type": "play", "cards": [hand[0]], "claimed_rank": rank}

    def _record_trace(self, obs: Dict[str, Any], response: dict, phase: str) -> None:
        self.reasoning_traces.append({
            "turn":         obs["turn"],
            "player_id":    self.player_id,
            "phase":        phase,
            "prompt_mode":  self.prompt_mode,
            "rank_cycle":   self._rank_cycle,
            "current_rank": obs["current_rank"],
            "hand_size":    obs["my_hand_size"],
            "pile_size":    obs["pile_size"],
            "reasoning":    response.get("reasoning", ""),
            "action":       response.get("action"),
            "cards":        response.get("cards", []),
        })

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> Optional[dict]:
        if self.provider in ("openai", "xai", "local"):
            return self._call_openai_compatible(user_prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(user_prompt)
        if self.provider == "google":
            return self._call_google(user_prompt)
        raise ValueError(f"Unknown provider: {self.provider!r}")

    def _parse_json_content(self, content: str) -> dict:
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client:
            return self._openai_client
        key = self._api_key or os.environ.get(self._ENV_KEYS.get(self.provider, "OPENAI_API_KEY"), "")
        if self.provider == "local" and not key:
            key = "ollama"
        if self.provider in ("openai", "xai") and not key:
            raise ValueError(f"API key not set for provider {self.provider!r}")
        url = self.base_url
        if url is None and self.provider == "xai":
            url = "https://api.x.ai/v1"
        elif url is None and self.provider == "local":
            url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        kwargs: Dict[str, Any] = {"api_key": key}
        if url:
            kwargs["base_url"] = url
        self._openai_client = OpenAI(**kwargs)
        return self._openai_client

    def _call_openai_compatible(self, user_prompt: str) -> Optional[dict]:
        client = self._get_openai_client()
        content = ""
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                content = (resp.choices[0].message.content or "").strip()
                return self._parse_json_content(content)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error attempt %s: %s", attempt + 1, e)
            except Exception as e:
                logger.warning("API error attempt %s: %s", attempt + 1, e)
        return None

    def _call_anthropic(self, user_prompt: str) -> Optional[dict]:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=key)
        content = ""
        for attempt in range(self.max_retries):
            try:
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=self.temperature,
                    system=self._system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
                return self._parse_json_content(content)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error attempt %s: %s", attempt + 1, e)
            except Exception as e:
                logger.warning("Anthropic API error attempt %s: %s", attempt + 1, e)
        return None

    def _call_google(self, user_prompt: str) -> Optional[dict]:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("pip install google-generativeai") from exc
        key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=key)
        content = ""
        for attempt in range(self.max_retries):
            try:
                m = genai.GenerativeModel(self.model, system_instruction=self._system_prompt)
                resp = m.generate_content(
                    user_prompt,
                    generation_config={"temperature": self.temperature, "max_output_tokens": 512},
                )
                content = (resp.text or "").strip()
                return self._parse_json_content(content)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error attempt %s: %s", attempt + 1, e)
            except Exception as e:
                logger.warning("Google API error attempt %s: %s", attempt + 1, e)
        return None
