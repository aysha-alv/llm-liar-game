"""
Multi-provider LLM agent for Liar (Cheat/Bullshit).

Implements the new engine interface: choose_action(obs) -> action dict.
Supports OpenAI-compatible APIs (OpenAI, xAI/Grok, Ollama), Anthropic, and Google Gemini.

Prompt modes:
  zero_shot  — rules only, no strategic guidance
  cot        — rules + strategy + explicit chain-of-thought instruction (default)
  few_shot   — rules + strategy + worked examples
"""
from __future__ import annotations
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from openai import OpenAI

from .base import BaseAgent
from engine.card import Card, JOKER_RANK, card_from_str

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

PROMPT_MODES = {"zero_shot", "cot", "few_shot"}

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "grok-3":              {"provider": "xai",       "model": "grok-3"},
    "gpt-4o":              {"provider": "openai",    "model": "gpt-4o"},
    "gpt-4o-mini":         {"provider": "openai",    "model": "gpt-4o-mini"},
    "claude-sonnet-4-5":   {"provider": "anthropic", "model": "claude-sonnet-4-5"},
    "gemini-2.0-flash":    {"provider": "google",    "model": "gemini-2.0-flash"},
}

# ---------------------------------------------------------------------------
# Prompt blocks
# ---------------------------------------------------------------------------

_RULES_BLOCK = """You are a player of the card game Liar (also called Cheat or Bullshit).

RULES:
- 108 cards are dealt (2 standard decks + 4 Jokers). Goal: be first to empty your hand.
- Players take turns placing cards face-down, claiming they match the CURRENT required rank.
- Jokers are wildcards — they count as any rank and are always honest.
- Ranks cycle: A → 2 → 3 → ... → K → A (repeating). Max 8 real cards per rank per cycle.
- You MAY lie: play wrong-rank cards while claiming the current rank.
- After any play, any OTHER player can challenge ("Bullshit!"):
    - Lied → liar picks up the entire discard pile.
    - Honest → challenger picks up the pile.
- First player to empty their hand wins."""

_STRATEGY_BLOCK = """
STRATEGIC PRINCIPLES:
1. PRIMARY goal: empty your hand. Every decision serves this.
2. Play honestly (real matches or Jokers) when you have them — risk-free progress.
3. When forced to bluff, play FEW cards (1 is safest). Don't over-claim.
4. Challenge SPARINGLY. Winning doesn't reduce YOUR hand. Losing means YOU pick up the pile.
   Only challenge when you're highly confident AND the pile loss is acceptable.
5. Lie frequency comes only from CAUGHT lies — a low history doesn't mean the current play is honest.
6. With 2 decks, 8 of each rank exist per cycle. Claims totalling >8 for one rank are impossible."""

_COT_BLOCK = """
Before outputting JSON, reason in the "reasoning" field:
- Do I have matching cards or Jokers?
- Am I winning or losing vs opponents?
- If there's a last_claim: what's the probability it's a lie? What's my pile risk?"""

_FEW_SHOT_BLOCK = """
EXAMPLES:

Example 1 — Honest play:
Rank=King. Hand has 2 Kings. → Play honestly.
{"action": "play", "cards": ["K♠","K♥"], "claimed_rank": "K", "reasoning": "Have matching cards."}

Example 2 — Forced minimal bluff:
Rank=7. No 7s in hand. 18 cards in hand. → Bluff 1 card.
{"action": "play", "cards": ["2♣"], "claimed_rank": "7", "reasoning": "No 7s. Minimal bluff."}

Example 3 — Pass on challenge (pile too large):
Last claim: 2 Aces. Pile=14. Opponent lie freq=0.1. My hand=8.
{"action": "play", "cards": ["3♦"], "claimed_rank": "3", "reasoning": "Pile too large to risk."}

Example 4 — Challenge (mathematically impossible):
Total claimed this rank cycle: 9 (exceeds 8 max). Pile=4.
{"action": "challenge", "reasoning": "9 claimed for this rank — exceeds 8 max. Certain lie."}"""

_OUTPUT_BLOCK = """
OUTPUT — strictly valid JSON, one of:

Play (always specify claimed_rank = current required rank):
{"action": "play", "cards": ["A♠","A♥"], "claimed_rank": "A", "reasoning": "..."}

Challenge the last claim:
{"action": "challenge", "reasoning": "..."}

IMPORTANT: Output ONLY the JSON object. No other text."""


def _build_system_prompt(mode: str) -> str:
    if mode == "zero_shot":
        return _RULES_BLOCK + _OUTPUT_BLOCK
    if mode == "few_shot":
        return _RULES_BLOCK + _STRATEGY_BLOCK + _FEW_SHOT_BLOCK + _OUTPUT_BLOCK
    return _RULES_BLOCK + _STRATEGY_BLOCK + _COT_BLOCK + _OUTPUT_BLOCK  # cot


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_USER_PROMPT_TEMPLATE = """=== GAME STATE (Turn {turn}) ===
Required rank: {current_rank}
My hand ({my_hand_size} cards): {hand_str}
  Matching current rank: {matching_count}x {current_rank}
  Jokers: {joker_count}

Opponent hand sizes: {opp_sizes_str}
Discard pile: {pile_size} cards

{last_claim_section}
=== YOUR TURN ===
{decision_prompt}"""

_PLAY_DECISION = "Choose cards to play from your hand."
_CHALLENGE_DECISION = """Player {claimer_id} just played {n_cards} card(s), claiming {n_cards}x {claimed_rank}.
You can CHALLENGE their claim or play your own cards (implicitly passing the challenge)."""


# ---------------------------------------------------------------------------
# LLMAgent
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
        memory_mode: str = "current_only",  # "current_only" | "full_history"
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
        self.memory_mode = memory_mode
        self._system_prompt = _build_system_prompt(prompt_mode)
        self._openai_client: Optional[OpenAI] = None

        # Runtime state
        self.reasoning_traces: List[dict] = []
        self.fallback_play_count: int = 0
        self._history: List[str] = []    # accumulated turn summaries if full_history
        self._last_belief: Optional[dict] = None

    def reset(self) -> None:
        self.reasoning_traces = []
        self.fallback_play_count = 0
        self._history = []
        self._last_belief = None

    def observe_event(self, event: Dict[str, Any]) -> None:
        if self.memory_mode != "full_history":
            return
        atype = event.get("action", {}).get("type")
        if atype == "play":
            player = event["player"]
            if player == self.player_id:
                summary = (f"Turn {event['turn']}: I played {event['n_cards']}x "
                           f"{event['claimed_rank']} ({'honest' if event.get('honest') else 'bluff'})")
            else:
                summary = (f"Turn {event['turn']}: P{player} claimed "
                           f"{event['n_cards']}x {event['claimed_rank']}")
        elif atype == "challenge":
            result = event.get("challenge_result", "?")
            summary = (f"Turn {event['turn']}: P{event['player']} challenged → "
                       f"{result}, pile→P{event.get('pile_goes_to','?')}")
        else:
            return
        self._history.append(summary)

    def report_belief(self, obs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._last_belief

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(obs)
        response = self._call_llm(prompt)

        if response is None:
            self.fallback_play_count += 1
            return self._fallback_action(obs)

        self._record_trace(obs, response)

        try:
            return self._parse_response(response, obs)
        except Exception as e:
            logger.warning("Failed to parse LLM response: %s — using fallback", e)
            self.fallback_play_count += 1
            return self._fallback_action(obs)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        opp_sizes: Dict[int, int] = obs.get("opponent_hand_sizes", {})
        lc = obs.get("last_claim")

        matching = [c for c in hand if c.rank == rank]
        jokers = [c for c in hand if c.rank == JOKER_RANK]
        hand_str = ", ".join(str(c) for c in hand) if hand else "(empty)"
        opp_str = "  ".join(f"P{pid}:{sz}" for pid, sz in sorted(opp_sizes.items()))

        if lc and lc["player_id"] != self.player_id:
            last_claim_section = (
                f"Last claim: P{lc['player_id']} played {lc['n_cards']} card(s) "
                f"claiming {lc['n_cards']}x {lc['claimed_rank']}"
            )
            decision_prompt = _CHALLENGE_DECISION.format(
                claimer_id=lc["player_id"],
                n_cards=lc["n_cards"],
                claimed_rank=lc["claimed_rank"],
            )
        else:
            last_claim_section = "Last claim: None (game start or post-challenge)"
            decision_prompt = _PLAY_DECISION

        history_section = ""
        if self.memory_mode == "full_history" and self._history:
            recent = self._history[-8:]
            history_section = "\nRecent history:\n" + "\n".join(f"  {h}" for h in recent) + "\n"

        return history_section + _USER_PROMPT_TEMPLATE.format(
            turn=obs["turn"],
            current_rank=rank,
            my_hand_size=obs["my_hand_size"],
            hand_str=hand_str,
            matching_count=len(matching),
            joker_count=len(jokers),
            opp_sizes_str=opp_str or "N/A",
            pile_size=obs["pile_size"],
            last_claim_section=last_claim_section,
            decision_prompt=decision_prompt,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: dict, obs: Dict[str, Any]) -> Dict[str, Any]:
        action = response.get("action", "play")
        self._last_belief = {"reasoning": response.get("reasoning", "")}

        if action == "challenge":
            lc = obs.get("last_claim")
            if lc and lc["player_id"] != self.player_id:
                return {"type": "challenge"}
            # Can't challenge — fall through to play
            logger.warning("LLM wanted to challenge but no valid last_claim — playing instead")

        # Parse play
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        card_strings: List[str] = response.get("cards", [])
        claimed_rank: str = response.get("claimed_rank", rank)

        # Enforce claimed_rank == current_rank (engine rule)
        if claimed_rank != rank:
            logger.warning("LLM claimed rank %r but current rank is %r — correcting", claimed_rank, rank)
            claimed_rank = rank

        # Resolve card strings to actual Card objects in hand
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

        return {"type": "play", "cards": parsed, "claimed_rank": claimed_rank}

    def _fallback_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rank = obs["current_rank"]
        hand: List[Card] = obs["my_hand"]
        if not hand:
            return {"type": "play", "cards": [], "claimed_rank": rank}
        matching = [c for c in hand if c.rank == rank]
        if matching:
            return {"type": "play", "cards": [matching[0]], "claimed_rank": rank}
        jokers = [c for c in hand if c.rank == JOKER_RANK]
        if jokers:
            return {"type": "play", "cards": [jokers[0]], "claimed_rank": rank}
        return {"type": "play", "cards": [hand[0]], "claimed_rank": rank}

    def _record_trace(self, obs: Dict[str, Any], response: dict) -> None:
        self.reasoning_traces.append({
            "turn": obs["turn"],
            "player_id": self.player_id,
            "prompt_mode": self.prompt_mode,
            "current_rank": obs["current_rank"],
            "hand_size": obs["my_hand_size"],
            "pile_size": obs["pile_size"],
            "had_last_claim": obs.get("last_claim") is not None,
            "reasoning": response.get("reasoning", ""),
            "action": response.get("action"),
            "cards": response.get("cards", []),
            "claimed_rank": response.get("claimed_rank"),
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
