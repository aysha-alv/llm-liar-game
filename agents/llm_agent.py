"""
Multi-provider LLM agent for Liar (Cheat/Bullshit).

Supports OpenAI-compatible APIs (OpenAI, xAI Grok, Ollama, vLLM), Anthropic, and Google Gemini.
Returns strict JSON: play | challenge | pass_challenge.
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from openai import OpenAI

from .base_agent import BaseAgent
from game.card import Card, Rank, Suit

if TYPE_CHECKING:
    from game.game_state import GameState, ClaimRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset model IDs → (provider, api_model_id)
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "grok-3": {"provider": "xai", "model": "grok-3"},
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "claude-sonnet-4-20250514": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "gemini-2.0-flash": {"provider": "google", "model": "gemini-2.0-flash"},
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert player of the card game Liar (also called Cheat or Bullshit).

RULES:
- 52 cards are dealt among all players. Goal: be first to empty your hand.
- Players take turns placing cards face-down and CLAIMING they match the current required rank.
- Everyone can see HOW MANY cards were placed (actual count); they cannot see the faces until a challenge.
- Ranks cycle: Ace → Two → Three → ... → King → Ace (repeating)
- You MAY lie about the cards you play. Lying includes playing the wrong rank OR claiming a different count than the number of cards you put down.
- After any play, any OTHER player can call "Bullshit" (challenge):
    - If the player lied → they pick up the entire discard pile
    - If the player was honest → the challenger picks up the pile
- After a challenge (win or lose), the rank advances and play continues.

STRATEGIC PRINCIPLES:
1. Your PRIMARY goal is to empty your hand. Every decision should serve this.
2. Play honestly when you have matching cards — this is risk-free progress.
3. When forced to bluff, play FEW cards (1 is safest) and claim a count that matches how many cards you actually place.
4. Challenge SPARINGLY. Winning a challenge does not reduce YOUR hand.
   Losing a challenge means YOU pick up the entire pile. Only challenge when you are highly confident AND the pile is small enough that a loss is acceptable.
5. Lie frequency estimates come only from past CAUGHT lies — they are noisy, not gospel. A moderate number does not mean this specific play is a lie.
6. Within one rank cycle (Ace through King once), at most 4 cards of a given rank exist in the deck. Totals can exceed 4 across later cycles because cards recycle through the discard pile.

OUTPUT FORMAT (strictly valid JSON):
When it is YOUR TURN to PLAY:
{
  "action": "play",
  "cards": ["CardString1", "CardString2"],
  "claimed_count": N,
  "reasoning": "brief explanation"
}

When you are DECIDING WHETHER TO CHALLENGE (not your play turn):
{
  "action": "challenge",
  "reasoning": "brief explanation"
}
or
{
  "action": "pass_challenge",
  "reasoning": "brief explanation"
}

IMPORTANT: Output ONLY the JSON object. No other text."""

PLAY_PROMPT_TEMPLATE = """
=== GAME STATE (Turn {turn_number}) ===

Required rank this turn: {current_rank}
Rank cycle index (Ace..King rounds): {rank_cycle}

My hand ({hand_size} cards): {hand_list}
  - Cards matching current rank: {matching_count}x {current_rank}

Hand sizes:
{hand_sizes}

Discard pile size: {discard_pile_size} cards

Estimated lie frequencies from CAUGHT lies only (0.0 = never caught lying, higher = caught more often relative to turns):
{lie_frequencies}

Recent claim history (last 8 turns; "placed" = face-down cards actually put on pile):
{claim_history}

=== YOUR TURN TO PLAY ===
Choose which cards to play from your hand and what count to claim (claimed_count must equal how many cards you list).
"""

CHALLENGE_PROMPT_TEMPLATE = """
=== GAME STATE (Turn {turn_number}) ===

Required rank this turn: {current_rank}
Rank cycle index (Ace..King rounds): {rank_cycle}

My hand ({hand_size} cards): {hand_list}
  - Cards matching current rank: {matching_count}x {current_rank}

Hand sizes:
{hand_sizes}

Discard pile size: {discard_pile_size} cards

Estimated lie frequencies from CAUGHT lies only:
{lie_frequencies}

Recent claim history (last 8 turns):
{claim_history}

=== CHALLENGE DECISION ===
Player {claimer_name} (ID={claimer_id}) placed {actual_cards_placed} card(s) face-down and claimed {claimed_count}x {current_rank}.
Their estimated caught-lie frequency: {claimer_lie_freq:.2f}
Total claimed count for {current_rank} in THIS rank cycle only (including this play): {total_rank_claimed}
Maximum possible {current_rank}s in the deck in a single cycle: 4
(Across multiple Ace→King rounds, more than 4 can be claimed in total because cards return to play via the pile.)

Do you challenge (call "Bullshit") or pass?
"""


# ---------------------------------------------------------------------------
# Card string parsing
# ---------------------------------------------------------------------------

RANK_LABEL_MAP = {r.label(): r for r in Rank}
SUIT_SYMBOL_MAP = {s.value: s for s in Suit}


def parse_card_string(s: str) -> Card:
    """Parse 'Ace♥' or 'King♠' etc. back to a Card object."""
    for rank_label, rank in RANK_LABEL_MAP.items():
        if s.startswith(rank_label):
            suit_symbol = s[len(rank_label) :]
            if suit_symbol in SUIT_SYMBOL_MAP:
                return Card(rank, SUIT_SYMBOL_MAP[suit_symbol])
    raise ValueError(f"Cannot parse card string: {s!r}")


def parse_model_spec(spec: str, default_provider: str) -> Tuple[str, str]:
    """
    'openai:gpt-4o' → ('openai', 'gpt-4o')
    'grok-3' with MODEL_CONFIGS → ('xai', 'grok-3')
    'custom-model' → (default_provider, 'custom-model')
    """
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
    """
    LLM player with pluggable provider (OpenAI-compatible, Anthropic, Google).
    """

    _ENV_KEYS = {
        "xai": "XAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "local": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    def __init__(
        self,
        name: str = "LLM",
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(name)
        self.provider = provider.lower().strip()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_url = base_url
        self._api_key = api_key
        self._openai_client: Optional[OpenAI] = None
        self.reasoning_traces: List[dict] = []
        self.fallback_play_count: int = 0

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is not None:
            return self._openai_client
        key = self._api_key
        if not key:
            env = self._ENV_KEYS.get(self.provider, "OPENAI_API_KEY")
            key = os.environ.get(env, "")
        if self.provider == "local" and not key:
            key = "ollama"
        if self.provider in ("openai", "xai") and not key:
            raise ValueError(
                f"API key not found for provider {self.provider!r}. "
                f"Set {self._ENV_KEYS.get(self.provider)} or pass api_key=."
            )
        url = self.base_url
        if url is None:
            if self.provider == "xai":
                url = "https://api.x.ai/v1"
            elif self.provider == "local":
                url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
            else:
                url = None
        kwargs: Dict[str, Any] = {"api_key": key}
        if url:
            kwargs["base_url"] = url
        self._openai_client = OpenAI(**kwargs)
        return self._openai_client

    def choose_play(self, state: "GameState", player_id: int) -> Tuple[List[Card], int]:
        player = state.players[player_id]
        prompt = self._build_play_prompt(state, player_id)
        response = self._call_llm(prompt)

        if response is None:
            self.fallback_play_count += 1
            return self._fallback_play(player, state.current_rank)

        self._record_trace(state, player_id, "play", response)

        try:
            return self._parse_play_response(response, player, state.current_rank)
        except Exception as e:
            logger.warning(f"Failed to parse play response: {e}. Using fallback.")
            self.fallback_play_count += 1
            return self._fallback_play(player, state.current_rank)

    def choose_challenge(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> bool:
        prompt = self._build_challenge_prompt(state, player_id, claim)
        response = self._call_llm(prompt)

        if response is None:
            return False

        self._record_trace(state, player_id, "challenge", response)

        try:
            return self._parse_challenge_response(response)
        except Exception as e:
            logger.warning(f"Failed to parse challenge response: {e}. Defaulting to no challenge.")
            return False

    def _build_play_prompt(self, state: "GameState", player_id: int) -> str:
        player = state.players[player_id]
        rank = state.current_rank
        obs = state.get_public_observation(player_id)

        hand_list = ", ".join(str(c) for c in player.hand)
        matching_count = player.count_rank(rank)

        hand_sizes_str = "\n".join(
            f"  Player {pid} ({state.players[pid].name}): {size} cards"
            for pid, size in obs["hand_sizes"].items()
            if pid != player_id
        )

        lie_freq_str = "\n".join(
            f"  Player {pid} ({state.players[pid].name}): {freq:.2f}"
            for pid, freq in obs["lie_frequencies"].items()
            if pid != player_id
        )

        history = obs["claim_history"][-8:]
        history_str = "\n".join(
            f"  Turn {i}: P{r['player']} placed {r.get('actual_cards_placed', '?')} card(s), "
            f"claimed {r['claimed_count']}x {r['claimed_rank']}"
            + (f" → CHALLENGED ({r['challenge_result']})" if r["was_challenged"] else "")
            for i, r in enumerate(history)
        ) or "  (No claims yet)"

        return PLAY_PROMPT_TEMPLATE.format(
            turn_number=state.turn_number,
            current_rank=rank.label(),
            rank_cycle=obs.get("current_rank_cycle", state.current_rank_cycle),
            hand_size=player.hand_size,
            hand_list=hand_list,
            matching_count=matching_count,
            hand_sizes=hand_sizes_str,
            discard_pile_size=state.discard_pile_size,
            lie_frequencies=lie_freq_str,
            claim_history=history_str,
        )

    def _build_challenge_prompt(self, state: "GameState", player_id: int, claim: "ClaimRecord") -> str:
        player = state.players[player_id]
        rank = state.current_rank
        obs = state.get_public_observation(player_id)

        hand_list = ", ".join(str(c) for c in player.hand)
        matching_count = player.count_rank(rank)

        hand_sizes_str = "\n".join(
            f"  Player {pid} ({state.players[pid].name}): {size} cards"
            for pid, size in obs["hand_sizes"].items()
            if pid != player_id
        )

        lie_freq_str = "\n".join(
            f"  Player {pid} ({state.players[pid].name}): {freq:.2f}"
            for pid, freq in obs["lie_frequencies"].items()
            if pid != player_id
        )

        history = obs["claim_history"][-8:]
        history_str = "\n".join(
            f"  Turn {i}: P{r['player']} placed {r.get('actual_cards_placed', '?')} card(s), "
            f"claimed {r['claimed_count']}x {r['claimed_rank']}"
            + (f" → CHALLENGED ({r['challenge_result']})" if r["was_challenged"] else "")
            for i, r in enumerate(history)
        ) or "  (No claims yet)"

        claimer_lie_freq = obs["lie_frequencies"].get(claim.player_id, 0.0)
        cycle = claim.rank_cycle
        total_rank_claimed = sum(
            r["claimed_count"]
            for r in obs["claim_history"]
            if r["claimed_rank"] == rank.label() and r.get("rank_cycle", cycle) == cycle
        ) + claim.claimed_count

        return CHALLENGE_PROMPT_TEMPLATE.format(
            turn_number=state.turn_number,
            current_rank=rank.label(),
            rank_cycle=obs.get("current_rank_cycle", state.current_rank_cycle),
            hand_size=player.hand_size,
            hand_list=hand_list,
            matching_count=matching_count,
            hand_sizes=hand_sizes_str,
            discard_pile_size=state.discard_pile_size,
            lie_frequencies=lie_freq_str,
            claim_history=history_str,
            claimer_name=state.players[claim.player_id].name,
            claimer_id=claim.player_id,
            claimed_count=claim.claimed_count,
            actual_cards_placed=len(claim.actual_cards),
            claimer_lie_freq=claimer_lie_freq,
            total_rank_claimed=total_rank_claimed,
        )

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

    def _call_openai_compatible(self, user_prompt: str) -> Optional[dict]:
        client = self._get_openai_client()
        content = ""
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                content = (response.choices[0].message.content or "").strip()
                return self._parse_json_content(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}\nContent: {content!r}")
            except Exception as e:
                logger.warning(f"API error on attempt {attempt + 1}: {e}")
        return None

    def _call_anthropic(self, user_prompt: str) -> Optional[dict]:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Install anthropic: pip install anthropic") from e

        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
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
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = ""
                for block in msg.content:
                    if hasattr(block, "text"):
                        content += block.text
                content = content.strip()
                return self._parse_json_content(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}\nContent: {content!r}")
            except Exception as e:
                logger.warning(f"Anthropic API error on attempt {attempt + 1}: {e}")
        return None

    def _call_google(self, user_prompt: str) -> Optional[dict]:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("Install google-generativeai: pip install google-generativeai") from e

        key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        content = ""
        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(
                    self.model,
                    system_instruction=SYSTEM_PROMPT,
                )
                resp = model.generate_content(
                    user_prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": 512,
                    },
                )
                content = (resp.text or "").strip()
                return self._parse_json_content(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}\nContent: {content!r}")
            except Exception as e:
                logger.warning(f"Google API error on attempt {attempt + 1}: {e}")
        return None

    def _parse_play_response(self, response: dict, player, rank: Rank) -> Tuple[List[Card], int]:
        action = response.get("action", "play")
        if action != "play":
            self.fallback_play_count += 1
            return self._fallback_play(player, rank)

        card_strings = response.get("cards", [])
        claimed_count = int(response.get("claimed_count", 1))

        hand_lookup: Dict[str, List[Card]] = {}
        for c in player.hand:
            key = str(c)
            hand_lookup.setdefault(key, []).append(c)

        parsed_cards: List[Card] = []
        for cs in card_strings:
            if cs in hand_lookup and hand_lookup[cs]:
                parsed_cards.append(hand_lookup[cs].pop(0))

        if not parsed_cards:
            self.fallback_play_count += 1
            return self._fallback_play(player, rank)

        claimed_count = max(1, min(4, claimed_count))
        return parsed_cards, claimed_count

    def _parse_challenge_response(self, response: dict) -> bool:
        return response.get("action") == "challenge"

    def _fallback_play(self, player, rank: Rank) -> Tuple[List[Card], int]:
        import random

        matching = [c for c in player.hand if c.rank == rank]
        if matching:
            return [matching[0]], 1
        return [random.choice(player.hand)], 1

    def _record_trace(self, state: "GameState", player_id: int, phase: str, response: dict) -> None:
        self.reasoning_traces.append(
            {
                "turn": state.turn_number,
                "player_id": player_id,
                "phase": phase,
                "current_rank": state.current_rank.label(),
                "hand_size": state.players[player_id].hand_size,
                "reasoning": response.get("reasoning", ""),
                "action": response.get("action"),
                "claimed_count": response.get("claimed_count"),
            }
        )


class GrokAgent(LLMAgent):
    """Backward-compatible alias: xAI Grok with OpenAI-compatible API."""

    DEFAULT_MODEL = "grok-3"

    def __init__(
        self,
        name: str = "Grok",
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        api_key = kwargs.get("api_key") or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY environment variable "
                "or pass api_key= to GrokAgent()."
            )
        kwargs = {**kwargs, "api_key": api_key}
        kwargs.setdefault("provider", "xai")
        super().__init__(
            name=name,
            model=model or self.DEFAULT_MODEL,
            **kwargs,
        )
