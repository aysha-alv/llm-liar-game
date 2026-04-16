# Liar Game LLM — Grok vs Imperfect Information

Implementation of the full research pipeline from:
**"Strategic Deception and Information-Theoretic Reasoning: A Comprehensive Analysis of Large Language Model Optimization for Imperfect Information Games"**

Tests and trains Grok (via xAI API) to play the card game **Liar** (Cheat/Bullshit) against rule-based strategy archetypes, using a 5-phase research pipeline.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your xAI API key
cp .env.example .env
# Edit .env and add your key from https://console.x.ai/
```

---

## Run the 5 Phases

### Phase 1 — Baseline Testing (START HERE)
Run an LLM (multi-provider) against sampled archetypes. Observable info only: lie frequencies use **caught** lies; challenge prompts use **per rank-cycle** claim totals.

```bash
python scripts/phase1_baseline.py --games 20 --players 4
# Grok (default: --provider xai --model grok-3)
python scripts/phase1_baseline.py --provider openai --model gpt-4o-mini
python scripts/phase1_baseline.py --models grok-3,openai:gpt-4o-mini
```
Presets live in `MODEL_CONFIGS` in `agents/llm_agent.py`. Outputs: win rates, challenge rate, pickup breakdown, fallback-play count, failure modes, traces in `data/traces/`.

---

### Phase 2 — Generate Synthetic Training Data
Creates expert reasoning traces for supervised fine-tuning.
```bash
python scripts/phase2_finetune.py --traces 500
```
Outputs: `data/traces/synthetic_traces.jsonl` (OpenAI fine-tune format)

---

### Phase 3 — Self-Play with Bluff-Aware Rewards
Runs self-play iterations with the paper's reward shaping (bluff_coeff=2.0).
```bash
python scripts/phase3_selfplay.py --iterations 5 --games 10
# or full 3-level curriculum:
python scripts/phase3_selfplay.py --curriculum
```
Outputs: training data in `data/self_play/`

---

### Phase 4 — CFR Strategy Training
Trains a Nash equilibrium strategy via Counterfactual Regret Minimization.
```bash
python scripts/phase4_cfr.py --iterations 50000 --show
```
Outputs: `data/cfr/strategy.json`

---

### Phase 5 — Full Evaluation
Round-robin tournament with TrueSkill + information-theoretic metrics.
```bash
python scripts/phase5_evaluate.py --games 100
```
Outputs: leaderboard, MI/KL/TrueSkill report in `data/results/`

---

## Architecture

```
liar-game-llm/
├── game/
│   ├── card.py          # Card, Deck, Rank, Suit
│   ├── game_state.py    # Full game state + observations
│   └── liar_game.py     # Game engine / orchestrator
├── agents/
│   ├── base_agent.py    # Abstract agent interface
│   ├── archetypes.py    # 6 rule-based baselines from paper
│   └── llm_agent.py     # LLMAgent (OpenAI / xAI / Anthropic / Google / local)
├── training/
│   ├── synthetic_traces.py  # Phase 2: trace generation
│   ├── reward_shaping.py    # Phase 3: bluff-aware rewards
│   ├── self_play.py         # Phase 3: self-play loop
│   └── cfr.py               # Phase 4: CFR solver
├── evaluation/
│   ├── metrics.py           # MI, KL divergence, TrueSkill
│   └── tournament.py        # Round-robin evaluation
└── scripts/
    ├── phase1_baseline.py
    ├── phase2_finetune.py
    ├── phase3_selfplay.py
    ├── phase4_cfr.py
    └── phase5_evaluate.py
```

---

## Strategy Archetypes (baselines)

| Agent | Strategy | Paper Win Rate |
|---|---|---|
| TheSaint | Never lies unless forced; conservative challenger | 51% |
| ComebackCloser | Bluffs when losing, honest when winning | 43% |
| GameTheorist | EIG-based discarding + Bayesian challenges | Variable |
| MrPathological | Always lies, 50% random challenges | 2% |
| TheAccountant | Tracks lie frequencies, adaptive challenges | Variable |
| TheCollector | Builds four-of-a-kinds via strategic challenges | Moderate |

---

## Key Metrics

| Metric | What it measures | Target |
|---|---|---|
| Win Rate | Games won | Higher |
| Mutual Information I(H;A) | How much actions reveal hand | **Lower** |
| KL Divergence | How much opponents are deceived | **Higher** |
| TrueSkill Rating | Bayesian skill vs all opponents | Higher |
| Bluff Catch Rate | Fraction of bluffs caught | **Lower** |

---

## Failure Modes to Watch (from paper)

- **Indexing errors** — Playing the wrong rank when matching cards exist
- **Computation errors** — Claiming more than 4 of a rank total
- **Control flow errors** — Forgetting game state after a challenge
- **Hallucination** — Claiming cards not in hand
- **Logic following** — Failing to challenge impossible claims

---

## Notes on Fine-tuning

The xAI Grok fine-tuning API is not yet publicly available. This repo:
1. Generates fine-tuning data in OpenAI-compatible JSONL format (Phase 2)
2. Accumulates self-play training data (Phase 3)
3. Is ready to plug in fine-tuned model ID once the API launches

When available: set `--model` / `LLMAgent(model=...)` or update `GrokAgent.DEFAULT_MODEL` for xAI fine-tunes.
