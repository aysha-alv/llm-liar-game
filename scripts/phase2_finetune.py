"""
Phase 2 — Generate synthetic reasoning traces for fine-tuning.

Usage:
    python scripts/phase2_finetune.py --traces 500
    python scripts/phase2_finetune.py --traces 2000 --output data/traces/custom.jsonl
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.synthetic_traces import SyntheticTraceGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Generate synthetic fine-tuning traces")
    parser.add_argument("--traces", type=int, default=500, help="Number of traces to generate")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path("data/traces/synthetic_traces.jsonl")

    gen = SyntheticTraceGenerator(seed=args.seed)
    traces = gen.generate(n_traces=args.traces, output_path=output)

    print(f"\n✓ Generated {len(traces)} reasoning traces")
    print(f"✓ Saved to {output}")
    print(f"\nFormat: OpenAI fine-tune JSONL (system / user / assistant messages)")
    print(f"To fine-tune Grok when xAI fine-tuning becomes available:")
    print(f"  - Upload {output} to the xAI fine-tuning API")
    print(f"  - Update GrokAgent.DEFAULT_MODEL to your fine-tuned model ID")


if __name__ == "__main__":
    main()
