#!/usr/bin/env python3
"""
Quick start script for text normalization.

Usage:
    python scripts/quick_start.py "John Smith/Sony Music"
    python scripts/quick_start.py --approach agentic "Text to normalize"
    python scripts/quick_start.py --batch 10
"""

import argparse
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


def quick_normalize(text: str, approach: str = "hybrid") -> None:
    """
    Quickly normalize a single text using the specified approach.

    Args:
        text: The text to normalize
        approach: The normalization approach to use ("hybrid" or "agentic")
    """
    print(f"🔧 Normalizing: '{text}'")
    print(f"📋 Using: {approach}")
    print("-" * 50)

    try:
        if approach.lower() in ["hybrid", "rule", "rules"]:
            from text_normalization import HybridNormalizationAgent

            agent = HybridNormalizationAgent()
            result = agent.normalize_text(text)

            print(f"✅ Result: '{result.normalized_text}'")
            print(f"🎯 Confidence: {result.confidence:.3f}")
            print(f"💭 Reasoning: {result.reasoning}")

        elif approach.lower() in ["agentic", "llm", "ai"]:
            from text_normalization.agentic_agent import ImprovedTextNormalizationAgent

            agent = ImprovedTextNormalizationAgent()
            result = agent.process(text)

            print(f"✅ Result: '{result.output_text}'")
            print(f"🎯 Confidence: {result.confidence:.3f}")
            print(f"🔧 Strategy: {result.strategy_used}")
            print(f"🔢 Tokens: {result.tokens_used}")

        else:
            print(f"❌ Unknown approach: {approach}")
            print("Available: hybrid, agentic")

    except Exception as e:
        print(f"❌ Error: {e}")


def quick_batch(sample_size: int = 5) -> None:
    """
    Run a quick batch test on dataset samples.

    Args:
        sample_size: Number of samples to process
    """
    print(f"📊 Quick batch test ({sample_size} samples)")
    print("-" * 50)

    dataset_path = project_root / "data" / "normalization_assesment_dataset_10k.csv"

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return

    try:
        import pandas as pd

        from text_normalization import HybridNormalizationAgent

        df = pd.read_csv(dataset_path)
        df = df.dropna(subset=["raw_comp_writers_text"])
        sample_df = df.head(sample_size)

        agent = HybridNormalizationAgent()

        print(f"📈 Processing {len(sample_df)} cases...")

        for i, (_, row) in enumerate(sample_df.iterrows()):
            raw_text = row["raw_comp_writers_text"]
            result = agent.normalize_text(raw_text)

            print(f"\n{i + 1}. Input:  {raw_text}")
            print(f"   Output: {result.normalized_text}")

    except ImportError:
        print("❌ pandas not installed. Run: pip install pandas")
    except Exception as e:
        print(f"❌ Error: {e}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick text normalization tool")
    parser.add_argument("text", nargs="?", help="Text to normalize")
    parser.add_argument(
        "--approach",
        default="hybrid",
        choices=["hybrid", "agentic", "rule", "llm", "ai"],
        help="Normalization approach",
    )
    parser.add_argument(
        "--batch", type=int, metavar="N", help="Run batch test with N samples"
    )

    args = parser.parse_args()

    if args.batch:
        quick_batch(args.batch)
    elif args.text:
        quick_normalize(args.text, args.approach)
    else:
        print("🎯 TEXT NORMALIZATION - QUICK START")
        print("=" * 40)
        print("\nExamples:")
        print('  python scripts/quick_start.py "John Smith/Sony Music"')
        print('  python scripts/quick_start.py --approach agentic "Text"')
        print("  python scripts/quick_start.py --batch 10")
        print("\nFor interactive demo, run: python scripts/run_demo.py")


if __name__ == "__main__":
    main()
    main()
