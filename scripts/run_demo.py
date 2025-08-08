#!/usr/bin/env python3
"""
Demo runner script for the text normalization project.

This script provides an easy way to run demonstrations and comparisons
of the text normalization approaches without dealing with CLI arguments.
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from text_normalization import HybridNormalizationAgent
from text_normalization.agentic_agent import ImprovedTextNormalizationAgent


def print_banner():
    """Print a nice banner for the demo."""
    print("🎯 " + "=" * 60)
    print("   TEXT NORMALIZATION DEMO")
    print("   Music Industry Composer/Writer Name Cleanup")
    print("=" * 64)


def print_menu():
    """Print the main menu options."""
    print("\n📋 DEMO OPTIONS:")
    print("  1. 🔧 Single Text Processing (Interactive)")
    print("  2. 🧪 Compare All 3 Approaches (Interactive)")
    print("  3. 📊 Quick Dataset Sample (5 cases)")
    print("  4. 📈 Medium Dataset Sample (25 cases)")
    print("  5. 📋 Large Dataset Sample (100 cases)")
    print("  6. 🔬 Research Comparison (Custom size)")
    print("  7. ❓ Show Example Cases")
    print("  0. 🚪 Exit")


def process_single_text():
    """Process a single text with user input."""
    print("\n🔧 SINGLE TEXT PROCESSING")
    print("-" * 40)

    # Get user input
    text = input("Enter text to normalize: ").strip()
    if not text:
        print("❌ No text provided!")
        return

    # Choose approach
    print("\nChoose approach:")
    print("  1. Rule-Based (Hybrid)")
    print("  2. Agentic (LLM-powered)")
    print("  3. Both (comparison)")

    choice = input("Select (1-3): ").strip()

    try:
        if choice in ["1", "3"]:
            print("\n📏 RULE-BASED APPROACH:")
            print("-" * 30)
            hybrid_agent = HybridNormalizationAgent()
            start_time = time.time()
            hybrid_result = hybrid_agent.normalize_text(text)
            hybrid_time = time.time() - start_time

            print(f"Input:      {text}")
            print(f"Output:     {hybrid_result.normalized_text}")
            print(f"Confidence: {hybrid_result.confidence:.3f}")
            print(f"Time:       {hybrid_time:.3f}s")
            print(f"Reasoning:  {hybrid_result.reasoning}")

        if choice in ["2", "3"]:
            print("\n🤖 AGENTIC APPROACH:")
            print("-" * 30)
            agentic_agent = ImprovedTextNormalizationAgent()
            start_time = time.time()
            agentic_result = agentic_agent.process(text)
            agentic_time = time.time() - start_time

            print(f"Input:      {text}")
            print(f"Output:     {agentic_result.output_text}")
            print(f"Confidence: {agentic_result.confidence:.3f}")
            print(f"Time:       {agentic_time:.3f}s")
            print(f"Strategy:   {agentic_result.strategy_used}")
            print(f"Tokens:     {agentic_result.tokens_used}")

        if choice == "3":
            print("\n⚖️  COMPARISON:")
            print("-" * 20)
            match = hybrid_result.normalized_text == agentic_result.output_text
            print(f"Results match: {'✅ Yes' if match else '❌ No'}")
            if not match:
                print(f"Rule-based: '{hybrid_result.normalized_text}'")
                print(f"Agentic:    '{agentic_result.output_text}'")

            speed_ratio = agentic_time / hybrid_time if hybrid_time > 0 else 1
            print(f"Speed ratio: {speed_ratio:.1f}x (agentic vs rule-based)")

    except Exception as e:
        print(f"❌ Error: {e}")


def run_dataset_sample(sample_size: int):
    """Run a sample from the dataset."""
    print(f"\n📊 DATASET SAMPLE ({sample_size} cases)")
    print("-" * 50)

    dataset_path = project_root / "data" / "normalization_assesment_dataset_10k.csv"

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the dataset is in the data/ directory")
        return

    try:
        import pandas as pd

        df = pd.read_csv(dataset_path)
        df = df.dropna(subset=["raw_comp_writers_text"])
        sample_df = df.head(sample_size)

        print(f"📈 Processing {len(sample_df)} cases...")

        # Initialize agents
        hybrid_agent = HybridNormalizationAgent()
        agentic_agent = ImprovedTextNormalizationAgent()

        results = []
        total_time = 0

        for idx, row in sample_df.iterrows():
            raw_text = row["raw_comp_writers_text"]
            print(
                f"\r📝 Case {idx + 1}/{len(sample_df)}: {raw_text[:50]}...",
                end="",
                flush=True,
            )

            start_time = time.time()

            # Process with both approaches
            hybrid_result = hybrid_agent.normalize_text(raw_text)
            agentic_result = agentic_agent.process(raw_text)

            case_time = time.time() - start_time
            total_time += case_time

            results.append(
                {
                    "input": raw_text,
                    "hybrid_output": hybrid_result.normalized_text,
                    "agentic_output": agentic_result.output_text,
                    "match": hybrid_result.normalized_text
                    == agentic_result.output_text,
                    "time": case_time,
                }
            )

        print("\n")  # New line after progress

        # Summary statistics
        matches = sum(1 for r in results if r["match"])
        match_rate = matches / len(results) * 100
        avg_time = total_time / len(results)

        print("✅ Completed! Summary:")
        print(f"  • Agreement rate: {match_rate:.1f}% ({matches}/{len(results)})")
        print(f"  • Average time:   {avg_time:.3f}s per case")
        print(f"  • Total time:     {total_time:.2f}s")

        # Show a few examples
        print("\n📋 Sample Results:")
        for i, result in enumerate(results[:3]):
            status = "✅" if result["match"] else "⚠️"
            print(f"  {status} Case {i + 1}: {result['input'][:40]}...")
            print(f"     → {result['hybrid_output']}")
            if not result["match"]:
                print(f"     → {result['agentic_output']} (agentic)")

        if len(results) > 3:
            print(f"     ... and {len(results) - 3} more cases")

    except ImportError:
        print("❌ pandas not installed. Install with: pip install pandas")
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")


def run_custom_research():
    """Run a custom research comparison."""
    print("\n🔬 CUSTOM RESEARCH COMPARISON")
    print("-" * 40)

    try:
        size = int(input("Enter sample size (1-1000): "))
        if not 1 <= size <= 1000:
            print("❌ Size must be between 1 and 1000")
            return

        print(f"\n🚀 Running research comparison with {size} samples...")
        print("This will generate a detailed report...")

        # Import and run the research comparison
        research_module = project_root / "research_comparison.py"
        if research_module.exists():
            import subprocess

            subprocess.run(
                [sys.executable, str(research_module), "--samples", str(size)],
                cwd=project_root,
            )
        else:
            print("❌ Research comparison script not found")

    except ValueError:
        print("❌ Invalid number")
    except Exception as e:
        print(f"❌ Error: {e}")


def show_example_cases():
    """Show some example cases and expected outputs."""
    print("\n❓ EXAMPLE CASES")
    print("-" * 30)

    examples = [
        {
            "input": "John Smith/Sony Music Publishing",
            "description": "Simple name with publisher",
            "expected": "John Smith",
        },
        {
            "input": "Smith, John & Doe, Jane",
            "description": "Inverted names with separator",
            "expected": "John Smith/Jane Doe",
        },
        {
            "input": "EMINEM",
            "description": "Artist name (single word)",
            "expected": "Eminem",
        },
        {
            "input": "<Unknown>/Johnson, Michael",
            "description": "Unknown tag with inversion",
            "expected": "Michael Johnson",
        },
        {
            "input": "Taylor Swift, Ed Sheeran & Bruno Mars",
            "description": "Multiple artists with separators",
            "expected": "Taylor Swift/Ed Sheeran/Bruno Mars",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   Input:    {example['input']}")
        print(f"   Expected: {example['expected']}")


def main():
    """Main demo function."""
    print_banner()

    while True:
        print_menu()
        choice = input("\nSelect option (0-7): ").strip()

        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            process_single_text()
        elif choice == "2":
            process_single_text()  # Same as option 1 with comparison
        elif choice == "3":
            run_dataset_sample(5)
        elif choice == "4":
            run_dataset_sample(25)
        elif choice == "5":
            run_dataset_sample(100)
        elif choice == "6":
            run_custom_research()
        elif choice == "7":
            show_example_cases()
        else:
            print("❌ Invalid option. Please select 0-7.")

        if choice != "0":
            input("\n⏸️  Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
