#!/usr/bin/env python3
"""
Main CLI interface for text normalization approaches.

Usage:
    python main.py --approach ruled_based --text "John Smith/Sony Music"
    python main.py --approach agentic --text "John Smith/Sony Music"
    python main.py --approach dual --text "John Smith/Sony Music"
    python main.py --batch --approach ruled_based --input data.csv
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# pylint: disable=wrong-import-position
# ruff: noqa: E402
from text_normalization.agentic_agent import (
    ImprovedTextNormalizationAgent as AgenticAgent,
)
from text_normalization.rule_based_agent import HybridNormalizationAgent as RuledAgent


class TextNormalizationCLI:
    """CLI interface for text normalization."""

    def __init__(self) -> None:
        """Initialize the CLI with both agents."""
        print("🔧 Initializing text normalization agents...")

        try:
            self.ruled_agent = RuledAgent()
            print("✅ Rule-based agent ready")
        except Exception as e:
            print(f"❌ Failed to initialize rule-based agent: {e}")
            self.ruled_agent = None

        try:
            self.agentic_agent = AgenticAgent()
            print("✅ Agentic agent ready")
        except Exception as e:
            print(f"❌ Failed to initialize agentic agent: {e}")
            self.agentic_agent = None

    def normalize_single(self, text: str, approach: str) -> Dict[str, Any]:
        """
        Normalize single text using specified approach.

        Args:
            text: The text to normalize
            approach: The approach to use ("ruled_based", "agentic", or "dual")

        Returns:
            Dictionary containing normalization results
        """

        if approach == "ruled_based":
            if not self.ruled_agent:
                return {"error": "Rule-based agent not available"}

            start_time = time.time()
            result = self.ruled_agent.normalize_text(text)
            processing_time = time.time() - start_time

            return {
                "approach": "rule-based",
                "input": text,
                "output": result.normalized_text,
                "processing_time": processing_time,
                "tools_used": result.tools_used,
                "reasoning": result.reasoning,
            }

        elif approach == "agentic":
            if not self.agentic_agent:
                return {"error": "Agentic agent not available"}

            start_time = time.time()
            result = self.agentic_agent.process(text)
            processing_time = time.time() - start_time

            return {
                "approach": "agentic",
                "input": text,
                "output": result.output_text,
                "processing_time": processing_time,
                "tokens_used": result.tokens_used,
                "strategy": result.strategy_used,
            }

        elif approach == "dual":
            # Run both approaches
            results = {}

            if self.ruled_agent:
                start_time = time.time()
                ruled_result = self.ruled_agent.normalize_text(text)
                ruled_time = time.time() - start_time

                results["ruled_based"] = {
                    "output": ruled_result.normalized_text,
                    "processing_time": ruled_time,
                    "tools_used": ruled_result.tools_used,
                }

            if self.agentic_agent:
                start_time = time.time()
                agentic_result = self.agentic_agent.process(text)
                agentic_time = time.time() - start_time

                results["agentic"] = {
                    "output": agentic_result.output_text,
                    "processing_time": agentic_time,
                    "tokens_used": agentic_result.tokens_used,
                    "strategy": agentic_result.strategy_used,
                }

            return {
                "approach": "dual",
                "input": text,
                "results": results,
                "agreement": (
                    results.get("ruled_based", {}).get("output")
                    == results.get("agentic", {}).get("output")
                ),
            }

        else:
            return {"error": f"Unknown approach: {approach}"}

    def normalize_batch(
        self, input_file: str, approach: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Normalize batch of texts from CSV file.

        Args:
            input_file: Path to input CSV file
            approach: Normalization approach to use
            output_file: Optional path to output CSV file

        Returns:
            Dictionary containing batch processing results
        """
        try:
            df = pd.read_csv(input_file)
            if "raw_comp_writers_text" not in df.columns:
                return {"error": "Input file must have 'raw_comp_writers_text' column"}

            results = []
            total_time = 0
            total_tokens = 0

            print(f"📊 Processing {len(df)} texts with {approach} approach...")

            for idx, row in df.iterrows():
                text = row["raw_comp_writers_text"]

                if pd.isna(text) or text.strip() == "":
                    continue

                result = self.normalize_single(text, approach)

                if "error" not in result:
                    results.append(result)
                    total_time += result.get("processing_time", 0)

                    if approach == "agentic":
                        total_tokens += result.get("tokens_used", 0)
                    elif approach == "dual" and "agentic" in result.get("results", {}):
                        total_tokens += result["results"]["agentic"].get(
                            "tokens_used", 0
                        )

                if (idx + 1) % 10 == 0:
                    print(f"Progress: {idx + 1}/{len(df)}")

            # Save results if output file specified
            if output_file:
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_file, index=False)
                print(f"💾 Results saved to: {output_file}")

            return {
                "approach": approach,
                "total_processed": len(results),
                "total_time": total_time,
                "avg_time": total_time / len(results) if results else 0,
                "total_tokens": total_tokens,
                "avg_tokens": total_tokens / len(results) if results else 0,
                "estimated_cost": total_tokens * 0.000002,
                "results": results,
            }

        except Exception as e:
            return {"error": f"Batch processing failed: {e}"}

    def print_single_result(self, result: Dict):
        """Print formatted single result."""

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print("\n📝 Text Normalization Result")
        print("=" * 50)
        print(f"Input:     {result['input']}")

        if result["approach"] == "dual":
            print("Approach:  Dual comparison")

            if "ruled_based" in result["results"]:
                ruled = result["results"]["ruled_based"]
                print("\n📏 Rule-based:")
                print(f"  Output:  {ruled['output']}")
                print(f"  Time:    {ruled['processing_time']:.3f}s")
                print(f"  Tools:   {ruled.get('tools_used', [])}")

            if "agentic" in result["results"]:
                agentic = result["results"]["agentic"]
                print("\n🤖 Agentic:")
                print(f"  Output:  {agentic['output']}")
                print(f"  Time:    {agentic['processing_time']:.3f}s")
                print(f"  Tokens:  {agentic.get('tokens_used', 0)}")
                print(f"  Strategy: {agentic.get('strategy', 'N/A')}")

            agreement = "✅" if result.get("agreement") else "❌"
            print(f"\n🤝 Agreement: {agreement}")

        else:
            print(f"Approach:  {result['approach']}")
            print(f"Output:    {result['output']}")
            print(f"Time:      {result['processing_time']:.3f}s")

            if "tokens_used" in result:
                print(f"Tokens:    {result['tokens_used']}")
                print(f"Cost:      ${result['tokens_used'] * 0.000002:.6f}")

            if "tools_used" in result:
                print(f"Tools:     {result['tools_used']}")

    def print_batch_summary(self, result: Dict):
        """Print formatted batch summary."""

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print("\n📊 Batch Processing Summary")
        print("=" * 50)
        print(f"Approach:       {result['approach']}")
        print(f"Processed:      {result['total_processed']} texts")
        print(f"Total time:     {result['total_time']:.2f}s")
        print(f"Avg time:       {result['avg_time']:.3f}s per text")

        if result["total_tokens"] > 0:
            print(f"Total tokens:   {result['total_tokens']:,}")
            print(f"Avg tokens:     {result['avg_tokens']:.1f} per text")
            print(f"Estimated cost: ${result['estimated_cost']:.4f}")


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Text Normalization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --approach ruled_based --text "John Smith/Sony Music"
  python main.py --approach agentic --text "John Smith/Sony Music"
  python main.py --approach dual --text "John Smith/Sony Music"
  python main.py --batch --approach ruled_based --input data.csv --output results.csv
        """,
    )

    parser.add_argument(
        "--approach",
        choices=["ruled_based", "agentic", "dual"],
        default="ruled_based",
        help="Normalization approach to use",
    )

    parser.add_argument("--text", type=str, help="Single text to normalize")

    parser.add_argument(
        "--batch", action="store_true", help="Process batch of texts from CSV file"
    )

    parser.add_argument("--input", type=str, help="Input CSV file for batch processing")

    parser.add_argument("--output", type=str, help="Output CSV file for batch results")

    args = parser.parse_args()

    # Validate arguments
    if not args.text and not args.batch:
        parser.error("Must specify either --text or --batch")

    if args.batch and not args.input:
        parser.error("--batch requires --input file")

    if args.text and args.batch:
        parser.error("Cannot use --text and --batch together")

    # Initialize CLI
    cli = TextNormalizationCLI()

    # Process request
    if args.text:
        # Single text normalization
        result = cli.normalize_single(args.text, args.approach)
        cli.print_single_result(result)

    elif args.batch:
        # Batch processing
        result = cli.normalize_batch(args.input, args.approach, args.output)
        cli.print_batch_summary(result)


if __name__ == "__main__":
    main()
