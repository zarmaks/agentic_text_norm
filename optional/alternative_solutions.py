"""
Alternative solutions when Mistral doesn't work well with ReAct
"""

import re
from typing import Dict, List

import pandas as pd
from tools_legacy import TextNormalizationTools


class HybridAgent:
    """
    Hybrid approach: Use LLM for decisions, but execute tools directly
    Much faster and more reliable than full ReAct
    """

    def __init__(self):
        self.tools = TextNormalizationTools()

    def process(self, text: str) -> Dict:
        """Process with intelligent tool selection but direct execution"""

        # Handle empty/invalid input
        if not text or (isinstance(text, float)):
            return {
                "input": str(text),
                "output": "",
                "confidence": 1.0,
                "method": "empty_input",
            }

        text = str(text).strip()

        # Analyze what we need to do
        analysis = self._analyze_needs(text)

        # Execute tools based on analysis
        result = text
        tools_used = []

        if analysis["has_publishers"]:
            result = self._extract_clean_text(self.tools.remove_publishers(result))
            tools_used.append("remove_publishers")

        if analysis["has_prefixes"]:
            result = self._extract_clean_text(self.tools.remove_prefixes(result))
            tools_used.append("remove_prefixes")

        if analysis["has_inversions"]:
            result = self._extract_clean_text(self.tools.fix_name_inversions(result))
            tools_used.append("fix_inversions")

        if analysis["needs_validation"]:
            result = self._extract_clean_text(self.tools.validate_names(result))
            tools_used.append("validate_names")

        return {
            "input": text,
            "output": result,
            "confidence": 0.85,
            "method": "hybrid",
            "tools_used": tools_used,
        }

    def _analyze_needs(self, text: str) -> Dict:
        """Analyze what processing is needed"""
        return {
            "has_publishers": bool(
                re.search(r"\b(publishing|music|records|ltd|llc|corp)\b", text, re.I)
            ),
            "has_prefixes": bool(re.search(r"^(CA|PA|SE|PE|MR|DR)\s+", text, re.I)),
            "has_inversions": "," in text,
            "needs_validation": True,
        }

    def _extract_clean_text(self, tool_output: str) -> str:
        """Extract clean text from tool output"""
        # Tools return strings like "Result: 'clean text'"
        match = re.search(r"Result: '([^']*)'", tool_output)
        if match:
            return match.group(1)
        return tool_output


class SimpleRuleAgent:
    """
    Pure rule-based approach - fastest and most reliable
    No LLM needed
    """

    def __init__(self):
        self.tools = TextNormalizationTools()

    def process(self, text: str) -> Dict:
        """Simple sequential processing"""

        # Handle empty/invalid
        if not text or (isinstance(text, float)):
            return {
                "input": str(text),
                "output": "",
                "confidence": 1.0,
                "method": "empty",
            }

        text = str(text).strip()

        # Apply all tools in sequence
        result = text

        # 1. Remove special patterns first
        result = self._extract_result(self.tools.remove_special_patterns(result))

        # 2. Remove publishers
        result = self._extract_result(self.tools.remove_publishers(result))

        # 3. Remove prefixes
        result = self._extract_result(self.tools.remove_prefixes(result))

        # 4. Fix inversions
        result = self._extract_result(self.tools.fix_name_inversions(result))

        # 5. Handle ampersands
        result = self._extract_result(self.tools.handle_ampersands(result))

        # 6. Validate
        result = self._extract_result(self.tools.validate_names(result))

        return {"input": text, "output": result, "confidence": 0.8, "method": "rules"}

    def _extract_result(self, output: str) -> str:
        """Extract result from tool output"""
        match = re.search(r"Result: '([^']*)'", output)
        return match.group(1) if match else output


def compare_approaches(csv_path: str, sample_size: int = 50):
    """Compare different approaches"""

    print("Comparing Different Approaches")
    print("=" * 60)

    # Load data
    df = pd.read_csv(csv_path).head(sample_size)

    # Initialize agents
    hybrid = HybridAgent()
    rules = SimpleRuleAgent()

    # Test each approach
    results = []

    for idx, row in df.iterrows():
        raw_text = row["raw_comp_writers_text"]
        expected = row["CLEAN_TEXT"] if pd.notna(row["CLEAN_TEXT"]) else ""

        # Hybrid approach
        hybrid_result = hybrid.process(raw_text)

        # Rules approach
        rules_result = rules.process(raw_text)

        results.append(
            {
                "raw": raw_text,
                "expected": expected,
                "hybrid_output": hybrid_result["output"],
                "rules_output": rules_result["output"],
                "hybrid_correct": hybrid_result["output"] == expected,
                "rules_correct": rules_result["output"] == expected,
            }
        )

    # Calculate accuracies
    results_df = pd.DataFrame(results)

    print(f"\nResults on {sample_size} samples:")
    print(f"Hybrid Agent Accuracy: {results_df['hybrid_correct'].mean():.2%}")
    print(f"Rules Agent Accuracy: {results_df['rules_correct'].mean():.2%}")

    # Show some examples
    print("\nExample outputs:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\nInput: {r['raw']}")
        print(f"Expected: {r['expected']}")
        print(f"Hybrid: {r['hybrid_output']} {'✓' if r['hybrid_correct'] else '✗'}")
        print(f"Rules: {r['rules_output']} {'✓' if r['rules_correct'] else '✗'}")

    return results_df


def quick_process_dataset(csv_path: str, output_path: str = "results_hybrid"):
    """Quick processing with hybrid approach"""

    print("Processing with Hybrid Agent (fast)")

    df = pd.read_csv(csv_path)
    agent = HybridAgent()

    results = []
    from tqdm import tqdm

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        raw = row["raw_comp_writers_text"]
        expected = row["CLEAN_TEXT"] if pd.notna(row["CLEAN_TEXT"]) else ""

        result = agent.process(raw)

        results.append(
            {
                "raw": raw,
                "expected": expected,
                "predicted": result["output"],
                "correct": result["output"] == expected,
                "confidence": result["confidence"],
            }
        )

    results_df = pd.DataFrame(results)

    # Save results
    import os

    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(f"{output_path}/hybrid_results.csv", index=False)

    accuracy = results_df["correct"].mean()
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Results saved to {output_path}/")

    return results_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

        # Compare approaches on small sample
        print("1. Comparing approaches on 50 samples...")
        compare_approaches(csv_path, 50)

        print("\n" + "=" * 60)
        print("2. Processing full dataset with Hybrid approach...")
        print("(This is MUCH faster than LangChain agent)")

        # Process full dataset
        # quick_process_dataset(csv_path)
    else:
        print("Usage: python alternative_solutions.py <csv_path>")
