#!/usr/bin/env python3
"""
Comprehensive 3-Way Text Normalization Research Tool

Automatically generates detailed research reports comparing:
1. Pure Rule-Based: RuleBasedCleaner (pattern matching only)
2. Hybrid Agent: HybridNormalizationAgent (rules + NER + LLM validation)
3. Agentic Agent: ImprovedTextNormalizationAgent (full LLM-powered)

Usage:
    python comprehensive_research.py [--samples N] [--output filename]

Features:
- Automatic report generation with detailed metrics
- Error categorization and analysis
- LLM usage tracking and cost estimation
- Tool usage statistics
- Performance benchmarking
- Detailed comparison tables
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add paths for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
optional_dir = current_dir / "optional"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(optional_dir))


@dataclass
class ProcessingResult:
    """Result of processing a single text with detailed metrics."""

    original_text: str
    expected_text: str
    output_text: str
    processing_time: float
    confidence: float
    tools_used: List[str]
    applied_rules: List[str]
    llm_invoked: bool
    tokens_used: int
    strategy_used: str
    error_type: Optional[str]
    is_correct: bool
    needs_review: bool


@dataclass
class ApproachMetrics:
    """Comprehensive metrics for a normalization approach."""

    name: str
    total_processed: int
    correct_results: int
    accuracy: float
    average_confidence: float
    average_processing_time: float
    total_llm_invocations: int
    total_tokens_used: int
    estimated_cost_usd: float
    error_categories: Dict[str, int]
    tool_usage_stats: Dict[str, int]
    confidence_distribution: Dict[str, int]
    processing_time_stats: Dict[str, float]


class ComprehensiveResearchTool:
    """Tool for conducting comprehensive 3-way normalization research."""

    def __init__(self):
        """Initialize the research tool."""
        self.results = {"pure_rules": [], "hybrid": [], "agentic": []}

        # Cost estimation (approximate GPT-3.5-turbo pricing)
        self.cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens

        # Initialize agents (lazy loading)
        self._agents_initialized = False
        self.pure_cleaner = None
        self.hybrid_agent = None
        self.agentic_agent = None

    def _initialize_agents(self):
        """Initialize all agents with error handling."""
        if self._agents_initialized:
            return

        try:
            print("🔧 Initializing agents...")

            # Import and initialize Pure Rules
            from optional.cleaner import RuleBasedCleaner

            self.pure_cleaner = RuleBasedCleaner()
            print("  ✅ Pure Rules (RuleBasedCleaner) initialized")

            # Import and initialize Hybrid
            from text_normalization import HybridNormalizationAgent

            self.hybrid_agent = HybridNormalizationAgent()
            print("  ✅ Hybrid Agent initialized")

            # Import and initialize Agentic
            from text_normalization.agentic_agent import ImprovedTextNormalizationAgent

            self.agentic_agent = ImprovedTextNormalizationAgent()
            print("  ✅ Agentic Agent initialized")

            self._agents_initialized = True
            print("🎯 All agents initialized successfully!\n")

        except Exception as e:
            print(f"❌ Error initializing agents: {e}")
            raise

    def normalize_with_pure_rules(self, text: str) -> ProcessingResult:
        """Process text with Pure Rules approach."""
        start_time = time.time()

        try:
            result = self.pure_cleaner.clean_text_with_confidence(text)
            processing_time = time.time() - start_time

            return ProcessingResult(
                original_text=text,
                expected_text="",  # Will be set later
                output_text=result.cleaned_text,
                processing_time=processing_time,
                confidence=getattr(result, "confidence", 0.0),
                tools_used=getattr(result, "applied_rules", []),
                applied_rules=getattr(result, "applied_rules", []),
                llm_invoked=False,
                tokens_used=0,
                strategy_used="rule_based_cleaning",
                error_type=None,  # Will be determined later
                is_correct=False,  # Will be determined later
                needs_review=getattr(result, "needs_review", False),
            )
        except Exception as e:
            return ProcessingResult(
                original_text=text,
                expected_text="",
                output_text=text,  # Return original on error
                processing_time=time.time() - start_time,
                confidence=0.0,
                tools_used=[],
                applied_rules=[],
                llm_invoked=False,
                tokens_used=0,
                strategy_used="error",
                error_type="processing_error",
                is_correct=False,
                needs_review=True,
            )

    def normalize_with_hybrid(self, text: str) -> ProcessingResult:
        """Process text with Hybrid Agent approach."""
        start_time = time.time()

        try:
            result = self.hybrid_agent.normalize_text(text)
            processing_time = time.time() - start_time

            # Estimate LLM usage (hybrid uses LLM selectively)
            llm_invoked = "llm" in getattr(result, "tools_used", [])
            tokens_estimated = 50 if llm_invoked else 0  # Conservative estimate

            return ProcessingResult(
                original_text=text,
                expected_text="",
                output_text=result.normalized_text,
                processing_time=processing_time,
                confidence=getattr(result, "confidence", 0.0),
                tools_used=getattr(result, "tools_used", []),
                applied_rules=getattr(result, "tools_used", []),
                llm_invoked=llm_invoked,
                tokens_used=tokens_estimated,
                strategy_used="hybrid_processing",
                error_type=None,
                is_correct=False,
                needs_review=False,
            )
        except Exception as e:
            return ProcessingResult(
                original_text=text,
                expected_text="",
                output_text=text,
                processing_time=time.time() - start_time,
                confidence=0.0,
                tools_used=[],
                applied_rules=[],
                llm_invoked=False,
                tokens_used=0,
                strategy_used="error",
                error_type="processing_error",
                is_correct=False,
                needs_review=True,
            )

    def normalize_with_agentic(self, text: str) -> ProcessingResult:
        """Process text with Agentic Agent approach."""
        start_time = time.time()

        try:
            result = self.agentic_agent.process(text)
            processing_time = time.time() - start_time

            return ProcessingResult(
                original_text=text,
                expected_text="",
                output_text=result.output_text,
                processing_time=processing_time,
                confidence=getattr(result, "confidence", 0.0),
                tools_used=getattr(result, "tools_used", []),
                applied_rules=getattr(result, "tools_used", []),
                llm_invoked=True,  # Agentic always uses LLM
                tokens_used=getattr(
                    result, "tokens_used", 100
                ),  # Estimate if not available
                strategy_used=getattr(result, "strategy_used", "agentic_processing"),
                error_type=None,
                is_correct=False,
                needs_review=False,
            )
        except Exception as e:
            return ProcessingResult(
                original_text=text,
                expected_text="",
                output_text=text,
                processing_time=time.time() - start_time,
                confidence=0.0,
                tools_used=[],
                applied_rules=[],
                llm_invoked=True,
                tokens_used=50,  # Estimate for failed attempt
                strategy_used="error",
                error_type="processing_error",
                is_correct=False,
                needs_review=True,
            )

    def categorize_error(
        self, original: str, expected: str, actual: str
    ) -> Optional[str]:
        """Categorize the type of error based on comparison."""
        if actual == expected:
            return None  # No error

        original_parts = [p.strip() for p in original.split("/") if p.strip()]
        expected_parts = [p.strip() for p in expected.split("/") if p.strip()]
        actual_parts = [p.strip() for p in actual.split("/") if p.strip()]

        # Over-segmentation: more parts than expected
        if len(actual_parts) > len(expected_parts):
            return "over_segmentation"

        # Under-segmentation: fewer parts than expected
        elif len(actual_parts) < len(expected_parts):
            return "under_segmentation"

        # Publisher removal issues
        elif any(
            "music" in part.lower()
            or "record" in part.lower()
            or "control" in part.lower()
            for part in actual_parts
        ):
            return "failed_publisher_removal"

        # Name inversion issues
        elif any("," in part for part in actual_parts):
            return "failed_name_inversion"

        # Special character issues
        elif any(char in actual for char in "<>()[]{}"):
            return "failed_special_char_removal"

        # Normalization issues (wrong format)
        elif actual != expected and len(actual_parts) == len(expected_parts):
            return "normalization_error"

        # Unknown error type
        else:
            return "unknown_error"

    def process_dataset(
        self, df: pd.DataFrame, max_samples: Optional[int] = None
    ) -> Dict[str, List[ProcessingResult]]:
        """Process the dataset with all three approaches."""
        # Filter out rows with empty CLEAN_TEXT
        df_clean = df.dropna(subset=["CLEAN_TEXT"])
        df_clean = df_clean[df_clean["CLEAN_TEXT"].str.strip() != ""]

        if max_samples:
            df_clean = df_clean.head(max_samples)

        total_samples = len(df_clean)
        print(f"📊 Processing {total_samples} samples...")

        results = {"pure_rules": [], "hybrid": [], "agentic": []}

        for idx, row in df_clean.iterrows():
            raw_text = row["raw_comp_writers_text"]
            expected_text = row["CLEAN_TEXT"]

            print(
                f"\r⏳ Processing sample {idx + 1}/{total_samples}...",
                end="",
                flush=True,
            )

            # Process with all three approaches
            for approach_name, process_func in [
                ("pure_rules", self.normalize_with_pure_rules),
                ("hybrid", self.normalize_with_hybrid),
                ("agentic", self.normalize_with_agentic),
            ]:
                result = process_func(raw_text)
                result.expected_text = expected_text
                result.is_correct = result.output_text.strip() == expected_text.strip()
                result.error_type = self.categorize_error(
                    raw_text, expected_text, result.output_text
                )

                results[approach_name].append(result)

        print(f"\n✅ Completed processing {total_samples} samples!")
        return results

    def calculate_metrics(
        self, results: List[ProcessingResult], approach_name: str
    ) -> ApproachMetrics:
        """Calculate comprehensive metrics for an approach."""
        if not results:
            return ApproachMetrics(
                name=approach_name,
                total_processed=0,
                correct_results=0,
                accuracy=0.0,
                average_confidence=0.0,
                average_processing_time=0.0,
                total_llm_invocations=0,
                total_tokens_used=0,
                estimated_cost_usd=0.0,
                error_categories={},
                tool_usage_stats={},
                confidence_distribution={},
                processing_time_stats={},
            )

        # Basic metrics
        total_processed = len(results)
        correct_results = sum(1 for r in results if r.is_correct)
        accuracy = correct_results / total_processed if total_processed > 0 else 0.0

        # Confidence and timing
        confidences = [r.confidence for r in results]
        times = [r.processing_time for r in results]

        average_confidence = np.mean(confidences) if confidences else 0.0
        average_processing_time = np.mean(times) if times else 0.0

        # LLM usage
        total_llm_invocations = sum(1 for r in results if r.llm_invoked)
        total_tokens_used = sum(r.tokens_used for r in results)
        estimated_cost_usd = (total_tokens_used / 1000) * self.cost_per_1k_tokens

        # Error categorization
        error_categories = {}
        for result in results:
            if result.error_type:
                error_categories[result.error_type] = (
                    error_categories.get(result.error_type, 0) + 1
                )

        # Tool usage statistics
        tool_usage_stats = {}
        for result in results:
            for tool in result.tools_used:
                tool_usage_stats[tool] = tool_usage_stats.get(tool, 0) + 1

        # Confidence distribution
        confidence_distribution = {
            "very_low (0.0-0.3)": sum(1 for c in confidences if 0.0 <= c < 0.3),
            "low (0.3-0.5)": sum(1 for c in confidences if 0.3 <= c < 0.5),
            "medium (0.5-0.7)": sum(1 for c in confidences if 0.5 <= c < 0.7),
            "high (0.7-0.9)": sum(1 for c in confidences if 0.7 <= c < 0.9),
            "very_high (0.9-1.0)": sum(1 for c in confidences if 0.9 <= c <= 1.0),
        }

        # Processing time statistics
        processing_time_stats = {
            "min": np.min(times) if times else 0.0,
            "max": np.max(times) if times else 0.0,
            "median": np.median(times) if times else 0.0,
            "std": np.std(times) if times else 0.0,
        }

        return ApproachMetrics(
            name=approach_name,
            total_processed=total_processed,
            correct_results=correct_results,
            accuracy=accuracy,
            average_confidence=average_confidence,
            average_processing_time=average_processing_time,
            total_llm_invocations=total_llm_invocations,
            total_tokens_used=total_tokens_used,
            estimated_cost_usd=estimated_cost_usd,
            error_categories=error_categories,
            tool_usage_stats=tool_usage_stats,
            confidence_distribution=confidence_distribution,
            processing_time_stats=processing_time_stats,
        )

    def generate_report(
        self,
        results: Dict[str, List[ProcessingResult]],
        metrics: Dict[str, ApproachMetrics],
        output_file: str,
    ) -> None:
        """Generate comprehensive research report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_content = f"""# Comprehensive Text Normalization Research Report

**Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}  
**Dataset:** normalization_assessment_dataset_10k.csv  
**Total Samples Processed:** {metrics['pure_rules'].total_processed}  

## Executive Summary

This report presents a comprehensive comparison of three text normalization approaches:

1. **Pure Rule-Based**: Pattern matching and predefined rules only
2. **Hybrid Agent**: Rules + NER + selective LLM validation  
3. **Agentic Agent**: Full LLM-powered with tool usage

### Key Findings

| Approach | Accuracy | Avg Confidence | LLM Usage | Est. Cost | Avg Time |
|----------|----------|----------------|-----------|-----------|----------|
| Pure Rules | {metrics['pure_rules'].accuracy:.1%} | {metrics['pure_rules'].average_confidence:.3f} | {metrics['pure_rules'].total_llm_invocations} | ${metrics['pure_rules'].estimated_cost_usd:.4f} | {metrics['pure_rules'].average_processing_time:.3f}s |
| Hybrid | {metrics['hybrid'].accuracy:.1%} | {metrics['hybrid'].average_confidence:.3f} | {metrics['hybrid'].total_llm_invocations} | ${metrics['hybrid'].estimated_cost_usd:.4f} | {metrics['hybrid'].average_processing_time:.3f}s |
| Agentic | {metrics['agentic'].accuracy:.1%} | {metrics['agentic'].average_confidence:.3f} | {metrics['agentic'].total_llm_invocations} | ${metrics['agentic'].estimated_cost_usd:.4f} | {metrics['agentic'].average_processing_time:.3f}s |

---

## Detailed Analysis

### 1. Pure Rule-Based Approach

**Performance:**
- Accuracy: {metrics['pure_rules'].accuracy:.1%} ({metrics['pure_rules'].correct_results}/{metrics['pure_rules'].total_processed})
- Average Confidence: {metrics['pure_rules'].average_confidence:.3f}
- Processing Speed: {metrics['pure_rules'].average_processing_time:.3f}s per sample

**Characteristics:**
- No LLM usage (0 invocations)
- Zero cost operation
- Deterministic results
- Fast processing

**Error Analysis:**
"""

        # Add error breakdown for Pure Rules
        if metrics["pure_rules"].error_categories:
            report_content += "\n**Error Categories:**\n"
            for error_type, count in sorted(
                metrics["pure_rules"].error_categories.items()
            ):
                percentage = (count / metrics["pure_rules"].total_processed) * 100
                report_content += f"- {error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n"
        else:
            report_content += "\n**No errors detected**\n"

        report_content += """

**Tool Usage:**
"""
        if metrics["pure_rules"].tool_usage_stats:
            for tool, count in sorted(metrics["pure_rules"].tool_usage_stats.items()):
                report_content += f"- {tool}: {count} times\n"

        report_content += """

**Confidence Distribution:**
"""
        for conf_range, count in metrics["pure_rules"].confidence_distribution.items():
            percentage = (
                (count / metrics["pure_rules"].total_processed) * 100
                if metrics["pure_rules"].total_processed > 0
                else 0
            )
            report_content += f"- {conf_range}: {count} ({percentage:.1f}%)\n"

        report_content += f"""

### 2. Hybrid Agent Approach

**Performance:**
- Accuracy: {metrics['hybrid'].accuracy:.1%} ({metrics['hybrid'].correct_results}/{metrics['hybrid'].total_processed})
- Average Confidence: {metrics['hybrid'].average_confidence:.3f}
- Processing Speed: {metrics['hybrid'].average_processing_time:.3f}s per sample

**LLM Usage:**
- Total Invocations: {metrics['hybrid'].total_llm_invocations}
- Total Tokens: {metrics['hybrid'].total_tokens_used:,}
- Estimated Cost: ${metrics['hybrid'].estimated_cost_usd:.4f}
- LLM Usage Rate: {(metrics['hybrid'].total_llm_invocations/metrics['hybrid'].total_processed)*100:.1f}%

**Error Analysis:**
"""

        # Add error breakdown for Hybrid
        if metrics["hybrid"].error_categories:
            report_content += "\n**Error Categories:**\n"
            for error_type, count in sorted(metrics["hybrid"].error_categories.items()):
                percentage = (count / metrics["hybrid"].total_processed) * 100
                report_content += f"- {error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n"
        else:
            report_content += "\n**No errors detected**\n"

        report_content += """

**Tool Usage:**
"""
        if metrics["hybrid"].tool_usage_stats:
            for tool, count in sorted(metrics["hybrid"].tool_usage_stats.items()):
                report_content += f"- {tool}: {count} times\n"

        report_content += f"""

### 3. Agentic Agent Approach

**Performance:**
- Accuracy: {metrics['agentic'].accuracy:.1%} ({metrics['agentic'].correct_results}/{metrics['agentic'].total_processed})
- Average Confidence: {metrics['agentic'].average_confidence:.3f}
- Processing Speed: {metrics['agentic'].average_processing_time:.3f}s per sample

**LLM Usage:**
- Total Invocations: {metrics['agentic'].total_llm_invocations}
- Total Tokens: {metrics['agentic'].total_tokens_used:,}
- Estimated Cost: ${metrics['agentic'].estimated_cost_usd:.4f}
- LLM Usage Rate: {(metrics['agentic'].total_llm_invocations/metrics['agentic'].total_processed)*100:.1f}%

**Error Analysis:**
"""

        # Add error breakdown for Agentic
        if metrics["agentic"].error_categories:
            report_content += "\n**Error Categories:**\n"
            for error_type, count in sorted(
                metrics["agentic"].error_categories.items()
            ):
                percentage = (count / metrics["agentic"].total_processed) * 100
                report_content += f"- {error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n"
        else:
            report_content += "\n**No errors detected**\n"

        report_content += """

**Tool Usage:**
"""
        if metrics["agentic"].tool_usage_stats:
            for tool, count in sorted(metrics["agentic"].tool_usage_stats.items()):
                report_content += f"- {tool}: {count} times\n"

        report_content += f"""

---

## Comparative Analysis

### Accuracy Comparison
1. **{max(metrics.items(), key=lambda x: x[1].accuracy)[0].replace('_', ' ').title()}**: {max(m.accuracy for m in metrics.values()):.1%} (Best)
2. **{sorted(metrics.items(), key=lambda x: x[1].accuracy, reverse=True)[1][0].replace('_', ' ').title()}**: {sorted(metrics.values(), key=lambda x: x.accuracy, reverse=True)[1].accuracy:.1%}
3. **{min(metrics.items(), key=lambda x: x[1].accuracy)[0].replace('_', ' ').title()}**: {min(m.accuracy for m in metrics.values()):.1%}

### Cost-Effectiveness Analysis
- **Pure Rules**: ${metrics['pure_rules'].estimated_cost_usd:.4f} (Free operation)
- **Hybrid**: ${metrics['hybrid'].estimated_cost_usd:.4f} ({metrics['hybrid'].total_tokens_used:,} tokens)
- **Agentic**: ${metrics['agentic'].estimated_cost_usd:.4f} ({metrics['agentic'].total_tokens_used:,} tokens)

### Performance vs Cost Trade-off
"""

        # Calculate efficiency metrics
        for name, metric in metrics.items():
            if metric.estimated_cost_usd > 0:
                accuracy_per_dollar = metric.accuracy / metric.estimated_cost_usd
                report_content += f"- **{name.replace('_', ' ').title()}**: {accuracy_per_dollar:.0f} accuracy points per dollar\n"
            else:
                report_content += f"- **{name.replace('_', ' ').title()}**: Infinite (free operation)\n"

        report_content += f"""

### Processing Speed Comparison
- **Fastest**: {min(metrics.items(), key=lambda x: x[1].average_processing_time)[0].replace('_', ' ').title()} ({min(m.average_processing_time for m in metrics.values()):.3f}s avg)
- **Slowest**: {max(metrics.items(), key=lambda x: x[1].average_processing_time)[0].replace('_', ' ').title()} ({max(m.average_processing_time for m in metrics.values()):.3f}s avg)

---

## Recommendations

### For Production Use:
"""

        # Generate recommendations based on results
        best_accuracy = max(metrics.items(), key=lambda x: x[1].accuracy)
        fastest = min(metrics.items(), key=lambda x: x[1].average_processing_time)
        cheapest = min(metrics.items(), key=lambda x: x[1].estimated_cost_usd)

        report_content += f"""
1. **Highest Accuracy**: Use **{best_accuracy[0].replace('_', ' ').title()}** for maximum quality ({best_accuracy[1].accuracy:.1%} accuracy)
2. **Fastest Processing**: Use **{fastest[0].replace('_', ' ').title()}** for speed-critical applications ({fastest[1].average_processing_time:.3f}s per sample)
3. **Most Cost-Effective**: Use **{cheapest[0].replace('_', ' ').title()}** for budget-conscious deployments (${cheapest[1].estimated_cost_usd:.4f} cost)

### Use Case Scenarios:
- **High-volume, cost-sensitive**: Pure Rule-Based approach
- **Balanced quality and cost**: Hybrid Agent approach  
- **Maximum quality, cost-tolerant**: Agentic Agent approach

---

## Technical Details

### Dataset Information
- **Source**: normalization_assessment_dataset_10k.csv
- **Total records**: {metrics['pure_rules'].total_processed}
- **Clean records processed**: {metrics['pure_rules'].total_processed} (empty CLEAN_TEXT excluded)

### Processing Environment
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Tool**: Comprehensive Research Framework v1.0
- **LLM Model**: GPT-3.5-turbo (estimated pricing)

### Methodology
1. Load dataset and filter valid records
2. Process each record with all three approaches
3. Compare outputs against expected CLEAN_TEXT
4. Categorize errors and analyze patterns
5. Calculate comprehensive metrics and statistics

---

*Report generated automatically by Comprehensive Text Normalization Research Tool*
"""

        # Write the report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"📄 Comprehensive report saved to: {output_file}")

    def run_research(
        self, max_samples: Optional[int] = None, output_file: Optional[str] = None
    ) -> None:
        """Run the complete research study."""
        # Initialize agents
        self._initialize_agents()

        # Load dataset
        print("📊 Loading dataset...")
        try:
            df = pd.read_csv("data/normalization_assesment_dataset_10k.csv")
            print(f"✅ Dataset loaded: {len(df)} total records")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return

        # Process dataset
        results = self.process_dataset(df, max_samples)

        # Calculate metrics
        print("\n📈 Calculating metrics...")
        metrics = {}
        for approach_name, approach_results in results.items():
            metrics[approach_name] = self.calculate_metrics(
                approach_results, approach_name
            )
            print(
                f"  ✅ {approach_name.replace('_', ' ').title()}: {metrics[approach_name].accuracy:.1%} accuracy"
            )

        # Generate report
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ensure reports directory exists
            Path("reports").mkdir(exist_ok=True)
            output_file = f"reports/comprehensive_research_report_{timestamp}.md"

        print("\n📝 Generating comprehensive report...")
        self.generate_report(results, metrics, output_file)

        # Print summary
        print("\n🎉 Research completed successfully!")
        print(f"📊 Processed {metrics['pure_rules'].total_processed} samples")
        print(f"📄 Report saved to: {output_file}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Text Normalization Research Tool"
    )
    parser.add_argument(
        "--samples", "-s", type=int, help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output filename (default: auto-generated)"
    )

    args = parser.parse_args()

    print("🔬 Comprehensive Text Normalization Research Tool")
    print("=" * 60)

    research_tool = ComprehensiveResearchTool()
    research_tool.run_research(max_samples=args.samples, output_file=args.output)


if __name__ == "__main__":
    main()
    main()
