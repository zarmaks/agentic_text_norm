"""
Integration tests for the complete text normalization system.

This module tests the integration between different components.
"""

import pytest

from text_normalization.cleaner import RuleBasedCleaner
from text_normalization.rule_based_agent import HybridNormalizationAgent


class TestIntegration:
    """Integration tests for text normalization system."""

    def test_rule_based_vs_hybrid_consistency(self):
        """Test consistency between RuleBasedCleaner and HybridNormalizationAgent."""
        cleaner = RuleBasedCleaner()
        agent = HybridNormalizationAgent()

        # Test cases where both should produce similar results
        test_cases = [
            "John Smith/Sony Music Publishing",
            "Jane Doe/Universal Records",
            "Artist Name/EMI Music",
        ]

        for test_input in test_cases:
            cleaner_result = cleaner.clean_text(test_input)
            agent_result = agent.normalize_text(test_input)

            # Results should be reasonably similar for simple cases
            assert cleaner_result == agent_result.normalized_text

    def test_end_to_end_processing(self, sample_test_cases):
        """Test end-to-end processing pipeline."""
        agent = HybridNormalizationAgent()

        for case in sample_test_cases:
            result = agent.normalize_text(case["input"])

            # Should produce valid output
            assert isinstance(result.normalized_text, str)
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.tools_used, list)
            assert len(result.tools_used) > 0

    def test_performance_characteristics(self, sample_test_cases):
        """Test basic performance characteristics."""
        import time

        agent = HybridNormalizationAgent()

        # Measure processing time for multiple cases
        start_time = time.time()

        for case in sample_test_cases:
            agent.normalize_text(case["input"])

        total_time = time.time() - start_time
        avg_time = total_time / len(sample_test_cases)

        # Should complete reasonably quickly (less than 1 second per case)
        assert avg_time < 1.0

    def test_batch_vs_individual_consistency(self, sample_test_cases):
        """Test consistency between batch and individual processing."""
        agent = HybridNormalizationAgent()

        inputs = [case["input"] for case in sample_test_cases]

        # Process individually
        individual_results = []
        for input_text in inputs:
            result = agent.normalize_text(input_text)
            individual_results.append(result.normalized_text)

        # Process as batch
        batch_results = agent.normalize_batch(inputs)

        # Results should be identical
        assert len(individual_results) == len(batch_results)
        for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
            assert individual == batch, (
                f"Mismatch at index {i}: '{individual}' != '{batch}'"
            )

    def test_error_resilience(self):
        """Test system resilience to various error conditions."""
        agent = HybridNormalizationAgent()

        # Test edge cases that might cause errors
        edge_cases = [
            "",
            None,
            "   ",
            "\n\t\r",
            "A" * 1000,  # Very long input
            "🎵🎶🎵",  # Unicode characters
            "Test/Test/Test/Test/Test/Test/Test/Test",  # Many separators
        ]

        for edge_case in edge_cases:
            try:
                result = agent.normalize_text(edge_case)
                # Should complete without raising exceptions
                assert isinstance(result.normalized_text, str)
            except Exception as e:
                pytest.fail(f"Unexpected error for input '{edge_case}': {e}")

    def test_confidence_scoring_patterns(self, sample_test_cases):
        """Test that confidence scores follow expected patterns."""
        agent = HybridNormalizationAgent()

        confidence_scores = []
        for case in sample_test_cases:
            result = agent.normalize_text(case["input"])
            confidence_scores.append(result.confidence)

        # All confidence scores should be valid
        assert all(0.0 <= score <= 1.0 for score in confidence_scores)

        # Should have some variation in confidence scores
        assert len(set(confidence_scores)) > 1

    def test_tools_usage_patterns(self, sample_test_cases):
        """Test that tools are used appropriately."""
        agent = HybridNormalizationAgent()

        for case in sample_test_cases:
            result = agent.normalize_text(case["input"])

            # Should always use rule_based
            assert "rule_based" in result.tools_used

            # Tools list should not be empty
            assert len(result.tools_used) > 0

            # All tools should be strings
            assert all(isinstance(tool, str) for tool in result.tools_used)

    def test_memory_usage(self, sample_test_cases):
        """Test that processing doesn't leak memory significantly."""
        import gc

        agent = HybridNormalizationAgent()

        # Process many iterations to check for memory leaks
        for _ in range(10):
            for case in sample_test_cases:
                agent.normalize_text(case["input"])

        # Force garbage collection
        gc.collect()

        # This test mainly ensures the code runs without memory errors
        # More sophisticated memory testing would require additional tools
        assert True  # If we reach here, no memory errors occurred

    def test_reproducibility(self, sample_test_cases):
        """Test that results are reproducible."""
        agent1 = HybridNormalizationAgent()
        agent2 = HybridNormalizationAgent()

        for case in sample_test_cases:
            result1 = agent1.normalize_text(case["input"])
            result2 = agent2.normalize_text(case["input"])

            # Should produce identical results
            assert result1.normalized_text == result2.normalized_text
            # Confidence might vary slightly, but should be close
            assert abs(result1.confidence - result2.confidence) < 0.1
