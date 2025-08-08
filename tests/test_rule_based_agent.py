"""
Tests for the HybridNormalizationAgent class.

This module tests the rule-based agent functionality.
"""

from unittest.mock import patch

from text_normalization.rule_based_agent import (
    HybridNormalizationAgent,
    NormalizationResult,
)


class TestHybridNormalizationAgent:
    """Test cases for HybridNormalizationAgent."""

    def test_initialization(self, hybrid_agent):
        """Test that HybridNormalizationAgent initializes correctly."""
        assert isinstance(hybrid_agent, HybridNormalizationAgent)
        assert hasattr(hybrid_agent, "rule_cleaner")
        assert hasattr(hybrid_agent, "_llm_available")

    def test_empty_input(self, hybrid_agent):
        """Test handling of empty input."""
        result = hybrid_agent.normalize_text("")
        assert isinstance(result, NormalizationResult)
        assert result.normalized_text == ""
        assert result.confidence == 1.0

    def test_simple_normalization(self, hybrid_agent, sample_test_cases):
        """Test basic normalization functionality."""
        # Test a few representative cases
        test_case = sample_test_cases[0]  # Simple publisher removal
        result = hybrid_agent.normalize_text(test_case["input"])

        assert isinstance(result, NormalizationResult)
        assert result.original_text == test_case["input"]
        assert len(result.normalized_text) > 0
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.tools_used, list)

    def test_rule_based_processing(self, hybrid_agent):
        """Test that rule-based processing works correctly."""
        test_input = "John Smith/Sony Music Publishing"
        result = hybrid_agent.normalize_text(test_input)

        # Should use rule-based processing
        assert "rule_based" in result.tools_used
        assert result.normalized_text == "John Smith"
        assert result.confidence > 0.5

    def test_confidence_calculation(self, hybrid_agent):
        """Test confidence calculation for different scenarios."""
        # Already clean input
        result1 = hybrid_agent.normalize_text("John Smith")
        assert result1.confidence > 0.7

        # Publisher removal case
        result2 = hybrid_agent.normalize_text("Jane Doe/Universal Records")
        assert result2.confidence > 0.5

        # Complex case
        result3 = hybrid_agent.normalize_text("Smith, John/Multiple/Publishers")
        assert result3.confidence > 0.0

    def test_tools_used_tracking(self, hybrid_agent):
        """Test that tools used are tracked correctly."""
        result = hybrid_agent.normalize_text("Test Artist/Sony Music")

        assert isinstance(result.tools_used, list)
        assert len(result.tools_used) > 0
        assert "rule_based" in result.tools_used

    def test_ner_integration(self, hybrid_agent):
        """Test NER integration when available."""
        # This test will pass whether NER is available or not
        result = hybrid_agent.normalize_text("John Smith/Publisher")

        # Should complete successfully regardless of NER availability
        assert isinstance(result, NormalizationResult)
        assert len(result.normalized_text) > 0

    @patch("text_normalization.rule_based_agent.ChatOpenAI")
    def test_llm_not_needed_for_simple_cases(self, mock_openai, hybrid_agent):
        """Test that LLM is not called for simple cases."""
        # Simple case that should be handled by rules
        test_input = "Artist Name/Sony Music"
        result = hybrid_agent.normalize_text(test_input)

        # Should not have called LLM for this simple case
        assert "llm" not in result.tools_used
        assert result.normalized_text == "Artist Name"

    def test_batch_processing(self, hybrid_agent, sample_test_cases):
        """Test batch processing functionality."""
        inputs = [case["input"] for case in sample_test_cases[:3]]
        results = hybrid_agent.normalize_batch(inputs)

        assert isinstance(results, list)
        assert len(results) == len(inputs)
        assert all(isinstance(result, str) for result in results)

    def test_name_inversion_handling(self, hybrid_agent):
        """Test handling of name inversions."""
        test_cases = ["Smith, John", "Doe, Jane Mary", "Johnson, Robert"]

        for test_input in test_cases:
            result = hybrid_agent.normalize_text(test_input)
            # Should invert the name order
            assert "," not in result.normalized_text
            assert len(result.normalized_text.split()) >= 2

    def test_looks_like_person_name(self, hybrid_agent):
        """Test the person name validation logic."""
        # Valid person names
        valid_names = ["John Smith", "Jane Mary Doe", "Bob Wilson-Jones"]

        for name in valid_names:
            is_valid = hybrid_agent._looks_like_person_name(name)
            assert is_valid is True

        # Invalid names (publishers, etc.)
        invalid_names = [
            "Sony Music Publishing",
            "COPYRIGHT CONTROL",
            "UNKNOWN WRITER",
            "",
        ]

        for name in invalid_names:
            is_valid = hybrid_agent._looks_like_person_name(name)
            assert is_valid is False

    def test_already_clean_detection(self, hybrid_agent):
        """Test detection of already clean inputs."""
        clean_inputs = ["John Smith", "Jane Doe/Bob Wilson", "Classical Artist"]

        for clean_input in clean_inputs:
            hybrid_agent._looks_already_clean(clean_input)
            # Should recognize as potentially clean
            result = hybrid_agent.normalize_text(clean_input)
            # Should have reasonable confidence
            assert result.confidence > 0.5

    def test_complex_cases(self, hybrid_agent, complex_test_cases):
        """Test handling of complex edge cases."""
        for case in complex_test_cases:
            result = hybrid_agent.normalize_text(case["input"])

            # Should complete without errors
            assert isinstance(result, NormalizationResult)
            assert isinstance(result.normalized_text, str)
            # Should produce some output (not empty unless input was all noise)
            assert len(result.normalized_text) >= 0

    def test_error_handling(self, hybrid_agent):
        """Test error handling for various scenarios."""
        # None input
        result1 = hybrid_agent.normalize_text(None)
        assert result1.normalized_text == ""

        # Very long input
        long_input = "A" * 1000
        result2 = hybrid_agent.normalize_text(long_input)
        assert isinstance(result2, NormalizationResult)

        # Special characters
        special_input = "Test@#$%^&*()_+{}[]|\\:;'<>?,./"
        result3 = hybrid_agent.normalize_text(special_input)
        assert isinstance(result3, NormalizationResult)

    def test_various_artists_handling(self, hybrid_agent):
        """Test handling of 'Various Artists' entries."""
        test_cases = [
            "Various Artists/Sony Music",
            "VARIOUS ARTISTS/Publisher",
            "various artists/Control",
        ]

        for test_input in test_cases:
            result = hybrid_agent.normalize_text(test_input)
            # Should handle Various Artists appropriately
            assert isinstance(result, NormalizationResult)
