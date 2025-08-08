"""
Test configuration and fixtures for text normalization tests.
"""

import os
from unittest.mock import Mock

import pytest

from text_normalization.cleaner import RuleBasedCleaner
from text_normalization.rule_based_agent import HybridNormalizationAgent


@pytest.fixture
def rule_based_cleaner():
    """Fixture providing a RuleBasedCleaner instance."""
    return RuleBasedCleaner()


@pytest.fixture
def hybrid_agent():
    """Fixture providing a HybridNormalizationAgent instance."""
    return HybridNormalizationAgent()


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API to avoid API calls during testing."""
    mock = Mock()
    mock.invoke.return_value.content = "Test Output"
    return mock


@pytest.fixture
def sample_test_cases():
    """Fixture providing sample test cases for normalization."""
    return [
        {
            "input": "John Smith/Sony Music Publishing",
            "expected_rule": "John Smith",
            "expected_agentic": "John Smith",
            "description": "Simple publisher removal",
        },
        {
            "input": "Larry June,The Alchemist,Boldy James",
            "expected_rule": "Larry June,The Alchemist,Boldy James",
            "expected_agentic": "Larry June/The Alchemist/Boldy James",
            "description": "Comma to slash normalization",
        },
        {
            "input": "<Unknown>/Mincey, Jeremy",
            "expected_rule": "Jeremy Mincey",
            "expected_agentic": "Mincey/Jeremy",
            "description": "Name inversion vs structure preservation",
        },
        {
            "input": "Jordan Riley/Adam Argyle/Martin Brammer",
            "expected_rule": "Jordan Riley/Adam Argyle/Martin Brammer",
            "expected_agentic": "Jordan Riley/Adam Argyle/Martin Brammer",
            "description": "Clean input - no changes needed",
        },
        {
            "input": "Lavel Jackson & Demarcus Ford",
            "expected_rule": "Lavel Jackson/Demarcus Ford",
            "expected_agentic": "Lavel Jackson/Demarcus Ford",
            "description": "Ampersand normalization",
        },
    ]


@pytest.fixture
def publisher_entities():
    """Fixture providing sample publisher entities for testing."""
    return [
        "Sony Music Publishing",
        "Universal Records",
        "EMI Music",
        "Warner Chappell",
        "BMG Rights",
        "COPYRIGHT CONTROL",
        "ASCAP",
        "BMI",
        "SESAC",
    ]


@pytest.fixture
def complex_test_cases():
    """Fixture providing complex edge cases for testing."""
    return [
        {
            "input": "Day & Murray (Bowles, Gaudet, Middleton & Shanahan)",
            "description": "Complex parentheses with multiple names and ampersands",
        },
        {
            "input": "UNKNOWN WRITER (999990)",
            "description": "Unknown writer with numeric ID",
        },
        {
            "input": "Various Artists/Copyright Control",
            "description": "Various Artists preservation with publisher removal",
        },
        {
            "input": "Smith, John/Universal Records",
            "description": "Name inversion with publisher",
        },
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for each test."""
    # Ensure we don't accidentally make real API calls
    if "OPENAI_API_KEY" in os.environ:
        original_key = os.environ["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = "test-key-not-real"
        yield
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        os.environ["OPENAI_API_KEY"] = "test-key-not-real"
        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
