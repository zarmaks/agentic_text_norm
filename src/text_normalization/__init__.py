"""
Text Normalization Package

A comprehensive implementation and comparison of text normalization techniques
for composer/writer names in the music industry.
"""

__version__ = "1.0.0"
__author__ = "Text Normalization Research Team"

from .agentic_agent import ImprovedTextNormalizationAgent
from .cleaner import RuleBasedCleaner
from .rule_based_agent import HybridNormalizationAgent

__all__ = [
    "ImprovedTextNormalizationAgent",
    "HybridNormalizationAgent",
    "RuleBasedCleaner",
]
