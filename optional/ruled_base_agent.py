"""
Improved hybrid text normalization agent with realistic confidence scoring.

This version uses rule-based cleaning for most cases and LLM only for 
genuinely complex scenarios, with honest confidence assessment.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from .cleaner import RuleBasedCleaner
except ImportError:
    from cleaner import RuleBasedCleaner

# Load environment variables
load_dotenv(".env", override=True)


@dataclass
class NormalizationResult:
    """Result of text normalization with confidence and reasoning."""
    original_text: str
    normalized_text: str
    confidence: float
    reasoning: str
    tools_used: List[str]


class HybridNormalizationAgent:
    """
    Hybrid agent with realistic confidence scoring.
    
    Key improvements:
    - Evidence-based confidence calculation
    - No artificial confidence inflation
    - Honest assessment of LLM vs rule-based performance
    """

    def __init__(self):
        """Initialize the hybrid agent."""
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.rule_cleaner = RuleBasedCleaner()

        # Initialize NER model (optional)
        self._init_ner_model()

        # Initialize LLM (lazy loading)
        self._llm = None
        self._llm_available = True

        # Few-shot examples for complex cases
        self.few_shot_examples = [
            {
                "input": "Day & Murray (Bowles, Gaudet, Middleton & Shanahan)",
                "output": "Day/Murray/Bowles/Gaudet/Middleton/Shanahan",
                "reason": "Extract names from parentheses, use / separator"
            },
            {
                "input": "Adam Pék/Cory James/Various Artists/Copyright Control",
                "output": "Adam Pék/Cory James/Various Artists",
                "reason": "Keep 'Various Artists', remove publishers"
            },
            {
                "input": "<Unknown>/Wright, Justyce Kaseem",
                "output": "Justyce Kaseem Wright",
                "reason": "Fix name order, remove <Unknown> tags"
            }
        ]

    def _init_ner_model(self):
        """Initialize NER model for person name validation."""
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self.logger.info("NER model loaded successfully")
        except Exception as e:
            self.logger.warning(f"NER model not available: {e}")
            self._nlp = None

    def _extract_person_names_ner(self, text: str) -> List[str]:
        """Extract person names using NER as validation step."""
        if not self._nlp:
            return []

        try:
            doc = self._nlp(text)
            person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            return person_names
        except Exception as e:
            self.logger.warning(f"NER extraction failed: {e}")
            return []

    @property
    def llm(self):
        """Lazy loading of LLM."""
        if self._llm is None and self._llm_available:
            try:
                self._llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            except Exception as e:
                self.logger.warning(f"LLM initialization failed: {e}")
                self._llm_available = False
        return self._llm

    def normalize_text(self, text: str) -> NormalizationResult:
        """
        Normalize text using hybrid approach with realistic confidence.
        """
        if not text or not text.strip():
            return NormalizationResult("", "", 1.0, "Empty input", [])

        original_text = text

        # Step 1: Always try rule-based first
        rule_result = self.rule_cleaner.clean_text(text)
        tools_used = ["rule_based"]

        # Step 2: NER validation for person names
        ner_persons = self._extract_person_names_ner(text)
        if ner_persons:
            tools_used.append("ner_validation")

        # Step 3: Check if LLM is needed for complex cases
        if self._needs_llm_processing(text, rule_result):
            llm_result = self._process_with_llm(text)
            if llm_result:
                llm_tools = tools_used + ["llm"]
                llm_confidence = self._calculate_realistic_confidence(
                    text, llm_result, llm_tools, ner_persons
                )
                return NormalizationResult(
                    original_text=original_text,
                    normalized_text=llm_result,
                    confidence=llm_confidence,
                    reasoning="LLM processing for complex case",
                    tools_used=llm_tools
                )

        # Step 4: Return rule-based result with realistic confidence
        final_confidence = self._calculate_realistic_confidence(
            text, rule_result, tools_used, ner_persons
        )
        return NormalizationResult(
            original_text=original_text,
            normalized_text=rule_result,
            confidence=final_confidence,
            reasoning="Rule-based processing",
            tools_used=tools_used
        )

    def _calculate_realistic_confidence(self, original: str, normalized: str,
                                      tools_used: List[str],
                                      ner_persons: List[str]) -> float:
        """
        Calculate realistic confidence based on evidence, not artificial inflation.
        """
        if not normalized:
            return 0.1  # Very low confidence for empty results

        if normalized == original:
            # No change - could be already clean OR processing failed
            if self._looks_already_clean(original):
                return 0.85  # High confidence for already clean names
            else:
                return 0.3  # Low confidence - likely processing failure

        # Base confidence based on content analysis
        base_confidence = 0.6

        # Evidence-based adjustments

        # 1. Check if result looks like valid person names
        normalized_parts = [p.strip() for p in normalized.split('/') if p.strip()]
        if all(self._looks_like_person_name(name) for name in normalized_parts):
            base_confidence += 0.15

        # 2. Successful reduction (removed publishers/noise)
        if len(normalized) < len(original):
            base_confidence += 0.1

        # 3. NER validation boost (modest, evidence-based)
        if ner_persons and "ner_validation" in tools_used:
            ner_coverage = sum(1 for person in ner_persons
                             if person.lower() in normalized.lower())
            if ner_coverage > 0:
                base_confidence += min(0.05, ner_coverage * 0.02)

        # 4. LLM tool assessment (honest evaluation)
        if "llm" in tools_used:
            if self._llm_result_quality_check(normalized, original):
                base_confidence += 0.05  # Modest boost for good LLM result
            else:
                base_confidence -= 0.15  # Penalty for poor LLM result

        # 5. Special case handling
        if "various artists" in normalized.lower() and "various artists" in original.lower():
            base_confidence += 0.05  # Correctly preserved

        # 6. Quality penalties
        if len(normalized) > len(original) * 1.5:
            base_confidence -= 0.1  # Suspicious expansion

        if any(keyword in normalized.lower() for keyword in
               ['control', 'publishing', 'music inc', 'ltd']):
            base_confidence -= 0.2  # Failed to remove publishers

        return max(0.1, min(1.0, base_confidence))

    def _looks_already_clean(self, text: str) -> bool:
        """Check if input text already looks like clean person names."""
        parts = [p.strip() for p in text.split('/') if p.strip()]

        # Single name or multiple proper names
        if len(parts) <= 3 and all(self._looks_like_person_name(p) for p in parts):
            # No obvious publishers or noise
            if not any(keyword in text.lower() for keyword in
                      ['control', 'publishing', 'music', 'records', 'entertainment']):
                return True
        return False

    def _looks_like_person_name(self, name: str) -> bool:
        """Check if a string looks like a person name."""
        if not name or len(name.strip()) < 2:
            return False

        name = name.strip()

        # Should have at least one capital letter
        if not any(c.isupper() for c in name):
            return False

        # Should not contain obvious publisher keywords
        publisher_keywords = ['control', 'publishing', 'music', 'records',
                            'entertainment', 'ltd', 'inc', 'corp', 'llc']
        if any(keyword in name.lower() for keyword in publisher_keywords):
            return False

        # Should not be all caps unless very short
        if name.isupper() and len(name) > 8:
            return False

        # Should not have excessive length
        if len(name) > 50:
            return False

        return True

    def _llm_result_quality_check(self, normalized: str, original: str) -> bool:
        """Assess if LLM processing produced a quality result."""
        if not normalized or normalized == original:
            return False

        # Should produce readable names
        normalized_parts = [p.strip() for p in normalized.split('/') if p.strip()]
        if not normalized_parts:
            return False

        # All parts should look like names
        if not all(self._looks_like_person_name(name) for name in normalized_parts):
            return False

        # Should be reasonable length
        if len(normalized) > len(original) * 2:
            return False

        return True

    def _needs_llm_processing(self, text: str, rule_result: str) -> bool:
        """
        Determine if LLM processing is needed - conservative approach.
        """
        if not self._llm_available:
            return False

        # Only use LLM for genuinely complex cases where rules likely failed

        # 1. Complex parentheses with multiple names
        if '(' in text and ')' in text:
            import re
            paren_content = re.findall(r'\\((.*?)\\)', text)
            for content in paren_content:
                if (',' in content or '&' in content) and len(content) > 10:
                    return True

        # 2. Various Artists preservation issues
        if ('various artists' in text.lower() and
            'various artists' not in rule_result.lower()):
            return True

        # 3. Name ordering issues (LastName, FirstName)
        if any(',' in part and len(part.split(',')) == 2
               for part in text.split('/')):
            return True

        # 4. Very poor rule-based result (empty when input was meaningful)
        if not rule_result and len(text) > 10 and not text.startswith('<'):
            return True

        return False

    def _process_with_llm(self, text: str) -> Optional[str]:
        """Process text with LLM using few-shot prompting."""
        if not self.llm:
            return None

        try:
            prompt = self._build_llm_prompt(text)
            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                result = response.content.strip()
            else:
                result = str(response).strip()

            # Basic validation
            if result and len(result) > 0:
                return result

        except Exception as e:
            self.logger.warning(f"LLM processing failed for '{text}': {e}")

        return None

    def _build_llm_prompt(self, text: str) -> str:
        """Build few-shot prompt for LLM processing."""
        prompt = """You are a composer/writer name normalizer. Clean the text to extract only person names.

RULES:
- Use "/" to separate multiple names (NEVER commas)
- Remove publishers, labels, organizations
- Keep "Various Artists" when it refers to multiple artists
- Fix name ordering: "Last, First" → "First Last"  
- Remove <Unknown> tags and publisher info
- Convert ALL CAPS names to proper case

Examples:
"""

        for example in self.few_shot_examples:
            prompt += f"Input: {example['input']}\\n"
            prompt += f"Output: {example['output']}\\n\\n"

        prompt += f"Input: {text}\\n"
        prompt += "Output:"

        return prompt

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a batch of texts."""
        results = []
        for text in texts:
            result = self.normalize_text(text)
            results.append(result.normalized_text)
        return results


if __name__ == "__main__":
    # Test the improved hybrid agent
    agent = HybridNormalizationAgent()

    test_cases = [
        "John Smith/Sony Music",  # Simple - should be high confidence rule-based
        "Day & Murray (Bowles, Gaudet, Middleton & Shanahan)",  # Complex - needs LLM
        "David Guetta/Sia",  # Already clean - should be high confidence
        "UNKNOWN WRITER/Control",  # Low quality - should be low confidence
    ]

    print("🔧 Improved Hybrid Agent with Realistic Confidence")
    print("=" * 55)

    for case in test_cases:
        result = agent.normalize_text(case)
        print(f"\\nInput: {case}")
        print(f"Output: {result.normalized_text}")
        print(f"Confidence: {result.confidence:.3f} (realistic)")
        print(f"Tools: {result.tools_used}")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 55)
