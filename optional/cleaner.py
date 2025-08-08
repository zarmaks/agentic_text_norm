"""
Rule-based text normalization for composer/writer names.

This module implements pattern-based cleaning to remove non-writer entities,
normalize name formats, and handle common text artifacts.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple


@dataclass
class CleaningRule:
    """Represents a text cleaning rule with pattern and replacement."""

    pattern: str
    replacement: str = ""
    description: str = ""
    case_sensitive: bool = False


@dataclass
class CleaningResult:
    """Result of text cleaning with confidence assessment."""

    original_text: str
    cleaned_text: str
    confidence: float
    applied_rules: List[str]
    needs_review: bool
    review_reasons: List[str]


class RuleBasedCleaner:
    """
    Rule-based text cleaner for composer/writer name normalization.
    
    This class applies a series of predefined rules to clean raw text input,
    removing publishers, copyright statements, and other non-writer entities.
    """

    def __init__(self):
        """Initialize the cleaner with predefined rules."""
        self.logger = logging.getLogger(__name__)
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Initialize cleaning rules for different entity types."""

        # Enhanced publisher list based on comprehensive dataset analysis
        self.publishers_to_remove = {
            # Copyright organizations
            'COPYRIGHT CONTROL', 'ASCAP', 'BMI', 'SESAC', 'PRS', 'SACEM', 'GEMA',
            'ZAIKS', 'APRA', 'SOCAN', 'JASRAC', 'SGAE', 'SIAE', 'SUISA',

            # Major publishers
            'SONY/ATV', 'SONY ATV', 'SONY', 'ATV', 'EMI', 'EMI MUSIC', 'EMI MUSIC PUBLISHING',
            'WARNER', 'WARNER CHAPPELL', 'WARNER BROS', 'UNIVERSAL', 'UNIVERSAL MUSIC',
            'BMG', 'BMG RIGHTS', 'KOBALT', 'CONCORD', 'IMAGEM', 'DOWNTOWN', "Deutsche Grammophon", "Decca", "Philips",
            'Polydor', 'Capitol', 'Atlantic', 'Island', 'Epic',
            'Columbia', 'RCA', 'Parlophone', 'Virgin',

            # Generic publishing terms
            'MUSIC PUBLISHING', 'PUBLISHING', 'MUSIC', 'RECORDS', 'PRODUCTIONS',
            'ENTERTAINMENT', 'SONGS', 'CATALOG', 'RIGHTS', 'MANAGEMENT',

            # Company suffixes
            'INC', 'LLC', 'LTD', 'LIMITED', 'CORP', 'CORPORATION', 'GMBH', 'SA', 'SRL',
            'CO', 'COMPANY', 'PTY', 'PLC', 'AG', 'AB', 'AS', 'NV', 'BV',

            # Other entities
            'UNKNOWN WRITER', 'UNKNOWN', '<UNKNOWN>', 'WRITER', 'CONTROL',
            'ADMINISTERED BY', 'ADMIN BY', 'OBO', 'C/O', 'DBA', 'AKA',

            # Specific entities found in analysis
            'MUSICALLIGATOR', 'BLUE STAMP MUSIC', 'A DAY A DREAM', 'VARIOUS ARTISTS',
            'MCSB TEAM', 'CONTENTID', 'PERF BY', 'PRIMARY WAVE', 'ROUND HILL',
            'SPIRIT MUSIC', 'PEER MUSIC', 'BUG MUSIC', 'WIXEN MUSIC',

            # Additional entities from testing
            'COPYRIGHT CONTROL (PRS)', 'DISTRICT 6 MUSIC PUBLISHING LTD',
            'BOWLES & HAWKES', 'BOOSEY AND HAWKES', 'BOOSEY & HAWKES',
        }

        # Compile regex patterns for efficient matching
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Pattern for numeric IDs like (999990) or 2589531
        self.numeric_id_pattern = re.compile(r'\(?\d{5,}\)?')

        # Pattern for "LastName, FirstName" format
        self.last_first_pattern = re.compile(r'^([A-Z][a-zA-Z\'-]+),\s*([A-Z][a-zA-Z\s\'-]+)$')

        # Pattern for parentheses content
        self.paren_pattern = re.compile(r'\([^)]+\)')

        # Pattern for angle brackets (like <Unknown>)
        self.angle_bracket_pattern = re.compile(r'<[^>]+>')

        # Pattern for multiple spaces
        self.multi_space_pattern = re.compile(r'\s+')

        # Create publisher removal pattern
        publisher_terms = '|'.join(re.escape(pub) for pub in self.publishers_to_remove)
        self.publisher_pattern = re.compile(
            rf'\b({publisher_terms})\b',
            re.IGNORECASE
        )

    def clean_text(self, raw_text: str) -> str:
        """
        Apply all cleaning rules to the input text.
        
        Args:
            raw_text: Raw composer/writer text to clean
            
        Returns:
            Cleaned text with non-writer entities removed
        """
        result = self.clean_text_with_confidence(raw_text)
        return result.cleaned_text

    def clean_text_with_confidence(self, raw_text: str) -> CleaningResult:
        """
        Apply all cleaning rules and return result with confidence assessment.
        
        Args:
            raw_text: Raw composer/writer text to clean
            
        Returns:
            CleaningResult with text, confidence, and metadata
        """
        null_values = ['nan', 'none', 'null', '']
        if not raw_text or str(raw_text).lower() in null_values:
            return CleaningResult(
                original_text=raw_text,
                cleaned_text="",
                confidence=0.1,
                applied_rules=[],
                needs_review=True,
                review_reasons=["Empty or null input"]
            )

        text = str(raw_text).strip()
        applied_rules = []

        # Step 1: Handle parentheses intelligently
        before_paren = text
        text = self._handle_parentheses(text)
        if text != before_paren:
            applied_rules.append("parentheses_handling")

        # Step 2: Normalize separators
        before_sep = text
        text = self._normalize_separators(text)
        if text != before_sep:
            applied_rules.append("separator_normalization")

        # Step 3: Remove publisher entities
        before_pub = text
        text = self._remove_publishers(text)
        if text != before_pub:
            applied_rules.append("publisher_removal")

        # Step 4: Split by separator and clean each part
        before_parts = text
        text = self._process_name_parts(text)
        if text != before_parts:
            applied_rules.append("name_processing")

        # Step 5: Final cleanup
        before_final = text
        text = self._final_cleanup(text)
        if text != before_final:
            applied_rules.append("final_cleanup")

        cleaned_text = text.strip()

        # Calculate confidence and review flags
        confidence = self._calculate_rule_based_confidence(raw_text, cleaned_text, applied_rules)
        needs_review, review_reasons = self._assess_review_need(raw_text, cleaned_text, confidence)

        return CleaningResult(
            original_text=raw_text,
            cleaned_text=cleaned_text,
            confidence=confidence,
            applied_rules=applied_rules,
            needs_review=needs_review,
            review_reasons=review_reasons
        )

    def _normalize_separators(self, text: str) -> str:
        """Normalize separators to use forward slash."""
        # Replace common separators with forward slash
        text = text.replace('&', '/')
        text = text.replace(';', '/')
        text = text.replace('|', '/')

        # Handle commas carefully - don't replace in "LastName, FirstName" format
        # This will be handled in individual name processing

        return text

    def _handle_parentheses(self, text: str) -> str:
        """Handle parentheses content intelligently."""
        def process_paren_content(match):
            content = match.group(0)[1:-1]  # Remove parentheses

            # Check if it contains publisher terms
            if self.publisher_pattern.search(content):
                return ''

            # Check if it's a numeric ID
            if self.numeric_id_pattern.match(content):
                return ''

            # Check if it looks like names (contains commas or &)
            if ',' in content or '&' in content:
                # It's likely a list of names, keep them
                return '/' + content

            return match.group(0)  # Keep as is

        return self.paren_pattern.sub(process_paren_content, text)

    def _remove_publishers(self, text: str) -> str:
        """Remove publisher names and entities using comprehensive list."""
        # Remove content in angle brackets first
        text = self.angle_bracket_pattern.sub('', text)

        # Remove numeric IDs
        text = self.numeric_id_pattern.sub('', text)

        # Remove publisher terms using enhanced pattern
        text = self.publisher_pattern.sub('', text)

        return text

    def _process_name_parts(self, text: str) -> str:
        """Process individual name parts after splitting."""
        if not text:
            return ""

        # Split by separator
        parts = text.split('/')

        # Clean each part
        cleaned_parts = []
        for part in parts:
            cleaned = self._clean_individual_name(part)
            if cleaned:  # Only add non-empty parts
                # Check if we need to split compound names
                split_names = self._split_compound_names(cleaned)
                cleaned_parts.extend(split_names)

        return '/'.join(cleaned_parts)

    def _clean_individual_name(self, name: str) -> str:
        """Clean an individual name component."""
        name = name.strip()

        # Skip if empty
        if not name:
            return ''

        # Fix name order if needed (LastName, FirstName -> FirstName LastName)
        name = self._fix_name_order(name)

        # Clean up spaces
        name = self.multi_space_pattern.sub(' ', name).strip()

        return name

    def _fix_name_order(self, name: str) -> str:
        """Fix 'LastName, FirstName' to 'FirstName LastName' format."""
        match = self.last_first_pattern.match(name.strip())
        if match:
            last_name, first_name = match.groups()
            return f"{first_name.strip()} {last_name.strip()}"
        return name

    def _split_compound_names(self, name: str) -> List[str]:
        """Split compound names like 'AHN TAI' into separate names."""
        # Skip if it's a known compound name or contains lowercase
        if any(c.islower() for c in name) or '-' in name:
            return [name]

        # Split ALL CAPS names that are space-separated
        words = name.split()
        if len(words) > 1 and all(word.isupper() for word in words):
            # Check if each word could be a name (not too short)
            if all(len(word) >= 3 for word in words):
                return words

        return [name]

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and validation."""
        if not text:
            return ""

        # Remove empty separators
        text = re.sub(r'/+', '/', text)
        text = re.sub(r'^/|/$', '', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _calculate_rule_based_confidence(self, original: str, cleaned: str, applied_rules: List[str]) -> float:
        """Calculate confidence score for rule-based cleaning result."""
        if not cleaned:
            return 0.1

        base_confidence = 0.6

        # Bonus for each rule applied (indicates processing happened)
        base_confidence += len(applied_rules) * 0.05

        # Check result quality indicators
        # 1. Contains clear names (capitalized words)
        name_pattern = re.compile(r'\b[A-Z][a-z]+\b')
        name_count = len(name_pattern.findall(cleaned))
        if name_count > 0:
            base_confidence += min(name_count * 0.1, 0.3)

        # 2. Proper separator usage
        if '/' in cleaned and '//' not in cleaned:
            base_confidence += 0.1

        # 3. No obvious publisher terms remaining
        publisher_terms = ['copyright', 'control', 'music', 'publishing', 'unknown', 'sony', 'universal']
        cleaned_lower = cleaned.lower()
        has_publishers = any(term in cleaned_lower for term in publisher_terms)
        if not has_publishers:
            base_confidence += 0.15

        # 4. Reasonable length reduction (indicates cleaning)
        length_ratio = len(cleaned) / max(len(original), 1)
        if 0.3 <= length_ratio <= 0.8:
            base_confidence += 0.1

        # 5. No numeric IDs
        if not re.search(r'\d{3,}', cleaned):
            base_confidence += 0.1

        # Penalties
        # 1. No change made
        if cleaned == original:
            base_confidence = min(base_confidence, 0.5)

        # 2. Contains special characters
        if re.search(r'[<>(){}[\]]', cleaned):
            base_confidence -= 0.2

        # 3. Too many ALL CAPS words
        all_caps_count = len(re.findall(r'\b[A-Z]{3,}\b', cleaned))
        if all_caps_count > 2:
            base_confidence -= 0.15

        return min(max(base_confidence, 0.0), 1.0)

    def _assess_review_need(self, original: str, cleaned: str, confidence: float) -> Tuple[bool, List[str]]:
        """Assess if manual review is needed."""
        reasons = []

        # Low confidence
        if confidence < 0.6:
            reasons.append(f"Low confidence: {confidence:.2f}")

        # Empty result
        if not cleaned:
            reasons.append("Empty result")

        # No processing happened
        if cleaned == original:
            reasons.append("No changes applied")

        # Contains problematic terms
        cleaned_lower = cleaned.lower()
        problematic_terms = ['copyright', 'control', 'unknown', 'music publishing']
        for term in problematic_terms:
            if term in cleaned_lower:
                reasons.append(f"Contains '{term}'")
                break

        # Contains numeric IDs
        if re.search(r'\d{4,}', cleaned):
            reasons.append("Contains numeric IDs")

        # Too many separators (might be over-split)
        if cleaned.count('/') > 5:
            reasons.append("Too many name separators")

        # Very short result for long input
        if len(original) > 50 and len(cleaned) < 10:
            reasons.append("Result too short for input length")

        # Complex original with simple result might need review
        complexity_indicators = ['(', '&', ';', 'copyright', 'music']
        original_lower = original.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in original_lower)

        if complexity_score >= 3 and len(cleaned.split('/')) == 1:
            reasons.append("Complex input with simple output")

        needs_review = len(reasons) > 0
        return needs_review, reasons

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        return [self.clean_text(text) for text in texts]

    def process_batch_with_confidence(self, texts: List[str]) -> List[CleaningResult]:
        """Process a batch of texts with confidence assessment."""
        return [self.clean_text_with_confidence(text) for text in texts]


# Utility functions
def load_custom_patterns(file_path: str) -> Set[str]:
    """Load custom cleaning patterns from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logging.warning(f"Pattern file not found: {file_path}")
        return set()


if __name__ == "__main__":
    # Quick test
    cleaner = RuleBasedCleaner()

    test_cases = [
        "<Unknown>/Wright, Justyce Kaseem",
        "Pixouu/Abdou Gambetta/Copyright Control",
        "Mike Hoyer/JERRY CHESNUT/SONY/ATV MUSIC PUBLISHING (UK) LIMITED"
    ]

    for test in test_cases:
        result = cleaner.clean_text(test)
        print(f"Input:  {test}")
        print(f"Output: {result}")
        print("-" * 50)
