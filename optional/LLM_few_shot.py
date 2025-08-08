"""
Simple LLM-based text normalizer using few-shot prompting.
No complex agents - just straightforward API calls with good examples.
"""

import json
from typing import Dict, List, Tuple

import openai
from writer_normalization_rules import WriterTextNormalizer


class SimpleLLMNormalizer:
    """
    Simple approach: Use rules first, then LLM for hard cases.
    """

    def __init__(self, api_key: str = None):
        # Initialize rule-based normalizer
        self.rule_normalizer = WriterTextNormalizer()

        # Set up OpenAI (or you can use any other LLM)
        if api_key:
            openai.api_key = api_key

        # Define few-shot examples for edge cases
        self.few_shot_examples = [
            # Example 1: Parentheses with names
            {
                "input": "Day & Murray (Bowles, Gaudet, Middleton & Shanahan)",
                "output": "Day/Murray/Bowles/Gaudet/Middleton/Shanahan",
                "explanation": "Names in parentheses should be included"
            },
            # Example 2: Various Artists should be kept
            {
                "input": "Adam Pék/Cory James/Various Artists/Copyright Control/Nick Sinna",
                "output": "Adam Pék/Cory James/Various Artists/Nick Sinna",
                "explanation": "Keep 'Various Artists' but remove publishers like 'Copyright Control'"
            },
            # Example 3: LastName, FirstName format
            {
                "input": "<Unknown>/Wright, Justyce Kaseem",
                "output": "Justyce Kaseem Wright",
                "explanation": "Fix name order and remove <Unknown>"
            },
            # Example 4: All caps names that need splitting
            {
                "input": "AHN TAI/FRICK SVEEN LOUISE/MINATOZAKI SANA",
                "output": "AHN/TAI/FRICK/SVEEN/LOUISE/MINATOZAKI/SANA",
                "explanation": "Split compound all-caps names"
            },
            # Example 5: Keep all legitimate artists
            {
                "input": "Chris Braide/Giorgio Tuinfort/David Guetta/Sia",
                "output": "Chris Braide/Giorgio Tuinfort/David Guetta/Sia",
                "explanation": "Keep all artist names, don't remove collaborators"
            },
            # Example 6: Remove numeric IDs
            {
                "input": "UNKNOWN WRITER (999990)",
                "output": "",
                "explanation": "Remove placeholder entries with numeric IDs"
            },
            # Example 7: Complex publisher names
            {
                "input": "John Doe/Sony/ATV Music Publishing (UK) LIMITED",
                "output": "John Doe",
                "explanation": "Remove company names and their variations"
            }
        ]

    def build_prompt(self, text: str) -> str:
        """
        Build a simple, effective prompt with few-shot examples.
        """
        prompt = """You are a music metadata normalizer. Your task is to clean writer/composer names by:
1. Removing publishers, companies, and entities (but keep 'Various Artists')
2. Converting separators (, & ;) to forward slash (/)
3. Fixing "LastName, FirstName" to "FirstName LastName"
4. Removing <Unknown> tags and numeric IDs
5. Keeping ALL legitimate artist/writer names

Here are examples:

"""
        # Add few-shot examples
        for example in self.few_shot_examples:
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n"
            prompt += f"Reason: {example['explanation']}\n\n"

        # Add the actual input
        prompt += "Now normalize this text:\n"
        prompt += f"Input: {text}\n"
        prompt += "Output:"

        return prompt

    def normalize_with_llm(self, text: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Use LLM to normalize text when rules aren't sufficient.
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise text normalizer. Output only the normalized text, no explanations."},
                    {"role": "user", "content": self.build_prompt(text)}
                ],
                temperature=0,  # We want consistent results
                max_tokens=200
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM error: {e}")
            # Fallback to rule-based
            return self.rule_normalizer.normalize_text(text)

    def needs_llm_processing(self, text: str, rule_result: str) -> bool:
        """
        Simple heuristic to decide if we need LLM processing.
        """
        # Cases that typically need LLM:
        # 1. Complex parentheses
        if '(' in text and ')' in text:
            return True

        # 2. All caps with spaces (might need splitting)
        parts = text.split('/')
        for part in parts:
            if part.isupper() and ' ' in part and len(part.split()) > 1:
                return True

        # 3. Contains "Various Artists" or similar special cases
        if 'various artists' in text.lower():
            return True

        # 4. Very long text with many names
        if text.count('/') + text.count('&') + text.count(',') > 5:
            return True

        # 5. Rule-based result is empty but input wasn't
        if not rule_result and text.strip():
            return True

        return False

    def normalize(self, text: str, always_use_llm: bool = False) -> str:
        """
        Main normalization function.
        
        Args:
            text: Raw text to normalize
            always_use_llm: Force LLM usage (for testing)
            
        Returns:
            Normalized text
        """
        # Step 1: Try rule-based first
        rule_result = self.rule_normalizer.normalize_text(text)

        # Step 2: Check if we need LLM
        if always_use_llm or self.needs_llm_processing(text, rule_result):
            return self.normalize_with_llm(text)

        return rule_result

    def batch_normalize(self, texts: List[str], batch_size: int = 10) -> List[str]:
        """
        Process multiple texts efficiently.
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for text in batch:
                results.append(self.normalize(text))

            # Progress
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")

        return results


# Simple usage example (no complex agent frameworks!)
if __name__ == "__main__":
    # Initialize normalizer
    normalizer = SimpleLLMNormalizer()  # Add your API key here

    # Test cases
    test_texts = [
        "Day & Murray (Bowles, Gaudet, Middleton & Shanahan)",
        "Adam Pék/Cory James/Various Artists/Copyright Control/Nick Sinna",
        "AHN TAI/FRICK SVEEN LOUISE/MINATOZAKI SANA",
        "Chris Braide/Giorgio Tuinfort/David Guetta/Sia",
        "<Unknown>/Wright, Justyce Kaseem",
        "UNKNOWN WRITER (999990)"
    ]

    print("=== Testing Hybrid Approach ===\n")

    for text in test_texts:
        # Try rule-based first
        rule_result = normalizer.rule_normalizer.normalize_text(text)

        # Check if needs LLM
        needs_llm = normalizer.needs_llm_processing(text, rule_result)

        print(f"Input: {text}")
        print(f"Rule Result: {rule_result}")
        print(f"Needs LLM: {needs_llm}")

        if needs_llm:
            # This would call the LLM
            print("LLM would process this case")

        print("-" * 50)
