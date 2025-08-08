#!/usr/bin/env python3
"""
Improved Real Agent with intelligent fallback and better decision making.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, List

from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

# Our tools
from .tools import TextNormalizationTools

load_dotenv(".env", override=True)


@dataclass
class NormalizationResult:
    """Result of text normalization."""

    original_text: str
    output_text: str
    confidence: float
    processing_time: float
    tools_used: List[str]
    tokens_used: int
    strategy_used: str  # "agent", "fallback", "direct"


class SmartFallbackLLM:
    """
    Smart fallback LLM with few-shot examples for edge cases.
    """

    def __init__(self, llm: Any) -> None:
        """
        Initialize the fallback LLM with few-shot examples.

        Args:
            llm: The language model to use for fallback processing
        """
        self.llm = llm

        # Few-shot examples for difficult cases
        self.few_shot_examples = [
            {
                "input": "Eminem",
                "output": "Eminem",
                "explanation": "Artist/band names should be preserved even if NER doesn't recognize them",
            },
            {
                "input": "Radiohead",
                "output": "Radiohead",
                "explanation": "Band names are valid even if they don't look like person names",
            },
            {
                "input": "BLACKPINK",
                "output": "BLACKPINK",
                "explanation": "Keep artistic names in their original format",
            },
            {
                "input": "Taylor Swift, Ed Sheeran, Bruno Mars & Lady Gaga",
                "output": "Taylor Swift/Ed Sheeran/Bruno Mars/Lady Gaga",
                "explanation": "Convert comma and ampersand separators to forward slash, no extra spaces",
            },
            {
                "input": "Post Malone",
                "output": "Post Malone",
                "explanation": "Artistic names should be kept as-is",
            },
            {
                "input": "Williams & Jones (Smith, Davis, Miller & Wilson)",
                "output": "Williams/Jones/Smith/Davis/Miller/Wilson",
                "explanation": "Extract names from parentheses and normalize separators",
            },
            {
                "input": "<Unknown>/Johnson, Michael",
                "output": "Michael Johnson",
                "explanation": "Remove <Unknown> and fix name order",
            },
        ]

    def build_fallback_prompt(self, text: str) -> str:
        """Build prompt for fallback LLM processing.

        Args:
            text: Input text to create prompt for

        Returns:
            Formatted prompt string for LLM processing
        """
        examples_text = "\n".join(
            [
                f"Input: {ex['input']}\nOutput: {ex['output']}\n"
                f"Reason: {ex['explanation']}\n"
                for ex in self.few_shot_examples
            ]
        )

        prompt = f"""You are a music metadata normalizer.

RULES:
1. Keep ALL artist/band names, even if they seem unusual
2. Remove publishers, companies, and <Unknown> tags
3. Convert separators (,&;) to forward slash (/) with NO extra spaces
4. Fix "LastName, FirstName" to "FirstName LastName"
5. Extract names from parentheses when appropriate
6. If input has valid names, NEVER return empty result

EXAMPLES:
{examples_text}

Now normalize this:
Input: {text}
Output:"""

        return prompt

    def normalize_with_fallback(self, text: str) -> str:
        """Use LLM as intelligent fallback when tools fail.

        Args:
            text: Input text to normalize using LLM

        Returns:
            Normalized text from LLM, or original text if LLM fails
        """
        try:
            prompt = self.build_fallback_prompt(text)

            if hasattr(self.llm, "predict"):
                # For OpenAI LLM
                result = self.llm.predict(prompt)
            else:
                # For ChatOpenAI
                result = self.llm.predict(prompt)

            # Clean and validate result
            result = result.strip()

            # If LLM returns empty but input wasn't empty, return original
            if not result and text.strip():
                return text.strip()

            return result

        except Exception as e:
            print(f"Fallback LLM error: {e}")
            # Last resort: return input cleaned
            return text.strip() if text.strip() else ""


class ImprovedTextNormalizationAgent:
    """Improved agent with intelligent decision making and fallback strategies.

    This agent combines ReACT reasoning with specialized tools to normalize
    text metadata, particularly focused on music writer/composer names.
    """

    def __init__(self, llm_provider: str = "openai") -> None:
        """Initialize the improved agent.

        Args:
            llm_provider: LLM provider to use (currently only 'openai')

        Raises:
            ValueError: If unsupported provider or missing API key
        """

        # Initialize LLM
        if llm_provider.lower() == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=openai_api_key,
                request_timeout=30,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Initialize tools
        self.tools_class = TextNormalizationTools()

        # Initialize fallback LLM
        self.fallback_llm = SmartFallbackLLM(self.llm)

        # Define available tools
        self.tools = self._create_tools()

        # Create agent with improved prompt
        self.agent = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """Create the available tools for text normalization.

        Returns:
            List of Tool objects with name, description, and function
        """
        return [
            Tool(
                name="analyze_structure",
                description=(
                    "Analyze text structure to understand what "
                    "processing is needed. Use this first to make "
                    "informed decisions."
                ),
                func=self.tools_class.analyze_text_structure,
            ),
            Tool(
                name="remove_angle_brackets",
                description=(
                    "Remove angle bracket content like <Unknown>. "
                    "Use early for cleaning."
                ),
                func=self.tools_class.remove_angle_brackets,
            ),
            Tool(
                name="remove_numeric_ids",
                description=(
                    "Remove numeric IDs like (999990) or standalone "
                    "numbers. Use early for cleaning."
                ),
                func=self.tools_class.remove_numeric_ids,
            ),
            Tool(
                name="extract_from_parentheses",
                description=(
                    "Extract names from parentheses intelligently. "
                    "Use when parentheses are present."
                ),
                func=self.tools_class.extract_from_parentheses,
            ),
            Tool(
                name="normalize_separators",
                description=(
                    "Convert all separators (,&;) to forward slash. "
                    "Use early to normalize format."
                ),
                func=self.tools_class.normalize_separators,
            ),
            Tool(
                name="remove_publishers",
                description=(
                    "Remove publisher and company names using "
                    "comprehensive list. Use if analysis shows "
                    "publishers are present."
                ),
                func=self.tools_class.remove_publishers,
            ),
            Tool(
                name="fix_inversions",
                description="Fix inverted names (LastName, FirstName). "
                "Use if analysis shows inversions are present.",
                func=self.tools_class.fix_name_inversions,
            ),
            Tool(
                name="remove_prefixes",
                description="Remove prefixes like MR, MS, DR. "
                "Use if analysis shows prefixes are present.",
                func=self.tools_class.remove_prefixes,
            ),
            Tool(
                name="remove_special_patterns",
                description="Remove special patterns like <Unknown>, traditional, PD. "
                "Use for cleaning unwanted terms.",
                func=self.tools_class.remove_special_patterns,
            ),
            Tool(
                name="split_compound_names",
                description="Split ALL CAPS compound names like 'AHN TAI' "
                "into separate names. Use after other processing.",
                func=self.tools_class.split_compound_names,
            ),
            Tool(
                name="smart_fallback",
                description="Use when other tools fail or produce empty "
                "results. Handles edge cases with AI.",
                func=self.fallback_llm.normalize_with_fallback,
            ),
        ]

    def _create_agent(self) -> AgentExecutor:
        """Create the improved agent with better prompt.

        Returns:
            Configured AgentExecutor with ReACT agent and tools
        """

        # Improved system prompt
        agent_prompt = """You are an intelligent text normalization agent.

WORKFLOW:
1. ALWAYS start with analyze_structure to understand what needs to be done
2. IF analysis shows 'has_publishers': True, IMMEDIATELY call remove_publishers
3. IF analysis shows 'has_inversions': True, call fix_inversions
4. IF analysis shows 'has_prefixes': True, call remove_prefixes
5. IF analysis shows 'has_special_chars': True, call remove_special_patterns
6. IF analysis shows 'has_separators_to_normalize': True, call normalize_separators
7. IF analysis shows 'has_compound_names': True, call split_compound_names

CRITICAL RULES:
- Use tools with EXACT input text - do NOT modify separators in tool calls
- NEVER change / to , when calling tools - preserve original format
- After processing with tools, provide Final Answer directly
- NEVER return empty results if input contains valid names
- If normal tools fail, use smart_fallback as last resort

PUBLISHER DETECTION:
- If analyze_structure shows 'has_publishers': True, you MUST call remove_publishers
- Publishers include: MESAM, ASCAP, BMI, SONY, WARNER, etc.
- Example: "MESAM/Servet Tunç" → analyze shows has_publishers=True → call remove_publishers → "Servet Tunç"

CRITICAL RESPONSE FORMAT:
- ALWAYS return ONLY the cleaned names separated by forward slashes
- NO descriptive text, NO quotes around the entire result
- Example CORRECT Final Answer: John Smith/Jane Doe
- Example WRONG Final Answer: "The names are valid: John Smith, Jane Doe"

Follow the workflow systematically!"""

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=6,  # Reduced iterations for simpler workflow
            early_stopping_method="generate",
            handle_parsing_errors=True,
            agent_kwargs={"prefix": agent_prompt},
        )

    def process(self, text: str) -> NormalizationResult:
        """Process text with improved logic and fallback strategies.

        Args:
            text: Input text to normalize

        Returns:
            NormalizationResult with output, confidence, timing, and metadata
        """

        if not text or not text.strip():
            return NormalizationResult(
                original_text=text,
                output_text="",
                confidence=1.0,
                processing_time=0.0,
                tools_used=[],
                tokens_used=0,
                strategy_used="direct",
            )

        start_time = time.time()
        tools_used = []
        strategy_used = "agent"

        try:
            # Try agent first
            result = self.agent.run(text)

            # Check if result is problematic
            if not result or result.strip() == "" or result.strip() == "/":
                print(
                    f"⚠️  Agent returned empty/invalid result for '{text}', using fallback..."
                )
                result = self.fallback_llm.normalize_with_fallback(text)
                strategy_used = "fallback"

            # Clean result
            result = result.strip()

            # Final safety check
            if not result and text.strip():
                result = text.strip()
                strategy_used = "direct"

            processing_time = time.time() - start_time

            # Calculate confidence based on strategy
            if strategy_used == "agent":
                confidence = 0.95
            elif strategy_used == "fallback":
                confidence = 0.85
            else:
                confidence = 0.5

            return NormalizationResult(
                original_text=text,
                output_text=result,
                confidence=confidence,
                processing_time=processing_time,
                tools_used=tools_used,
                tokens_used=50,  # Estimated
                strategy_used=strategy_used,
            )

        except Exception as e:
            print(f"❌ Agent error: {e}")

            # Fallback to LLM
            try:
                result = self.fallback_llm.normalize_with_fallback(text)
                processing_time = time.time() - start_time

                return NormalizationResult(
                    original_text=text,
                    output_text=result,
                    confidence=0.75,
                    processing_time=processing_time,
                    tools_used=["smart_fallback"],
                    tokens_used=30,
                    strategy_used="fallback",
                )

            except Exception as e2:
                print(f"❌ Fallback error: {e2}")
                processing_time = time.time() - start_time

                return NormalizationResult(
                    original_text=text,
                    output_text=text.strip(),
                    confidence=0.1,
                    processing_time=processing_time,
                    tools_used=[],
                    tokens_used=0,
                    strategy_used="direct",
                )


# Test the improved agent
if __name__ == "__main__":
    print("🚀 Testing Improved Agent")
    print("=" * 50)

    # Test cases that previously failed
    test_cases = [
        "Alkaline",
        "Helltrain",
        "ABSTRAXION",
        "Thunder Bklu",
        "CJ Arey, Anthony Conti, Kasey Karlsen & Kyle O'Braitis",
        "Ahmad Eerfan",
        "CA JACOBSEN BIBBI/CA JACOBSEN STEFFEN",
    ]

    agent = ImprovedTextNormalizationAgent()

    for test in test_cases:
        print(f"\n🧪 Testing: {test}")
        result = agent.process(test)
        print(f"✅ Result: '{result.output_text}'")
        print(
            f"📊 Strategy: {result.strategy_used}, Confidence: {result.confidence:.2f}"
        )
        print(f"⏱️  Time: {result.processing_time:.2f}s")
