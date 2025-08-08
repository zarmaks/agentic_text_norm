# Agentic Text Normalization

> Advanced AI-powered text normalization comparing rule-based, hybrid, and agentic LLM approaches.

[![CI](https://github.com/zarmaks/agentic_text_norm/actions/workflows/ci.yml/badge.svg)](https://github.com/zarmaks/agentic_text_norm/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT](https://img.shields.io/badge/powered%20by-OpenAI%20GPT-green.svg)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-orange.svg)](https://langchain.com/)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-blue.svg)](https://spacy.io/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Research](https://img.shields.io/badge/type-research-purple.svg)](docs/methodology.md)

## 📊 Research Findings & Reports

### Key Insights (100-Sample Analysis)

1. **Agentic Agent Performance**: 92.0% accuracy with highest confidence (0.949)
2. **Hybrid Agent Efficiency**: 88.0% accuracy with selective LLM usage (5% of cases)
3. **Rule-Based Speed**: 86.0% accuracy with instant processing and zero cost
4. **Cost-Accuracy Trade-offs**: Pure Rules (free), Hybrid ($0.05/100), Agentic ($1.00/100)
5. **Error Patterns**: Under-segmentation (7%), Over-segmentation (7%), Normalization errors (1-3%)

### 🔬 Auto-Generated Research Reports

**Create your own reports**: `python comprehensive_research.py --samples 50`

- **[📊 100-Sample Comprehensive Analysis](reports/research_report_100samples.md)**: Complete performance metrics and error analysis
- **[📋 Research Methodology](docs/methodology.md)**: Technical approach and validation framework
- **Auto-Generated**: `comprehensive_research_report_*.md` files created on-demand

## 🎯 Project Overview

This project implements and compares **three distinct approaches** to text normalization for music industry data, specifically targeting composer and writer name cleaning. The system processes raw, messy text containing publisher names, copyright notices, and formatting issues into clean, standardized composer/writer credits.

### 🔍 Three Approaches Explained

#### 1. **Pure Rule-Based Approach** (`RuleBasedAgent`)

- **Method**: Deterministic pattern matching with predefined rules
- **Technology**: Python regex patterns + comprehensive publisher lists
- **Strengths**: Fast, zero cost, deterministic
- **Location**: `src/text_normalization/rule_based_agent.py`

```python
from src.text_normalization.rule_based_agent import RuleBasedAgent
agent = RuleBasedAgent()
result = agent.normalize_text("Sony Music/John Doe/Copyright Control")
# Output: "John Doe"
```

#### 2. **Hybrid Agent Approach** (`HybridAgent`)

- **Method**: Rules + NER validation + selective LLM intervention
- **Technology**: Rule patterns + spaCy NER + GPT for complex cases
- **Strengths**: Balanced accuracy vs cost, smart LLM usage (5% rate)
- **Location**: Research comparison implementation

```python
# Hybrid uses rules first, then LLM only for complex cases
hybrid_agent = HybridAgent()
result = hybrid_agent.process("Day & Murray (Bowles, Gaudet)")
# Uses LLM for complex parentheses parsing
```

#### 3. **Agentic Agent Approach** (`ImprovedTextNormalizationAgent`)

- **Method**: Full LLM-powered agent with specialized tools
- **Technology**: GPT + LangChain + 12 specialized normalization tools
- **Strengths**: Highest accuracy, handles complex edge cases
- **Location**: `src/text_normalization/agentic_agent.py`

```python
from src.text_normalization.agentic_agent import ImprovedTextNormalizationAgent
agent = ImprovedTextNormalizationAgent()
result = agent.process("Day & Murray (Bowles, Gaudet)")
# Output: NormalizationResult with processed text
```

### 📊 Performance Comparison (100 Samples - Latest Results)

| Approach                | Accuracy | Avg Confidence | Speed  | Cost/100 | LLM Usage | Best For                    |
| ----------------------- | -------- | -------------- | ------ | -------- | --------- | --------------------------- |
| **Pure Rules**    | 86.0%    | 0.608          | 0.000s | $0.00    | 0%        | High-volume, cost-sensitive |
| **Hybrid Agent**  | 88.0%    | 0.796          | 0.049s | $0.05    | 5%        | Production balance          |
| **Agentic Agent** | 92.0%    | 0.949          | 3.345s | $1.00    | 100%      | Maximum quality             |

> 📄 **[View Research Methodology](docs/methodology.md)** for detailed technical approach and validation framework.

> 📊 **[View Full Research Report](reports/research_report_100samples.md)** for comprehensive analysis, error categorization, and detailed metrics.

## 🚀 Key Features

### Advanced Processing Capabilities

- **Publisher Removal**: 80+ major publishers (Sony, Universal, Warner, etc.)
- **Name Normalization**: "LastName, FirstName" → "FirstName LastName"
- **Separator Standardization**: "&", ";" → "/" for consistent formatting
- **Special Character Handling**: Remove `<Unknown>`, brackets, copyright symbols
- **Compound Name Splitting**: "JOHN SMITH" → "JOHN/SMITH"
- **Parentheses Intelligence**: Extract or remove based on context

### 🔬 Automated Research Pipeline

- **Comprehensive Analysis**: `comprehensive_research.py` automatically evaluates all approaches
- **Auto-Generated Reports**: Detailed markdown reports with metrics, error analysis, and recommendations
- **Flexible Sample Sizes**: Test with 5, 50, 100+ samples as needed
- **Cost Analysis**: Real-time token usage and cost estimation
- **Error Categorization**: Systematic classification of failure modes
- **Performance Metrics**: Accuracy, confidence, speed, and efficiency comparisons

### Agentic Workflow Architecture

The agentic approach implements a sophisticated **ReAct (Reasoning + Acting) pattern**:

```
Input Text → Analyze Structure → Plan Strategy → Execute Tools → Validate → Output
     ↓              ↓               ↓             ↓           ↓        ↓
"Sony/John"    Publishers?     Remove Sony    Tool: remove   Check    "John"
               Yes!            Keep John      publishers     result    ✓
```

**Tool Ecosystem**: 12 specialized tools including:

- `analyze_structure`: Detect patterns and complexity
- `remove_publishers`: Eliminate music industry entities
- `fix_name_inversions`: Handle "LastName, FirstName" format
- `normalize_separators`: Standardize delimiters
- `extract_from_parentheses`: Smart parentheses handling
- `split_compound_names`: Break apart combined names

## 🛠️ Technical Architecture

### Technology Stack

- **Core**: Python 3.11+, OpenAI API
- **LLM Framework**: LangChain with ReAct agents
- **NLP**: spaCy for Named Entity Recognition
- **Testing**: pytest with 39 comprehensive tests
- **Code Quality**: Ruff for formatting and linting
- **Data Analysis**: pandas, numpy for research metrics

### Project Structure

```
agentic_text_norm/
├── 📁 src/text_normalization/       # Core implementation
│   ├── agentic_agent.py            # Main LLM-powered agent
│   ├── rule_based_agent.py         # Pure rule-based approach
│   ├── tools.py                    # 12 specialized tools
│   └── cleaner.py                  # Text cleaning utilities
├── 📁 scripts/                     # Easy-to-use demo scripts
│   ├── run_demo.py                 # Interactive demonstration
│   ├── quick_start.py              # Command-line interface
│   └── run_tests.py                # Test runner
├── 📁 tests/                       # Comprehensive test suite (39 tests)
├── 📁 data/                        # 10K music credits dataset
├── 📁 docs/                        # Documentation
│   └── methodology.md              # Research methodology
├── 📁 reports/                     # 📊 Research reports & analysis
│   ├── research_report_100samples.md  # 100-sample analysis
│   ├── research_report_50samples.md   # 50-sample analysis
├── 📁 optional/                    # Legacy/alternative implementations
├── comprehensive_research.py       # 🔬 Research automation tool
├── main.py                         # CLI entry point
└── requirements.txt                # Dependencies
```

## Research Findings & Reports

### Key Insights (100-Sample Analysis)

1. **Agentic Agent Performance**: 92.0% accuracy with highest confidence (0.949)
2. **Hybrid Agent Efficiency**: 88.0% accuracy with selective LLM usage (5% of cases)
3. **Rule-Based Speed**: 86.0% accuracy with instant processing and zero cost
4. **Cost-Accuracy Trade-offs**: Pure Rules (free), Hybrid ($0.05/100), Agentic ($1.00/100)
5. **Error Patterns**: Under-segmentation (7%), Over-segmentation (7%), Normalization errors (1-3%)

### 🔬 Auto-Generated Research Reports

**Create your own reports**: `python comprehensive_research.py --samples 50`

- **[📊 100-Sample Research Report](reports/research_report_100samples.md)**: Comprehensive performance analysis with detailed metrics
- **[All Research Reports](reports/)**: Complete collection of analysis reports

### Performance Breakdown

- **Pure Rules**: Fast (0.000s), free operation, deterministic results
- **Hybrid Agent**: Balanced approach with 5% LLM usage for complex cases
- **Agentic Agent**: Full LLM reasoning with 100% tool-calling capability

## 🎯 Use Cases & Applications

### Industry Applications

1. **Music Streaming Platforms**: Clean metadata for search and recommendations
2. **Rights Management Systems**: Accurate composer identification for royalties
3. **Data Migration Projects**: Standardize legacy music database formats
4. **Content Management**: Real-time text cleaning for music industry APIs

### Research Applications

1. **NLP Benchmarking**: Compare rule-based vs LLM approaches
2. **Cost-Accuracy Analysis**: Study trade-offs in production deployments
3. **Agent Architecture**: Research tool-calling patterns and reasoning workflows
4. **Domain Adaptation**: Test normalization on specialized text types

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```powershell
# Clone repository
git clone https://github.com/zarmaks/agentic_text_norm.git
cd agentic_text_norm

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 🔬 Research Tool (Recommended First Step!)

The `comprehensive_research.py` tool automatically generates detailed performance reports comparing all three approaches:

```bash
# Generate comprehensive research report (auto-saves to reports/)
python comprehensive_research.py --samples 10   # Quick analysis
python comprehensive_research.py --samples 50   # Detailed analysis
python comprehensive_research.py --samples 100  # Full research

# Example output: reports/comprehensive_research_report_20250808_123456.md
# Contains: accuracy metrics, error analysis, cost breakdown, timing data
```

### Easy Demo Scripts (Alternative Options)

```bash
# Interactive demo with all approaches
python scripts/run_demo.py

# Quick single text processing
python scripts/quick_start.py "Sony Music/John Doe"

# Compare approaches on specific text
python scripts/quick_start.py --approach agentic "Complex Text Input"

# Run all tests
python scripts/run_tests.py
```

### Command Line Interface

```bash
# Process single text
python main.py --text "Sony Music/John Doe" --approach rules
python main.py --text "Complex case" --approach agentic

# Batch processing
python main.py --batch data/input.csv --output results.csv --approach agentic
```

### Programmatic Usage

```python
# Rule-based approach (fastest)
from src.text_normalization.rule_based_agent import RuleBasedAgent
rules_agent = RuleBasedAgent()
result = rules_agent.normalize_text("Sony Music/John Doe")

# Agentic approach (highest quality)
from src.text_normalization.agentic_agent import ImprovedTextNormalizationAgent
agentic_agent = ImprovedTextNormalizationAgent()
result = agentic_agent.process("Complex Text/Publisher Name")
print(result.output_text)
```

## 🧪 Testing & Validation

### Comprehensive Test Suite

```bash
# Run all 39 tests
pytest

# Run with coverage report  
pytest --cov=src

# Run specific test categories
pytest tests/test_rule_based_agent.py     # Rule-based tests
pytest tests/test_integration.py          # Integration tests
pytest tests/test_cleaner.py             # Text cleaner tests

# Quick test runner script
python scripts/run_tests.py
```

### Research & Methodology

- **[📋 Research Methodology](docs/methodology.md)**: Comprehensive documentation of approach, validation framework, and evaluation criteria
- **Validation Dataset**: 10K real-world music credits from industry sources
- **Automated Research Pipeline**: Generate performance reports with any sample size
- **Error Analysis**: Systematic categorization of failure modes and edge cases

## 🚀 Development & Contributing

### Development Setup

```bash
# Install all dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Add your OPENAI_API_KEY to .env

# Run tests to verify setup
pytest

# Generate a research report
python comprehensive_research.py --samples 10
```

### Code Quality Tools

- **Ruff**: Formatting and linting (configured in `pyproject.toml`)
- **pytest**: Testing framework with coverage reporting
- **Type Hints**: Improved for better IDE support
- **Pre-commit**: Automated code quality checks

### Project Status

- ✅ **Core Implementation**: Complete and tested
- ✅ **Publisher Detection**: Bug fixed, 100% accuracy achieved
- ✅ **Test Coverage**: 39 comprehensive tests passing
- ✅ **Documentation**: Complete methodology and usage guides
- ✅ **Research Pipeline**: Automated evaluation and reporting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **[Research Methodology](docs/methodology.md)**: Detailed technical approach
- **[100-Sample Research Report](reports/research_report_100samples.md)**: Comprehensive performance analysis with detailed metrics
- **[All Research Reports](reports/)**: Complete collection of analysis reports
- **[Test Coverage](htmlcov/index.html)**: HTML coverage report (after running tests)

---

*Built for AI/ML Research Engineering position - demonstrating advanced NLP, LLM integration, and systematic research methodology.*
