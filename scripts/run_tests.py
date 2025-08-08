#!/usr/bin/env python3
"""
Test runner script for the text normalization project.

This script runs all tests and generates coverage reports.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    print("-" * 60)

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=False, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    project_root = Path(__file__).parent.parent  # Go up one level from scripts/
    print(f"📁 Working directory: {project_root}")

    # Change to project directory
    import os

    os.chdir(project_root)

    success = True

    # Run linting
    success &= run_command("ruff check src/ tests/", "Code linting with ruff")

    # Run formatting check
    success &= run_command("ruff format --check src/ tests/", "Code formatting check")

    # Run tests with coverage
    success &= run_command(
        "python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term",
        "Running tests with coverage",
    )

    # Type checking (if mypy is available)
    try:
        import mypy

        success &= run_command(
            "mypy src/ --ignore-missing-imports", "Type checking with mypy"
        )
    except ImportError:
        print("⚠️  MyPy not available, skipping type checking")

    if success:
        print("\n🎉 All checks passed!")
        print("\n📊 Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print("\n💥 Some checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
