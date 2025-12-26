#!/usr/bin/env python3
"""
Test suite generator for CodeTransformBench.
Generates property-based test suites using LLMs via OpenRouter.

Target: 450-500 test suites with 80%+ branch coverage.
Budget: €10-15
"""

import asyncio
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config, load_prompt_template
from src.api.openrouter_client import OpenRouterClient
from src.utils.db_utils import get_db_session, get_all_functions
from src.utils.logger import logger
from src.utils.progress import create_progress_bar
from database.models import Function


class TestGenerator:
    """
    Generates test suites for functions using LLMs.
    """

    # Model to use for test generation (best balance of quality/cost)
    TEST_GEN_MODEL = 'anthropic/claude-3.5-sonnet'

    # File extensions by language
    FILE_EXTENSIONS = {
        'python': '.py',
        'java': '.java',
        'javascript': '.js',
        'cpp': '.cpp'
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize test generator.

        Args:
            output_dir: Directory to save test files (default: data/test_suites)
        """
        self.output_dir = output_dir or config.DATA_TEST_SUITES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenRouterClient()

        self.stats = {
            'total': 0,
            'generated': 0,
            'failed': 0,
            'total_cost': 0.0
        }

        logger.info(f"Initialized test generator, output: {self.output_dir}")

    def load_template(self, language: str) -> str:
        """Load prompt template for a language."""
        template_name = f'test_gen_{language}'
        try:
            return load_prompt_template(template_name)
        except FileNotFoundError:
            logger.error(f"Template not found: {template_name}")
            raise

    def render_prompt(self, template: str, code: str) -> str:
        """Render prompt template with code."""
        return template.replace('{code}', code)

    def extract_code_from_response(self, response: str, language: str) -> Optional[str]:
        """
        Extract code from LLM response.

        Handles markdown code blocks and attempts to find code even in messy responses.

        Args:
            response: LLM response text
            language: Programming language

        Returns:
            Extracted code or None if failed
        """
        # Try to extract from markdown code blocks first
        patterns = [
            r'```(?:' + language + r')?\s*\n(.*?)```',  # ```python ... ```
            r'```\s*\n(.*?)```',  # ``` ... ```
            r'```(?:' + language + r')?\s+(.*?)```',  # ```python ... ``` (with space)
            r'```\s+(.*?)```',  # ``` ... ``` (with space)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest match (likely the main code)
                code = max(matches, key=len).strip()
                if code:
                    return code

        # Fallback: try to find code-like content based on language
        if language == 'python':
            # Look for Python code patterns
            if any(keyword in response for keyword in ['def test_', 'import pytest', 'from hypothesis', '@given', '@seed']):
                # Try to extract just the code part, removing explanatory text
                lines = response.split('\n')
                code_lines = []
                in_code = False
                for line in lines:
                    if any(kw in line for kw in ['import ', 'from ', 'def ', '@']):
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                if code_lines:
                    return '\n'.join(code_lines).strip()

        elif language == 'javascript':
            # Look for JavaScript test patterns
            if any(keyword in response for keyword in ['describe(', 'test(', 'it(', 'fc.assert', 'fc.property']):
                # Extract code part
                lines = response.split('\n')
                code_lines = []
                in_code = False
                for line in lines:
                    if any(kw in line for kw in ['const ', 'require(', 'import ', 'describe(', 'test(', 'it(', 'fc.']):
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                if code_lines:
                    return '\n'.join(code_lines).strip()

        elif language == 'java':
            # Look for Java test patterns
            if any(keyword in response for keyword in ['import org.junit', '@Test', 'class ', 'public ']):
                # Extract code part
                lines = response.split('\n')
                code_lines = []
                in_code = False
                for line in lines:
                    if any(kw in line for kw in ['import ', 'package ', 'class ', '@Test', 'public ']):
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                if code_lines:
                    return '\n'.join(code_lines).strip()

        elif language == 'cpp':
            # Look for C++ test patterns
            if any(keyword in response for keyword in ['#include', 'TEST(', 'EXPECT_', 'ASSERT_', 'RC_GTEST_PROP']):
                # Extract code part
                lines = response.split('\n')
                code_lines = []
                in_code = False
                for line in lines:
                    if any(kw in line for kw in ['#include', 'using ', 'TEST(', 'namespace ', 'int main']):
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                if code_lines:
                    return '\n'.join(code_lines).strip()

        # Last resort: if response looks like code (has keywords), return it
        code_indicators = {
            'python': ['def ', 'import ', 'class ', '@'],
            'javascript': ['function ', 'const ', 'describe(', 'test('],
            'java': ['public ', 'class ', 'import ', '@Test'],
            'cpp': ['#include', 'void ', 'int ', 'TEST(']
        }

        if language in code_indicators:
            if any(indicator in response for indicator in code_indicators[language]):
                # Remove common explanatory prefixes/suffixes
                cleaned = response.strip()
                # Remove "Here is" or "Here's" at the start
                cleaned = re.sub(r'^(Here is|Here\'s|This is).*?:\s*', '', cleaned, flags=re.IGNORECASE)
                if cleaned:
                    return cleaned

        logger.warning(f"Could not extract code from response for {language}")
        logger.debug(f"Response preview: {response[:200]}...")
        return None

    async def generate_test_suite(
        self,
        function: Dict[str, Any],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Generate test suite for a function.

        Args:
            function: Function dict with id, language, code
            max_retries: Maximum retry attempts

        Returns:
            Dict with test code and metadata, or None if failed
        """
        language = function['language']
        template = self.load_template(language)
        prompt = self.render_prompt(template, function['code'])

        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Generating tests for {function['id']} (attempt {attempt + 1})")

                # Generate test code
                result = await self.client.generate(
                    model=self.TEST_GEN_MODEL,
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=2000,
                    retry_on_error=True
                )

                # Extract code from response
                test_code = self.extract_code_from_response(result['content'], language)

                if not test_code:
                    last_error = "Could not extract code from response"
                    logger.warning(f"{function['id']}: {last_error}, retrying...")
                    continue

                # Update statistics
                self.stats['generated'] += 1
                self.stats['total_cost'] += result['cost_usd']

                logger.success(f"Generated tests for {function['id']} (cost: ${result['cost_usd']:.4f})")

                return {
                    'function_id': function['id'],
                    'language': language,
                    'test_code': test_code,
                    'cost_usd': result['cost_usd'],
                    'tokens_input': result['tokens_input'],
                    'tokens_output': result['tokens_output'],
                    'latency_ms': result['latency_ms']
                }

            except Exception as e:
                last_error = str(e)
                logger.error(f"{function['id']} attempt {attempt + 1} failed: {e}")

                # If it's a credit error, stop immediately
                if '402' in last_error or 'credit' in last_error.lower():
                    logger.error("Insufficient credits, stopping...")
                    raise

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        logger.error(f"Failed to generate tests for {function['id']} after {max_retries} attempts: {last_error}")
        self.stats['failed'] += 1
        return None

    def save_test_file(self, test_data: Dict[str, Any]) -> Path:
        """
        Save test code to file.

        Args:
            test_data: Test data dict with function_id, language, test_code

        Returns:
            Path to saved test file
        """
        function_id = test_data['function_id']
        language = test_data['language']
        test_code = test_data['test_code']

        ext = self.FILE_EXTENSIONS[language]
        filename = f"{function_id}_test{ext}"
        filepath = self.output_dir / filename

        filepath.write_text(test_code, encoding='utf-8')

        return filepath

    async def generate_all_tests(
        self,
        functions: Optional[List[Function]] = None,
        max_concurrent: int = 5,
        budget_limit_usd: float = 20.0
    ) -> Dict[str, Any]:
        """
        Generate test suites for all functions.

        Args:
            functions: List of functions (defaults to all validated functions in DB)
            max_concurrent: Maximum concurrent API requests
            budget_limit_usd: Stop if total cost exceeds this

        Returns:
            Statistics dict
        """
        if functions is None:
            with get_db_session() as session:
                db_functions = session.query(Function).all()
                # Convert to dicts to avoid SQLAlchemy detached instance errors
                functions = []
                for func in db_functions:
                    functions.append({
                        'id': func.id,
                        'language': func.language,
                        'code': func.code,
                        'cyclomatic_complexity': func.cyclomatic_complexity
                    })

        self.stats['total'] = len(functions)

        logger.info(f"Generating tests for {len(functions)} functions...")
        logger.info(f"Budget limit: ${budget_limit_usd:.2f}")

        # Process functions in batches
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_with_limit(func: Dict[str, Any]):
            # Check budget
            if self.stats['total_cost'] >= budget_limit_usd:
                logger.warning(f"Budget limit reached (${self.stats['total_cost']:.2f}), skipping remaining functions")
                return None

            async with semaphore:
                test_data = await self.generate_test_suite(func)

                if test_data:
                    # Save test file
                    filepath = self.save_test_file(test_data)

                    # Update database: set test_suite_path
                    with get_db_session() as session:
                        db_func = session.query(Function).filter(Function.id == func['id']).first()
                        if db_func:
                            db_func.test_suite_path = str(filepath)
                            session.commit()

                return test_data

        # Create progress bar
        with create_progress_bar("Generating tests", len(functions)) as progress:
            task = progress.add_task("Generating", total=len(functions))

            # Process all functions
            tasks = [_generate_with_limit(func) for func in functions]

            for coro in asyncio.as_completed(tasks):
                await coro
                progress.update(task, advance=1)

                # Print progress every 10 functions
                if self.stats['generated'] % 10 == 0:
                    logger.info(
                        f"Progress: {self.stats['generated']}/{self.stats['total']} "
                        f"(failed: {self.stats['failed']}, cost: ${self.stats['total_cost']:.2f})"
                    )

        # Final statistics
        logger.success(f"\n✓ Test generation complete!")
        logger.info(f"Generated: {self.stats['generated']}/{self.stats['total']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total cost: ${self.stats['total_cost']:.2f}")
        logger.info(f"Avg cost per test: ${self.stats['total_cost'] / max(1, self.stats['generated']):.4f}")

        return self.stats


async def main():
    """Main entry point for test generator."""
    logger.info("Starting test suite generation...")

    generator = TestGenerator()

    # Get functions from database and convert to dicts
    with get_db_session() as session:
        db_functions = session.query(Function).all()

        if not db_functions:
            logger.error("No functions found in database. Run data collection first.")
            return

        # Convert to dicts to avoid SQLAlchemy detached instance errors
        functions = []
        for func in db_functions:
            functions.append({
                'id': func.id,
                'language': func.language,
                'code': func.code,
                'cyclomatic_complexity': func.cyclomatic_complexity
            })

    logger.info(f"Found {len(functions)} functions in database")

    # Generate tests
    stats = await generator.generate_all_tests(
        functions=functions,
        max_concurrent=5,
        budget_limit_usd=15.0  # Stop if cost exceeds €15
    )

    # Print final statistics
    logger.info("\n" + "=" * 60)
    logger.info("Test Generation Statistics")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
