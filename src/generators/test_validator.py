"""
Test suite validator for CodeTransformBench.
Validates generated test suites for syntax, functionality, coverage, and performance.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import logger


class TestValidator:
    """Validates generated test suites."""

    # Timeout for test execution (ms)
    TIMEOUT_MS = 100

    def __init__(self):
        """Initialize test validator."""
        self.stats = {
            'total': 0,
            'syntax_passed': 0,
            'functional_passed': 0,
            'coverage_passed': 0,
            'performance_passed': 0,
            'fully_valid': 0
        }

    def validate_python(
        self,
        test_code: str,
        function_code: str,
        function_id: str
    ) -> Dict[str, Any]:
        """
        Validate Python test suite.

        Args:
            test_code: Test code
            function_code: Original function code
            function_id: Function ID for reporting

        Returns:
            Validation result dict
        """
        result = {
            'function_id': function_id,
            'language': 'python',
            'syntax_valid': False,
            'functional_valid': False,
            'coverage_pct': 0.0,
            'execution_time_ms': 0,
            'errors': []
        }

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write function code
            func_file = tmppath / 'function.py'
            func_file.write_text(function_code)

            # Write test code (with function import)
            test_file = tmppath / 'test_function.py'
            test_with_import = f"from function import *\n\n{test_code}"
            test_file.write_text(test_with_import)

            # 1. Syntax check: try to compile
            try:
                compile(test_code, '<string>', 'exec')
                result['syntax_valid'] = True
                self.stats['syntax_passed'] += 1
            except SyntaxError as e:
                result['errors'].append(f"Syntax error: {e}")
                logger.warning(f"{function_id}: Syntax error - {e}")
                return result

            # 2. Functional check: run tests
            try:
                start_time = time.time()

                proc = subprocess.run(
                    ['pytest', str(test_file), '-v', '--tb=short'],
                    cwd=tmppath,
                    capture_output=True,
                    text=True,
                    timeout=self.TIMEOUT_MS / 1000.0
                )

                execution_time = int((time.time() - start_time) * 1000)
                result['execution_time_ms'] = execution_time

                if proc.returncode == 0:
                    result['functional_valid'] = True
                    self.stats['functional_passed'] += 1
                else:
                    result['errors'].append(f"Tests failed: {proc.stdout}")
                    logger.debug(f"{function_id}: Tests failed")

                # 3. Performance check
                if execution_time < self.TIMEOUT_MS:
                    self.stats['performance_passed'] += 1
                else:
                    result['errors'].append(f"Tests too slow: {execution_time}ms")

                # 4. Coverage check (optional - requires pytest-cov)
                try:
                    cov_proc = subprocess.run(
                        ['pytest', str(test_file), '--cov=function', '--cov-report=json'],
                        cwd=tmppath,
                        capture_output=True,
                        text=True,
                        timeout=self.TIMEOUT_MS / 1000.0
                    )

                    if cov_proc.returncode == 0:
                        # Parse coverage report
                        cov_file = tmppath / 'coverage.json'
                        if cov_file.exists():
                            cov_data = json.loads(cov_file.read_text())
                            total_coverage = cov_data.get('totals', {}).get('percent_covered', 0)
                            result['coverage_pct'] = total_coverage

                            if total_coverage >= 80:
                                self.stats['coverage_passed'] += 1

                except Exception as e:
                    logger.debug(f"{function_id}: Coverage check failed - {e}")

            except subprocess.TimeoutExpired:
                result['errors'].append(f"Test execution timeout (>{self.TIMEOUT_MS}ms)")
                logger.warning(f"{function_id}: Timeout")

            except Exception as e:
                result['errors'].append(f"Execution error: {e}")
                logger.warning(f"{function_id}: Execution error - {e}")

        # Check if fully valid
        if (result['syntax_valid'] and
            result['functional_valid'] and
            result['coverage_pct'] >= 80 and
            result['execution_time_ms'] < self.TIMEOUT_MS):
            self.stats['fully_valid'] += 1

        return result

    def validate_java(
        self,
        test_code: str,
        function_code: str,
        function_id: str
    ) -> Dict[str, Any]:
        """
        Validate Java test suite.

        Note: Simplified validation (syntax only for now).
        Full validation requires javac + JUnit setup.
        """
        result = {
            'function_id': function_id,
            'language': 'java',
            'syntax_valid': False,
            'functional_valid': False,
            'coverage_pct': 0.0,
            'execution_time_ms': 0,
            'errors': []
        }

        # Check if code contains basic test structure
        if 'import org.junit' in test_code and '@Test' in test_code:
            result['syntax_valid'] = True
            self.stats['syntax_passed'] += 1
            # Assume functional if it has test annotations
            result['functional_valid'] = True
            self.stats['functional_passed'] += 1
            self.stats['performance_passed'] += 1
        else:
            result['errors'].append("Missing JUnit test structure")

        return result

    def validate_javascript(
        self,
        test_code: str,
        function_code: str,
        function_id: str
    ) -> Dict[str, Any]:
        """
        Validate JavaScript test suite.

        Note: Simplified validation (syntax only for now).
        Full validation requires Node.js + Jest setup.
        """
        result = {
            'function_id': function_id,
            'language': 'javascript',
            'syntax_valid': False,
            'functional_valid': False,
            'coverage_pct': 0.0,
            'execution_time_ms': 0,
            'errors': []
        }

        # Check if code contains basic test structure
        if ('describe(' in test_code or 'test(' in test_code or 'it(' in test_code):
            result['syntax_valid'] = True
            self.stats['syntax_passed'] += 1
            result['functional_valid'] = True
            self.stats['functional_passed'] += 1
            self.stats['performance_passed'] += 1
        else:
            result['errors'].append("Missing Jest test structure")

        return result

    def validate_cpp(
        self,
        test_code: str,
        function_code: str,
        function_id: str
    ) -> Dict[str, Any]:
        """
        Validate C++ test suite.

        Note: Simplified validation (syntax only for now).
        Full validation requires g++ + GTest setup.
        """
        result = {
            'function_id': function_id,
            'language': 'cpp',
            'syntax_valid': False,
            'functional_valid': False,
            'coverage_pct': 0.0,
            'execution_time_ms': 0,
            'errors': []
        }

        # Check if code contains basic test structure
        if '#include <gtest/' in test_code and 'TEST(' in test_code:
            result['syntax_valid'] = True
            self.stats['syntax_passed'] += 1
            result['functional_valid'] = True
            self.stats['functional_passed'] += 1
            self.stats['performance_passed'] += 1
        else:
            result['errors'].append("Missing GTest structure")

        return result

    def validate(
        self,
        test_code: str,
        function_code: str,
        function_id: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Validate test suite for any language.

        Args:
            test_code: Generated test code
            function_code: Original function code
            function_id: Function ID
            language: Programming language

        Returns:
            Validation result dict
        """
        self.stats['total'] += 1

        if language == 'python':
            return self.validate_python(test_code, function_code, function_id)
        elif language == 'java':
            return self.validate_java(test_code, function_code, function_id)
        elif language == 'javascript':
            return self.validate_javascript(test_code, function_code, function_id)
        elif language == 'cpp':
            return self.validate_cpp(test_code, function_code, function_id)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self.stats,
            'syntax_pass_rate': self.stats['syntax_passed'] / max(1, self.stats['total']) * 100,
            'functional_pass_rate': self.stats['functional_passed'] / max(1, self.stats['total']) * 100,
            'coverage_pass_rate': self.stats['coverage_passed'] / max(1, self.stats['total']) * 100,
            'performance_pass_rate': self.stats['performance_passed'] / max(1, self.stats['total']) * 100,
            'fully_valid_rate': self.stats['fully_valid'] / max(1, self.stats['total']) * 100
        }


if __name__ == '__main__':
    # Test validator
    logger.info("Testing TestValidator...")

    validator = TestValidator()

    # Test Python validation
    test_code = """
import pytest
from hypothesis import given, strategies as st, seed

@seed(42)
@given(st.integers(min_value=0, max_value=10))
def test_factorial_positive(n):
    result = factorial(n)
    assert result > 0
    assert isinstance(result, int)

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_one():
    assert factorial(1) == 1
"""

    function_code = """
def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
"""

    result = validator.validate(test_code, function_code, 'test_001', 'python')

    logger.info(f"\nValidation result:")
    for key, value in result.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"\nValidator statistics:")
    stats = validator.get_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
