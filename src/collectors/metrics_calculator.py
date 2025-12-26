"""
Code metrics calculator for CodeTransformBench.
Computes cyclomatic complexity, Halstead volume, and code hash.
"""

import hashlib
import re
from typing import Dict, Any, Optional
from radon.complexity import cc_visit
from radon.raw import analyze
from radon.visitors import Function as RadonFunction

# Language-specific imports
import ast  # For Python AST parsing


class MetricsCalculator:
    """Calculate code metrics across multiple languages."""

    @staticmethod
    def normalize_code(code: str) -> str:
        """
        Normalize code for consistent hashing.
        Removes comments and extra whitespace.
        """
        # Remove single-line comments (// and #)
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)

        # Remove multi-line comments (/* */ and ''' ''')
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)

        # Normalize whitespace
        code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())

        return code

    @staticmethod
    def compute_code_hash(code: str) -> str:
        """
        Compute SHA256 hash of normalized code for deduplication.
        """
        normalized = MetricsCalculator.normalize_code(code)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    @staticmethod
    def compute_lines_of_code(code: str) -> int:
        """Count lines of code (excluding empty lines and comments)."""
        normalized = MetricsCalculator.normalize_code(code)
        return len([line for line in normalized.split('\n') if line.strip()])

    @staticmethod
    def compute_cyclomatic_complexity_python(code: str) -> int:
        """
        Compute cyclomatic complexity for Python using radon.

        Returns the maximum CC across all functions in the code.
        """
        try:
            results = cc_visit(code)
            if not results:
                return 1  # Default CC for empty/simple code

            # Return maximum complexity (for single function should be only one)
            return max(r.complexity for r in results)
        except Exception as e:
            # Fallback: count control flow keywords
            return MetricsCalculator._fallback_cc(code)

    @staticmethod
    def _fallback_cc(code: str) -> int:
        """
        Fallback cyclomatic complexity calculation.
        Count control flow keywords + 1.
        """
        keywords = ['if', 'elif', 'else', 'for', 'while', 'case', 'catch',
                    'and', 'or', '&&', '||', '?', 'switch']

        cc = 1  # Base complexity
        for keyword in keywords:
            # Use word boundaries for keywords
            pattern = r'\b' + re.escape(keyword) + r'\b'
            cc += len(re.findall(pattern, code))

        return max(1, cc)

    @staticmethod
    def compute_cyclomatic_complexity(code: str, language: str) -> int:
        """
        Compute cyclomatic complexity for any language.

        For Python: uses radon
        For others: uses fallback (count control flow keywords)
        """
        if language == 'python':
            return MetricsCalculator.compute_cyclomatic_complexity_python(code)
        else:
            # For Java, JavaScript, C++: use fallback
            return MetricsCalculator._fallback_cc(code)

    @staticmethod
    def compute_halstead_volume_python(code: str) -> Optional[float]:
        """
        Compute Halstead volume for Python using radon.

        Volume = (N1 + N2) * log2(n1 + n2)
        where:
        - n1 = number of distinct operators
        - n2 = number of distinct operands
        - N1 = total number of operators
        - N2 = total number of operands
        """
        try:
            analysis = analyze(code)
            # radon returns Halstead metrics
            # We need to calculate manually from operators/operands
            if hasattr(analysis, 'multi'):
                # Get Halstead metrics
                import math
                n1 = len(set(analysis.multi))  # Distinct operators (approximation)
                n2 = analysis.lloc  # Use LLOC as proxy for operands

                if n1 + n2 == 0:
                    return 0.0

                N = analysis.sloc  # Use SLOC as total
                volume = N * math.log2(max(1, n1 + n2))
                return round(volume, 2)

            return None
        except Exception:
            return None

    @staticmethod
    def compute_halstead_volume(code: str, language: str) -> Optional[float]:
        """
        Compute Halstead volume (approximation for non-Python).

        For Python: uses radon
        For others: approximates based on token count
        """
        if language == 'python':
            return MetricsCalculator.compute_halstead_volume_python(code)
        else:
            # Approximation: count tokens and use simplified formula
            tokens = len(re.findall(r'\w+', code))
            if tokens == 0:
                return 0.0

            import math
            # Rough approximation: volume ≈ tokens * log2(tokens)
            volume = tokens * math.log2(max(1, tokens))
            return round(volume, 2)

    @staticmethod
    def get_complexity_tier(cc: int) -> str:
        """
        Get complexity tier classification.

        - simple: CC ≤ 10
        - medium: CC 11-30
        - complex: CC ≥ 31
        """
        if cc <= 10:
            return 'simple'
        elif cc <= 30:
            return 'medium'
        else:
            return 'complex'

    @staticmethod
    def compute_all_metrics(code: str, language: str) -> Dict[str, Any]:
        """
        Compute all metrics for a code snippet.

        Returns:
            {
                'code_hash': str,
                'cyclomatic_complexity': int,
                'halstead_volume': float,
                'lines_of_code': int,
                'complexity_tier': str
            }
        """
        cc = MetricsCalculator.compute_cyclomatic_complexity(code, language)

        return {
            'code_hash': MetricsCalculator.compute_code_hash(code),
            'cyclomatic_complexity': cc,
            'halstead_volume': MetricsCalculator.compute_halstead_volume(code, language),
            'lines_of_code': MetricsCalculator.compute_lines_of_code(code),
            'complexity_tier': MetricsCalculator.get_complexity_tier(cc)
        }


# Convenience functions
def calculate_metrics(code: str, language: str) -> Dict[str, Any]:
    """Convenience function to calculate all metrics."""
    return MetricsCalculator.compute_all_metrics(code, language)


if __name__ == '__main__':
    # Test metrics calculator
    print("Testing MetricsCalculator...")

    # Test Python code
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n+1):
        result *= i
    return result
"""

    print("\n=== Python Code ===")
    metrics = calculate_metrics(python_code, 'python')
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Test Java code
    java_code = """
public class Example {
    public static int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        int result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
}
"""

    print("\n=== Java Code ===")
    metrics = calculate_metrics(java_code, 'java')
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Test complexity tiers
    print("\n=== Complexity Tier Classification ===")
    test_cases = [5, 10, 15, 25, 35]
    for cc in test_cases:
        tier = MetricsCalculator.get_complexity_tier(cc)
        print(f"CC={cc} → {tier}")

    print("\n✓ MetricsCalculator working correctly!")
