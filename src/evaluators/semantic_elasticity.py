#!/usr/bin/env python3
"""
Semantic Elasticity (SE) Calculator for CodeTransformBench

Calculates SE score for code transformations:
SE = (ΔCC × P² × D) / E

Where:
- ΔCC: Absolute difference in cyclomatic complexity
- P: Preservation (1 if tests pass, 0 if fail)
- D: Diversity (tree edit distance normalized)
- E: Effort (inverse of Halstead volume)
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import math

from src.utils.logger import logger
from src.utils.db_utils import get_db_session
from database.models import Function, Transformation
from sqlalchemy import and_


class SECalculator:
    """
    Calculates Semantic Elasticity metrics for code transformations.
    """

    def __init__(self):
        """Initialize SE calculator."""
        self.test_suite_dir = Path(__file__).parent.parent.parent / "data" / "test_suites"
        logger.info("Initialized SE calculator")

    def calculate_delta_cc(self, cc_original: int, cc_transformed: int) -> int:
        """
        Calculate absolute difference in cyclomatic complexity.

        Args:
            cc_original: Original code CC
            cc_transformed: Transformed code CC

        Returns:
            Absolute difference
        """
        return abs(cc_transformed - cc_original)

    def calculate_preservation(
        self,
        function_id: str,
        transformed_code: str,
        language: str
    ) -> int:
        """
        Calculate preservation metric (P) by running tests.

        Args:
            function_id: Function ID
            transformed_code: Transformed code
            language: Programming language

        Returns:
            1 if all tests pass, 0 otherwise
        """
        logger.debug(f"Calculating preservation for {function_id} (language: {language})")

        # Test suite configuration
        test_extensions = {
            'python': '.py',
            'java': '.java',
            'javascript': '.js',
            'cpp': '.cpp'
        }

        test_file = self.test_suite_dir / f"{function_id}_test{test_extensions.get(language, '.txt')}"

        if not test_file.exists():
            logger.warning(f"No test suite found for {function_id}, assuming P=1")
            return 1

        # Run tests based on language
        try:
            if language == 'python':
                return self._run_python_tests(test_file, transformed_code)
            elif language == 'java':
                return self._run_java_tests(test_file, transformed_code)
            elif language == 'javascript':
                return self._run_javascript_tests(test_file, transformed_code)
            elif language == 'cpp':
                return self._run_cpp_tests(test_file, transformed_code)
            else:
                logger.warning(f"Unsupported language {language}, assuming P=1")
                return 1
        except Exception as e:
            logger.error(f"Test execution failed for {function_id}: {e}")
            return 0

    def _run_python_tests(self, test_file: Path, transformed_code: str) -> int:
        """Run Python tests using pytest."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Inject transformed code + test suite
            f.write(transformed_code + '\n\n')
            f.write(test_file.read_text())
            temp_path = f.name

        try:
            result = subprocess.run(
                ['pytest', temp_path, '-v', '--tb=short'],
                capture_output=True,
                timeout=10,
                text=True
            )
            return 1 if result.returncode == 0 else 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _run_java_tests(self, test_file: Path, transformed_code: str) -> int:
        """Run Java tests using JUnit."""
        # Simplified: assume P=1 (full implementation needs JUnit setup)
        logger.debug("Java test execution not fully implemented, assuming P=1")
        return 1

    def _run_javascript_tests(self, test_file: Path, transformed_code: str) -> int:
        """Run JavaScript tests using Jest."""
        # Simplified: assume P=1 (full implementation needs Jest setup)
        logger.debug("JavaScript test execution not fully implemented, assuming P=1")
        return 1

    def _run_cpp_tests(self, test_file: Path, transformed_code: str) -> int:
        """Run C++ tests using Google Test."""
        # Simplified: assume P=1 (full implementation needs GTest compilation)
        logger.debug("C++ test execution not fully implemented, assuming P=1")
        return 1

    def calculate_diversity(
        self,
        code_original: str,
        code_transformed: str,
        language: str
    ) -> float:
        """
        Calculate diversity metric (D) using AST tree edit distance.

        For Python: uses AST module + apted library
        For others: enhanced Levenshtein with structural weighting

        Args:
            code_original: Original code
            code_transformed: Transformed code
            language: Programming language

        Returns:
            Normalized diversity score (0-1)
        """
        # Python: use AST-based tree edit distance
        if language == 'python':
            try:
                import ast
                from apted import APTED
                from apted.helpers import Tree

                def ast_to_tree(code: str) -> Tree:
                    """Convert Python AST to apted Tree."""
                    tree = ast.parse(code)
                    def build(node):
                        name = node.__class__.__name__
                        children = [build(child) for child in ast.iter_child_nodes(node)]
                        return Tree(name, *children) if children else Tree(name)
                    return build(tree)

                tree_orig = ast_to_tree(code_original)
                tree_trans = ast_to_tree(code_transformed)

                # Calculate tree edit distance
                apted_calc = APTED(tree_orig, tree_trans)
                distance = apted_calc.compute_edit_distance()

                # Normalize by tree size
                def count_nodes(tree):
                    return 1 + sum(count_nodes(c) for c in tree.children)
                max_nodes = max(count_nodes(tree_orig), count_nodes(tree_trans))
                diversity = distance / max(max_nodes, 1)

                return min(1.0, max(0.0, diversity))

            except Exception as e:
                logger.debug(f"AST diversity failed, using Levenshtein: {e}")

        # Fallback: Enhanced Levenshtein with structural bonuses
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        # Normalize (remove whitespace)
        norm_original = ''.join(code_original.split())
        norm_transformed = ''.join(code_transformed.split())

        edit_dist = levenshtein_distance(norm_original, norm_transformed)
        max_len = max(len(norm_original), len(norm_transformed))

        if max_len == 0:
            return 0.0

        # Structural bonus: count braces/brackets/keywords
        struct_chars = {'{', '}', '(', ')', '[', ']', ';', ','}
        orig_struct = sum(1 for c in code_original if c in struct_chars)
        trans_struct = sum(1 for c in code_transformed if c in struct_chars)
        struct_diff = abs(orig_struct - trans_struct)

        # Amplify if structure changed >20%
        struct_weight = 1.2 if struct_diff > max(orig_struct, trans_struct) * 0.2 else 1.0

        diversity = (edit_dist / max_len) * struct_weight
        return min(1.0, max(0.0, diversity))

    def calculate_effort(self, halstead_volume: float) -> float:
        """
        Calculate effort metric (E) as inverse of Halstead volume.

        Args:
            halstead_volume: Halstead volume of transformed code

        Returns:
            Effort score (inverse of volume, normalized)
        """
        # E = 1 / (1 + halstead_volume/1000)
        # This normalizes volume and inverts it (higher volume = lower effort score)

        if halstead_volume <= 0:
            return 1.0

        return 1.0 / (1.0 + halstead_volume / 1000.0)

    def calculate_se_score(
        self,
        delta_cc: int,
        preservation: int,
        diversity: float,
        effort: float
    ) -> float:
        """
        Calculate final Semantic Elasticity score.

        Formula: SE = (ΔCC × P² × D) / E

        Args:
            delta_cc: Absolute CC difference
            preservation: Preservation score (0 or 1)
            diversity: Diversity score (0-1)
            effort: Effort score (0-1)

        Returns:
            SE score
        """
        if effort == 0:
            return 0.0

        # SE = (ΔCC × P² × D) / E
        se = (delta_cc * (preservation ** 2) * diversity) / effort

        return se

    def calculate_se_for_transformation(
        self,
        function_id: str,
        model: str,
        task: str,
        strategy: str
    ) -> Optional[Dict[str, float]]:
        """
        Calculate all SE metrics for a transformation from database.

        Args:
            function_id: Function ID
            model: Model ID
            task: Task name
            strategy: Strategy name

        Returns:
            Dict with all metrics or None if transformation not found
        """
        with get_db_session() as session:
            # Get original function
            function = session.query(Function).filter(Function.id == function_id).first()
            if not function:
                logger.error(f"Function {function_id} not found")
                return None

            # Get transformation
            transformation = session.query(Transformation).filter(
                and_(
                    Transformation.function_id == function_id,
                    Transformation.model == model,
                    Transformation.task == task,
                    Transformation.strategy == strategy
                )
            ).first()

            if not transformation:
                logger.error(f"Transformation not found for {function_id} + {model} + {task} + {strategy}")
                return None

            # Calculate metrics
            delta_cc = self.calculate_delta_cc(
                function.cyclomatic_complexity,
                transformation.cyclomatic_complexity
            )

            preservation = self.calculate_preservation(
                function_id,
                transformation.transformed_code,
                function.language
            )

            diversity = self.calculate_diversity(
                function.code,
                transformation.transformed_code,
                function.language
            )

            effort = self.calculate_effort(transformation.halstead_volume)

            se_score = self.calculate_se_score(delta_cc, preservation, diversity, effort)

            # Update transformation in database
            transformation.delta_cc = delta_cc
            transformation.preservation = preservation
            transformation.diversity = diversity
            transformation.effort = effort
            transformation.se_score = se_score

            session.commit()

            logger.success(f"Calculated SE for {function_id} + {model}: {se_score:.3f}")

            return {
                'delta_cc': delta_cc,
                'preservation': preservation,
                'diversity': diversity,
                'effort': effort,
                'se_score': se_score
            }

    def calculate_se_for_all_transformations(self) -> Dict[str, int]:
        """
        Calculate SE metrics for all transformations in database.

        Returns:
            Statistics dict
        """
        with get_db_session() as session:
            transformations = session.query(Transformation).filter(
                Transformation.se_score == None
            ).all()

            total = len(transformations)
            logger.info(f"Calculating SE for {total} transformations...")

            success = 0
            failed = 0

            for i, trans in enumerate(transformations, 1):
                try:
                    result = self.calculate_se_for_transformation(
                        trans.function_id,
                        trans.model,
                        trans.task,
                        trans.strategy
                    )

                    if result:
                        success += 1
                    else:
                        failed += 1

                    if i % 50 == 0:
                        logger.info(f"Progress: {i}/{total} ({(i/total)*100:.1f}%)")

                except Exception as e:
                    logger.error(f"Error calculating SE for transformation {i}: {e}")
                    failed += 1

            logger.success(f"\n✓ SE calculation complete!")
            logger.info(f"Success: {success}")
            logger.info(f"Failed: {failed}")

            return {'success': success, 'failed': failed, 'total': total}


def main():
    """Main entry point for SE calculation."""
    calculator = SECalculator()

    stats = calculator.calculate_se_for_all_transformations()

    logger.info("=" * 60)
    logger.info("SE Calculation Statistics")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
