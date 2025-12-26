#!/usr/bin/env python3
"""
Phase 4: Run Complete Transformation Experiments

Executes 6,300 transformations:
- 100 functions (stratified by language and complexity)
- 7 models (3 tier1, 2 tier2, 2 tier3)
- 3 tasks (obfuscate, deobfuscate, refactor)
- 3 intensity levels (light, medium, heavy)
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluators.transformation_pipeline import TransformationPipeline
from src.evaluators.semantic_elasticity import SECalculator
from src.utils.db_utils import get_db_session
from src.utils.logger import logger
from src.config import get_models_by_tier
from database.models import Function
from sqlalchemy import func


def select_stratified_functions(total: int = 100) -> List[Dict[str, Any]]:
    """
    Select functions stratified by language and complexity.

    Target distribution:
    - Python: 30, Java: 30, JavaScript: 20, C++: 20
    - Simple (CC≤10): 40%, Medium (CC 11-30): 40%, Complex (CC>30): 20%

    Args:
        total: Total number of functions to select

    Returns:
        List of function dicts
    """
    logger.info("=" * 80)
    logger.info("SELECTING STRATIFIED SAMPLE OF FUNCTIONS")
    logger.info("=" * 80)

    # Language distribution
    lang_targets = {
        'python': 30,
        'java': 30,
        'javascript': 20,
        'cpp': 20
    }

    # Complexity tiers (percentage of each language's quota)
    complexity_dist = {
        'simple': 0.4,   # CC ≤ 10
        'medium': 0.4,   # CC 11-30
        'complex': 0.2   # CC > 30
    }

    selected_functions = []

    with get_db_session() as session:
        for language, lang_count in lang_targets.items():
            logger.info(f"\nSelecting {language} functions (target: {lang_count})...")

            # Calculate per-complexity targets
            simple_count = int(lang_count * complexity_dist['simple'])
            medium_count = int(lang_count * complexity_dist['medium'])
            complex_count = lang_count - simple_count - medium_count  # Remaining

            # Select simple functions (CC ≤ 10)
            simple_funcs = session.query(Function).filter(
                Function.language == language,
                Function.cyclomatic_complexity <= 10
            ).order_by(func.random()).limit(simple_count).all()

            # Select medium functions (CC 11-30)
            medium_funcs = session.query(Function).filter(
                Function.language == language,
                Function.cyclomatic_complexity > 10,
                Function.cyclomatic_complexity <= 30
            ).order_by(func.random()).limit(medium_count).all()

            # Select complex functions (CC > 30)
            complex_funcs = session.query(Function).filter(
                Function.language == language,
                Function.cyclomatic_complexity > 30
            ).order_by(func.random()).limit(complex_count).all()

            # Combine and convert to dicts
            lang_functions = simple_funcs + medium_funcs + complex_funcs

            for f in lang_functions:
                selected_functions.append({
                    'id': f.id,
                    'language': f.language,
                    'code': f.code,
                    'cyclomatic_complexity': f.cyclomatic_complexity
                })

            logger.info(f"  Simple (CC≤10): {len(simple_funcs)}")
            logger.info(f"  Medium (CC 11-30): {len(medium_funcs)}")
            logger.info(f"  Complex (CC>30): {len(complex_funcs)}")
            logger.info(f"  Total {language}: {len(lang_functions)}")

    logger.info(f"\n✓ Selected {len(selected_functions)} functions total")
    logger.info("=" * 80)

    return selected_functions


async def run_experiments():
    """Run complete Phase 4 experiments."""

    logger.info("=" * 80)
    logger.info("PHASE 4: TRANSFORMATION EXPERIMENTS")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("  - 100 functions (stratified)")
    logger.info("  - 7 models (3 tier1, 2 tier2, 2 tier3)")
    logger.info("  - 3 tasks (obfuscate, deobfuscate, refactor)")
    logger.info("  - 3 intensity levels (light, medium, heavy)")
    logger.info("  - Total: 6,300 transformations")
    logger.info("  - Budget limit: $2,000")
    logger.info("=" * 80)

    # Select 100 functions
    functions = select_stratified_functions(total=100)

    if len(functions) < 100:
        logger.warning(f"⚠️  Only found {len(functions)} functions!")
        logger.warning("Proceeding with available functions...")

    if len(functions) == 0:
        logger.error("No functions found in database! Cannot proceed.")
        return

    # Model configuration (6 models across 3 tiers)
    tier1_models = [m['id'] for m in get_models_by_tier('tier1')]  # 2 models
    tier2_models = [m['id'] for m in get_models_by_tier('tier2')]  # 2 models
    tier3_models = [m['id'] for m in get_models_by_tier('tier3')]  # 2 models

    all_models = tier1_models + tier2_models + tier3_models

    logger.info(f"\nModels selected ({len(all_models)}):")
    for i, model in enumerate(all_models, 1):
        logger.info(f"  {i}. {model}")

    # Task configuration
    tasks = ['obfuscate', 'deobfuscate', 'refactor']

    # Strategy configuration (intensity levels)
    strategies = ['zero_shot_light', 'zero_shot_medium', 'zero_shot_heavy']

    logger.info(f"\nTasks: {tasks}")
    logger.info(f"Strategies: {strategies}")

    # Starting experiments
    logger.info("\n" + "=" * 80)
    logger.warning("⚠️  STARTING EXPERIMENTS")
    logger.warning("This will use OpenRouter API credits.")
    logger.info("=" * 80)

    # Initialize pipeline with budget limit
    pipeline = TransformationPipeline(budget_limit_usd=2000.0)  # $2000 total budget

    # Run transformations
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRANSFORMATION PIPELINE")
    logger.info("=" * 80)

    stats = await pipeline.run_transformations(
        functions=functions,
        models=all_models,
        tasks=tasks,
        strategies=strategies,
        max_concurrent=5,  # Parallel requests
        resume=True  # Skip existing transformations
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("TRANSFORMATION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Completed: {stats['completed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped (existing): {stats['skipped']}")
    logger.info(f"Total cost: ${stats['total_cost']:.2f}")
    logger.info("=" * 80)

    # Calculate SE metrics for all transformations
    if stats['completed'] > 0:
        logger.info("\n" + "=" * 80)
        logger.info("CALCULATING SE METRICS FOR ALL TRANSFORMATIONS")
        logger.info("=" * 80)

        calculator = SECalculator()
        se_stats = calculator.calculate_se_for_all_transformations()

        logger.info("\n" + "=" * 80)
        logger.info("SE CALCULATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Success: {se_stats['success']}")
        logger.info(f"Failed: {se_stats['failed']}")
        logger.info(f"Total: {se_stats['total']}")
        logger.info("=" * 80)

    logger.success("\n✓ PHASE 4 EXPERIMENTS COMPLETE!")
    logger.info(f"Next steps:")
    logger.info(f"  1. Analyze results in database (transformations table)")
    logger.info(f"  2. Query leaderboard view for model rankings")
    logger.info(f"  3. Generate visualizations and statistical analysis")


if __name__ == '__main__':
    asyncio.run(run_experiments())
