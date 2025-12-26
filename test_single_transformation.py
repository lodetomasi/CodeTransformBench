#!/usr/bin/env python3
"""
Test script to verify transformation pipeline works end-to-end.

Tests:
1. Load a function from database
2. Transform it using one model
3. Calculate SE metrics
4. Verify checkpoint/resume works
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluators.transformation_pipeline import TransformationPipeline
from src.evaluators.semantic_elasticity import SECalculator
from src.utils.db_utils import get_db_session
from src.utils.logger import logger
from database.models import Function, Transformation


async def test_single_transformation():
    """Test transformation pipeline with a single function."""

    logger.info("=" * 80)
    logger.info("TESTING TRANSFORMATION PIPELINE - SINGLE FUNCTION")
    logger.info("=" * 80)

    # Get one Python function
    with get_db_session() as session:
        db_func = session.query(Function).filter(Function.language == 'python').first()

        if not db_func:
            logger.error("No Python function found in database!")
            return

        func = {
            'id': db_func.id,
            'language': db_func.language,
            'code': db_func.code,
            'cyclomatic_complexity': db_func.cyclomatic_complexity
        }

    logger.info(f"\nSelected function: {func['id']}")
    logger.info(f"Language: {func['language']}")
    logger.info(f"Cyclomatic Complexity: {func['cyclomatic_complexity']}")
    logger.info(f"\nOriginal code (first 500 chars):")
    logger.info("-" * 80)
    logger.info(func['code'][:500] + "...")
    logger.info("-" * 80)

    # Initialize pipeline
    pipeline = TransformationPipeline(budget_limit_usd=10.0)

    # Test with one model (cheapest one for testing)
    test_model = 'meta-llama/llama-3.1-8b-instruct'
    test_task = 'obfuscate'
    test_strategy = 'zero_shot'

    logger.info(f"\nTransformation config:")
    logger.info(f"  Model: {test_model}")
    logger.info(f"  Task: {test_task}")
    logger.info(f"  Strategy: {test_strategy}")

    # Check if already exists
    exists = pipeline.check_transformation_exists(func['id'], test_model, test_task, test_strategy)

    if exists:
        logger.warning(f"\n⚠️  Transformation already exists in database!")
        logger.info("Testing resume capability - will skip this transformation")
    else:
        logger.info("\n✓ Transformation doesn't exist yet, will create it")

    # Run transformation
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING TRANSFORMATION...")
    logger.info("=" * 80)

    stats = await pipeline.run_transformations(
        functions=[func],
        models=[test_model],
        tasks=[test_task],
        strategies=[test_strategy],
        max_concurrent=1,
        resume=True  # Test resume capability
    )

    logger.info("\n" + "=" * 80)
    logger.info("TRANSFORMATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Completed: {stats['completed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Cost: ${stats['total_cost']:.4f}")

    # Retrieve and show transformed code
    with get_db_session() as session:
        transformation = session.query(Transformation).filter(
            Transformation.function_id == func['id'],
            Transformation.model == test_model,
            Transformation.task == test_task,
            Transformation.strategy == test_strategy
        ).first()

        if transformation:
            logger.info("\n" + "=" * 80)
            logger.info("TRANSFORMED CODE")
            logger.info("=" * 80)
            logger.info(transformation.transformed_code)
            logger.info("=" * 80)

            logger.info(f"\nTransformed CC: {transformation.cyclomatic_complexity}")
            logger.info(f"Halstead Volume: {transformation.halstead_volume:.2f}")
            logger.info(f"Cost: ${transformation.cost_usd:.4f}")
            logger.info(f"Latency: {transformation.latency_ms}ms")

    # Calculate SE metrics
    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING SE METRICS...")
    logger.info("=" * 80)

    calculator = SECalculator()
    se_result = calculator.calculate_se_for_transformation(
        func['id'],
        test_model,
        test_task,
        test_strategy
    )

    if se_result:
        logger.success("\n✓ SE Calculation Complete!")
        logger.info("=" * 80)
        logger.info(f"ΔCC (complexity change): {se_result['delta_cc']}")
        logger.info(f"P (preservation): {se_result['preservation']}")
        logger.info(f"D (diversity): {se_result['diversity']:.3f}")
        logger.info(f"E (effort): {se_result['effort']:.3f}")
        logger.info(f"SE Score: {se_result['se_score']:.3f}")
        logger.info("=" * 80)
    else:
        logger.error("✗ SE Calculation Failed!")

    # Test resume capability
    logger.info("\n" + "=" * 80)
    logger.info("TESTING RESUME CAPABILITY...")
    logger.info("=" * 80)
    logger.info("Running pipeline again with same parameters...")

    stats2 = await pipeline.run_transformations(
        functions=[func],
        models=[test_model],
        tasks=[test_task],
        strategies=[test_strategy],
        max_concurrent=1,
        resume=True
    )

    logger.info(f"\nSecond run skipped: {stats2['skipped']}")
    logger.info(f"Second run completed: {stats2['completed']}")

    if stats2['skipped'] == 1 and stats2['completed'] == 0:
        logger.success("✓ Resume capability works correctly!")
    else:
        logger.error("✗ Resume capability issue - transformation was re-run!")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE!")
    logger.info("=" * 80)


if __name__ == '__main__':
    asyncio.run(test_single_transformation())
