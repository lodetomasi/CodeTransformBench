#!/usr/bin/env python3
"""
Test script to verify test generation for a single function.
Shows the prompt being used and tests code extraction.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.generators.test_generator import TestGenerator
from src.utils.db_utils import get_db_session
from src.utils.logger import logger
from database.models import Function


async def test_single_generation():
    """Test generation for a single function."""

    generator = TestGenerator()

    # Get one function from each language and convert to dicts
    with get_db_session() as session:
        db_python = session.query(Function).filter(Function.language == 'python').first()
        db_java = session.query(Function).filter(Function.language == 'java').first()
        db_javascript = session.query(Function).filter(Function.language == 'javascript').first()
        db_cpp = session.query(Function).filter(Function.language == 'cpp').first()

        # Convert to dicts to avoid detached instance errors
        python_func = None
        if db_python:
            python_func = {
                'id': db_python.id,
                'language': db_python.language,
                'code': db_python.code,
                'cyclomatic_complexity': db_python.cyclomatic_complexity
            }

        java_func = None
        if db_java:
            java_func = {
                'id': db_java.id,
                'language': db_java.language,
                'code': db_java.code,
                'cyclomatic_complexity': db_java.cyclomatic_complexity
            }

        javascript_func = None
        if db_javascript:
            javascript_func = {
                'id': db_javascript.id,
                'language': db_javascript.language,
                'code': db_javascript.code,
                'cyclomatic_complexity': db_javascript.cyclomatic_complexity
            }

        cpp_func = None
        if db_cpp:
            cpp_func = {
                'id': db_cpp.id,
                'language': db_cpp.language,
                'code': db_cpp.code,
                'cyclomatic_complexity': db_cpp.cyclomatic_complexity
            }

    # Test Python
    if python_func:
        logger.info("=" * 80)
        logger.info("TESTING PYTHON FUNCTION")
        logger.info("=" * 80)
        logger.info(f"Function ID: {python_func['id']}")
        logger.info(f"Cyclomatic Complexity: {python_func['cyclomatic_complexity']}")
        logger.info(f"\nFunction code:\n{python_func['code'][:500]}...\n")

        # Show the template
        template = generator.load_template('python')
        logger.info("=" * 80)
        logger.info("PROMPT TEMPLATE (Python):")
        logger.info("=" * 80)
        logger.info(template)

        # Show the full prompt
        prompt = generator.render_prompt(template, python_func['code'])
        logger.info("=" * 80)
        logger.info("FULL PROMPT (first 1000 chars):")
        logger.info("=" * 80)
        logger.info(prompt[:1000] + "...\n")

        # Generate test
        logger.info("=" * 80)
        logger.info("GENERATING TEST...")
        logger.info("=" * 80)

        result = await generator.generate_test_suite(python_func)

        if result:
            logger.success("✓ Test generation SUCCESSFUL!")
            logger.info(f"Cost: ${result['cost_usd']:.4f}")
            logger.info(f"Tokens: {result['tokens_input']} in + {result['tokens_output']} out")
            logger.info(f"Latency: {result['latency_ms']}ms")
            logger.info("=" * 80)
            logger.info("GENERATED TEST CODE:")
            logger.info("=" * 80)
            logger.info(result['test_code'])
            logger.info("=" * 80)

            # Save to file
            filepath = generator.save_test_file(result)
            logger.success(f"✓ Saved to: {filepath}")
        else:
            logger.error("✗ Test generation FAILED!")

    # Test all languages
    all_funcs = [
        ('Python', python_func),
        ('Java', java_func),
        ('JavaScript', javascript_func),
        ('C++', cpp_func)
    ]

    results = []

    for lang_name, func in all_funcs:
        if not func:
            logger.warning(f"\n⚠ No {lang_name} function found, skipping...")
            continue

        logger.info("\n" + "=" * 80)
        logger.info(f"TESTING {lang_name.upper()} FUNCTION")
        logger.info("=" * 80)
        logger.info(f"Function ID: {func['id']}")
        logger.info(f"Cyclomatic Complexity: {func['cyclomatic_complexity']}")
        logger.info(f"\nFunction code (first 300 chars):\n{func['code'][:300]}...\n")

        # Generate test
        logger.info(f"Generating {lang_name} test...")
        result = await generator.generate_test_suite(func)

        if result:
            logger.success(f"✓ {lang_name} test generation SUCCESSFUL!")
            logger.info(f"Cost: ${result['cost_usd']:.4f}")
            logger.info(f"Tokens: {result['tokens_input']} in + {result['tokens_output']} out")

            # Save to file
            filepath = generator.save_test_file(result)
            logger.success(f"✓ Saved to: {filepath}")

            results.append({
                'language': lang_name,
                'success': True,
                'cost': result['cost_usd'],
                'tokens': result['tokens_input'] + result['tokens_output']
            })
        else:
            logger.error(f"✗ {lang_name} test generation FAILED!")
            results.append({
                'language': lang_name,
                'success': False,
                'cost': 0,
                'tokens': 0
            })

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - Test Generation Results")
    logger.info("=" * 80)

    total_cost = 0
    total_tokens = 0
    success_count = 0

    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        logger.info(f"{r['language']:12} | {status:12} | ${r['cost']:.4f} | {r['tokens']:5} tokens")
        total_cost += r['cost']
        total_tokens += r['tokens']
        if r['success']:
            success_count += 1

    logger.info("=" * 80)
    logger.info(f"Total: {success_count}/{len(results)} successful")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Estimated cost for 240 functions: ${total_cost * 60:.2f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    asyncio.run(test_single_generation())
