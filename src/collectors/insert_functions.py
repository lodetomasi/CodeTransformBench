
#!/usr/bin/env python3
"""
Function inserter for CodeTransformBench.
Loads functions from JSONL files and inserts them into the database.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.utils.db_utils import get_db_session
from src.utils.logger import logger
from database.models import Function


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load functions from JSONL file."""
    functions = []
    with open(file_path, 'r') as f:
        for line in f:
            functions.append(json.loads(line))
    return functions


def insert_functions(functions: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Insert functions into database.

    Args:
        functions: List of function dictionaries

    Returns:
        Statistics about insertion
    """
    stats = {
        'total': len(functions),
        'inserted': 0,
        'duplicates': 0,
        'errors': 0
    }

    with get_db_session() as session:
        for func_data in functions:
            try:
                # Create Function object
                function = Function(
                    id=func_data['id'],
                    language=func_data['language'],
                    code=func_data['code'],
                    code_hash=func_data['code_hash'],
                    cyclomatic_complexity=func_data['cyclomatic_complexity'],
                    halstead_volume=func_data.get('halstead_volume'),
                    lines_of_code=func_data['lines_of_code'],
                    domain=func_data.get('domain', 'other'),
                    source=func_data['source'],
                    task_name=func_data.get('task_name', ''),
                    validated=False  # Will be validated after test generation
                )

                # Try to insert
                session.add(function)
                session.flush()  # Flush to catch constraint violations

                stats['inserted'] += 1
                logger.success(f"Inserted {function.id}")

            except Exception as e:
                session.rollback()

                # Check if duplicate
                if 'duplicate key' in str(e).lower() or 'unique constraint' in str(e).lower():
                    stats['duplicates'] += 1
                    logger.debug(f"Duplicate skipped: {func_data['id']}")
                else:
                    stats['errors'] += 1
                    logger.error(f"Error inserting {func_data['id']}: {e}")

                # Continue with next function
                continue

        # Commit all successful insertions
        session.commit()

    return stats


def validate_distribution(required: Dict[str, int], actual: Dict[str, int]) -> bool:
    """
    Validate that distribution meets requirements.

    Args:
        required: Required distribution
        actual: Actual distribution

    Returns:
        True if distribution is acceptable
    """
    logger.info("\n=== Distribution Validation ===")
    logger.info("Language distribution:")

    all_ok = True
    for lang, required_count in required.items():
        actual_count = actual.get(lang, 0)
        percentage = (actual_count / required_count * 100) if required_count > 0 else 0

        status = "✓" if actual_count >= required_count * 0.9 else "✗"
        logger.info(f"  {lang}: {actual_count}/{required_count} ({percentage:.1f}%) {status}")

        if actual_count < required_count * 0.9:  # Allow 10% tolerance
            all_ok = False

    return all_ok


def main():
    """Main entry point for function inserter."""
    logger.info("Starting function insertion...")

    # Load functions from JSONL files
    rosetta_file = config.DATA_PROCESSED_DIR / 'rosetta_code_functions.jsonl'
    algorithms_file = config.DATA_PROCESSED_DIR / 'algorithms_functions.jsonl'

    all_functions = []

    if rosetta_file.exists():
        rosetta_functions = load_jsonl(rosetta_file)
        logger.info(f"Loaded {len(rosetta_functions)} functions from Rosetta Code")
        all_functions.extend(rosetta_functions)
    else:
        logger.warning(f"Rosetta Code file not found: {rosetta_file}")

    if algorithms_file.exists():
        algorithms_functions = load_jsonl(algorithms_file)
        logger.info(f"Loaded {len(algorithms_functions)} functions from TheAlgorithms")
        all_functions.extend(algorithms_functions)
    else:
        logger.warning(f"TheAlgorithms file not found: {algorithms_file}")

    if not all_functions:
        logger.error("No functions to insert!")
        return

    logger.info(f"\nTotal functions to insert: {len(all_functions)}")

    # Insert into database
    stats = insert_functions(all_functions)

    # Print statistics
    logger.info("\n=== Insertion Statistics ===")
    logger.info(f"Total: {stats['total']}")
    logger.info(f"Inserted: {stats['inserted']}")
    logger.info(f"Duplicates: {stats['duplicates']}")
    logger.info(f"Errors: {stats['errors']}")

    # Check distribution
    by_lang = defaultdict(int)
    by_tier = defaultdict(int)

    with get_db_session() as session:
        functions = session.query(Function).all()

        for func in functions:
            by_lang[func.language] += 1

            if func.cyclomatic_complexity <= 10:
                by_tier['simple'] += 1
            elif func.cyclomatic_complexity <= 30:
                by_tier['medium'] += 1
            else:
                by_tier['complex'] += 1

    # Required distribution
    required_lang = {
        'python': 150,
        'java': 150,
        'javascript': 100,
        'cpp': 100
    }

    # Validate
    lang_ok = validate_distribution(required_lang, dict(by_lang))

    logger.info("\nComplexity distribution:")
    for tier, count in sorted(by_tier.items()):
        logger.info(f"  {tier}: {count}")

    # Final status
    if lang_ok and stats['inserted'] >= 450:  # At least 90% of target
        logger.success("\n✓ Function insertion SUCCESSFUL!")
        logger.success(f"Database contains {len(functions)} validated functions")
    else:
        logger.warning("\n⚠ Function insertion completed with warnings")
        logger.warning(f"Database contains {len(functions)} functions (target: 500)")


if __name__ == '__main__':
    main()
