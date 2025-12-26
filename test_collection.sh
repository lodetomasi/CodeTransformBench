#!/bin/bash
# Quick test script to validate data collection pipeline
# Collects only ~20 functions for testing

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "CodeTransformBench - Quick Test"
echo "Target: ~20 functions for validation"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Testing Rosetta Code scraper (10 functions)..."
python -c "
from src.collectors.rosetta_scraper import RosettaCodeScraper
from src.config import config

scraper = RosettaCodeScraper()
# Small target for quick test
target = {'python': 3, 'java': 3, 'javascript': 2, 'cpp': 2}
functions = scraper.collect_functions(target)

print(f'\n✓ Collected {len(functions)} functions from Rosetta Code')

# Save
output_path = config.DATA_PROCESSED_DIR / 'rosetta_code_functions.jsonl'
scraper.save_to_jsonl(functions, output_path)
"

echo ""
echo "Testing TheAlgorithms cloner (10 functions)..."
python -c "
from src.collectors.algorithms_cloner import TheAlgorithmsCloner
from src.config import config

cloner = TheAlgorithmsCloner()
# Small target for quick test
target = {'python': 3, 'java': 3, 'javascript': 2, 'cpp': 2}
functions = cloner.collect_functions(target)

print(f'\n✓ Collected {len(functions)} functions from TheAlgorithms')

# Save
output_path = config.DATA_PROCESSED_DIR / 'algorithms_functions.jsonl'
cloner.save_to_jsonl(functions, output_path)
"

echo ""
echo "Testing database insertion..."
python src/collectors/insert_functions.py

echo ""
echo "=========================================="
echo "✓ Quick test complete!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  python src/utils/db_utils.py"
echo ""
