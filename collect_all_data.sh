#!/bin/bash
# Master script to collect all 500 functions for CodeTransformBench
# Executes Phase 2: Data Collection

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "CodeTransformBench - Data Collection"
echo "Target: 500 functions across 4 languages"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Step 1/3: Collecting from Rosetta Code (260 functions)..."
echo "This will take 1-2 hours due to rate limiting (1 req/sec)"
python src/collectors/rosetta_scraper.py

echo ""
echo "Step 2/3: Collecting from TheAlgorithms (240 functions)..."
echo "This will take 20-30 minutes"
python src/collectors/algorithms_cloner.py

echo ""
echo "Step 3/3: Inserting functions into database..."
python src/collectors/insert_functions.py

echo ""
echo "=========================================="
echo "✓ Data collection complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check database: python src/utils/db_utils.py"
echo "  2. Start Phase 3: Test generation"
echo ""
