#!/bin/bash
# Monitor Phase 4 Pipeline Progress

echo "================================================================================"
echo "PHASE 4 PIPELINE MONITOR"
echo "================================================================================"
echo ""

# Overall progress
echo "📊 OVERALL PROGRESS:"
psql codetransform -c "
SELECT
    COUNT(*) as completed_transformations,
    COUNT(DISTINCT function_id) as functions_transformed,
    COUNT(DISTINCT model) as models_used,
    ROUND(SUM(cost_usd)::numeric, 4) as total_cost_usd,
    ROUND(AVG(latency_ms)::numeric, 0) as avg_latency_ms
FROM transformations;
" -x

echo ""
echo "📈 BY MODEL:"
psql codetransform -c "
SELECT
    model,
    COUNT(*) as count,
    ROUND(SUM(cost_usd)::numeric, 4) as cost,
    ROUND(AVG(delta_cc)::numeric, 2) as avg_delta_cc
FROM transformations
GROUP BY model
ORDER BY count DESC;
"

echo ""
echo "📋 BY TASK × STRATEGY:"
psql codetransform -c "
SELECT
    task,
    strategy,
    COUNT(*) as count,
    ROUND(AVG(delta_cc)::numeric, 2) as avg_delta_cc
FROM transformations
GROUP BY task, strategy
ORDER BY task, strategy;
"

echo ""
echo "⏱️  RECENT TRANSFORMATIONS (last 5):"
psql codetransform -c "
SELECT
    function_id,
    model,
    task,
    strategy,
    delta_cc,
    ROUND(cost_usd::numeric, 6) as cost,
    timestamp
FROM transformations
ORDER BY timestamp DESC
LIMIT 5;
"

echo ""
echo "================================================================================"
echo "Target: 3,024 transformations (56 functions × 6 models × 3 tasks × 3 strategies)"
echo "Log: tail -f /tmp/phase4_clean.log"
echo "================================================================================"
