#!/bin/bash
# Quick monitoring script for Phase 4 pipeline

echo "════════════════════════════════════════════════════════════════"
echo "📊 PHASE 4 PIPELINE MONITOR"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if pipeline is running
if pgrep -f "run_phase4_experiments.py" > /dev/null; then
    echo "✅ Pipeline: RUNNING (PID: $(pgrep -f "run_phase4_experiments.py"))"
else
    echo "⚠️  Pipeline: NOT RUNNING"
fi

echo ""
echo "📈 DATABASE STATS:"
echo "────────────────────────────────────────────────────────────────"

psql codetransform -c "
SELECT
    COUNT(*) as total_transformations,
    COUNT(*) FILTER (WHERE se_score IS NULL) as pending_se_calc,
    COUNT(DISTINCT function_id) as unique_functions,
    COUNT(DISTINCT model) as unique_models,
    ROUND(SUM(cost_usd)::numeric, 2) as total_cost_usd,
    ROUND(AVG(latency_ms)::numeric, 0) as avg_latency_ms
FROM transformations;
" -x

echo ""
echo "📋 BREAKDOWN BY MODEL:"
echo "────────────────────────────────────────────────────────────────"

psql codetransform -c "
SELECT
    model,
    COUNT(*) as count,
    ROUND(SUM(cost_usd)::numeric, 2) as cost_usd,
    ROUND(AVG(latency_ms)::numeric, 0) as avg_latency_ms
FROM transformations
GROUP BY model
ORDER BY cost_usd DESC;
"

echo ""
echo "📊 BREAKDOWN BY TASK × INTENSITY:"
echo "────────────────────────────────────────────────────────────────"

psql codetransform -c "
SELECT
    task,
    strategy,
    COUNT(*) as count
FROM transformations
GROUP BY task, strategy
ORDER BY task, strategy;
"

echo ""
echo "🕐 LAST 10 LOG ENTRIES:"
echo "────────────────────────────────────────────────────────────────"
tail -10 phase4_full_run.log 2>/dev/null || echo "No log file found"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Run: ./monitor.sh (or: watch -n 30 ./monitor.sh for auto-refresh)"
echo "════════════════════════════════════════════════════════════════"
