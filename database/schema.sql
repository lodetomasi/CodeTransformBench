-- CodeTransformBench PostgreSQL Schema
-- Creates database structure for LLM code transformation benchmark

-- Drop existing tables if they exist (for clean re-initialization)
DROP TABLE IF EXISTS cost_tracking CASCADE;
DROP TABLE IF EXISTS transformations CASCADE;
DROP TABLE IF EXISTS functions CASCADE;

-- Table 1: functions - Source code corpus
CREATE TABLE functions (
    id VARCHAR(100) PRIMARY KEY,  -- Format: {lang}_{source}_{task}_{num}
    language VARCHAR(20) NOT NULL CHECK (language IN ('python', 'java', 'javascript', 'cpp')),
    code TEXT NOT NULL,
    code_hash CHAR(64) UNIQUE NOT NULL,  -- SHA256 for deduplication
    cyclomatic_complexity INTEGER NOT NULL CHECK (cyclomatic_complexity >= 1),
    halstead_volume FLOAT,
    lines_of_code INTEGER NOT NULL CHECK (lines_of_code > 0),
    domain VARCHAR(50),  -- e.g., 'algorithms', 'data_structures', 'strings', 'math', 'io'
    source VARCHAR(50) NOT NULL,  -- 'rosetta_code' or 'the_algorithms'
    task_name VARCHAR(200),
    test_suite_path VARCHAR(255),
    validated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Complexity tier for stratification
CREATE INDEX idx_functions_complexity ON functions(cyclomatic_complexity);
CREATE INDEX idx_functions_language ON functions(language);
CREATE INDEX idx_functions_domain ON functions(domain);
CREATE INDEX idx_functions_validated ON functions(validated);

-- Table 2: transformations - Experiment results
CREATE TABLE transformations (
    id SERIAL PRIMARY KEY,
    function_id VARCHAR(100) NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    model VARCHAR(100) NOT NULL,
    task VARCHAR(50) NOT NULL CHECK (task IN ('obfuscate', 'deobfuscate', 'refactor')),
    strategy VARCHAR(50) NOT NULL CHECK (strategy IN ('zero_shot', 'few_shot_k3', 'few_shot_k5', 'chain_of_thought', 'self_reflection')),
    temperature FLOAT DEFAULT 0.2,
    transformed_code TEXT,

    -- Semantic Elasticity components
    delta_cc FLOAT,  -- |CC(orig) - CC(trans)|
    preservation INTEGER CHECK (preservation IN (0, 1)),  -- 1 if tests pass, 0 if fail
    diversity FLOAT,  -- tree_edit_distance / max_size
    effort FLOAT,  -- 1 / (1 + halstead_volume/1000)
    se_score FLOAT,  -- (delta_cc * preservation^2 * diversity) / effort

    -- API metadata
    cost_usd FLOAT,
    latency_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    error_type VARCHAR(50),  -- 'parse_error', 'timeout', 'rate_limit', etc.

    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Prevent duplicate experiments
    UNIQUE(function_id, model, task, strategy)
);

-- Indexes for leaderboard queries (GROUP BY model, task)
CREATE INDEX idx_transformations_model_task ON transformations(model, task, se_score DESC);
CREATE INDEX idx_transformations_function ON transformations(function_id);
CREATE INDEX idx_transformations_timestamp ON transformations(timestamp);

-- Table 3: cost_tracking - Budget monitoring
CREATE TABLE cost_tracking (
    date DATE NOT NULL,
    model VARCHAR(100) NOT NULL,
    total_calls INTEGER DEFAULT 0,
    total_cost_usd FLOAT DEFAULT 0.0,
    avg_latency_ms FLOAT,
    successful_calls INTEGER DEFAULT 0,
    failed_calls INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (date, model)
);

-- Trigger: Auto-update cost_tracking on INSERT to transformations
CREATE OR REPLACE FUNCTION update_cost_tracking()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO cost_tracking (date, model, total_calls, total_cost_usd, avg_latency_ms, successful_calls, failed_calls)
    VALUES (
        CURRENT_DATE,
        NEW.model,
        1,
        COALESCE(NEW.cost_usd, 0.0),
        COALESCE(NEW.latency_ms::FLOAT, 0.0),
        CASE WHEN NEW.error_type IS NULL THEN 1 ELSE 0 END,
        CASE WHEN NEW.error_type IS NOT NULL THEN 1 ELSE 0 END
    )
    ON CONFLICT (date, model) DO UPDATE SET
        total_calls = cost_tracking.total_calls + 1,
        total_cost_usd = cost_tracking.total_cost_usd + COALESCE(EXCLUDED.total_cost_usd, 0.0),
        avg_latency_ms = (
            (cost_tracking.avg_latency_ms * cost_tracking.total_calls + COALESCE(EXCLUDED.avg_latency_ms, 0.0))
            / (cost_tracking.total_calls + 1)
        ),
        successful_calls = cost_tracking.successful_calls + EXCLUDED.successful_calls,
        failed_calls = cost_tracking.failed_calls + EXCLUDED.failed_calls,
        updated_at = CURRENT_TIMESTAMP;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_cost_tracking
AFTER INSERT ON transformations
FOR EACH ROW
EXECUTE FUNCTION update_cost_tracking();

-- Create view for leaderboard (pre-computed aggregation)
CREATE OR REPLACE VIEW leaderboard AS
SELECT
    model,
    task,
    strategy,
    COUNT(*) as total_transformations,
    SUM(CASE WHEN preservation = 1 THEN 1 ELSE 0 END) as successful,
    ROUND(CAST(AVG(CASE WHEN preservation = 1 THEN 1.0 ELSE 0.0 END) * 100 AS NUMERIC), 2) as success_rate_pct,
    ROUND(CAST(AVG(se_score) AS NUMERIC), 3) as mean_se,
    ROUND(CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY se_score) AS NUMERIC), 3) as median_se,
    ROUND(CAST(STDDEV(se_score) AS NUMERIC), 3) as std_se,
    SUM(cost_usd) as total_cost_usd,
    ROUND(CAST(AVG(latency_ms) AS NUMERIC), 0) as avg_latency_ms
FROM transformations
WHERE se_score IS NOT NULL
GROUP BY model, task, strategy
ORDER BY mean_se DESC;

-- Verification queries
-- Check all tables exist:
-- SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- Check indexes:
-- SELECT indexname, tablename FROM pg_indexes WHERE schemaname = 'public';

-- Check trigger:
-- SELECT tgname, tgrelid::regclass FROM pg_trigger WHERE tgname = 'trigger_update_cost_tracking';

COMMENT ON TABLE functions IS 'Source code corpus: 500 functions across 4 languages';
COMMENT ON TABLE transformations IS 'Experiment results: 90K transformations with SE scores';
COMMENT ON TABLE cost_tracking IS 'Budget monitoring: daily cost aggregation by model';
COMMENT ON COLUMN transformations.se_score IS 'Semantic Elasticity: (delta_cc * P^2 * D) / E';
