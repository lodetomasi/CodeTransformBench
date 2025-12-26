/# CodeTransformBench: DeepMind Research Execution Protocol
## Complete Implementation Guide - Phases 1-4

**Research Lead**: Claude Code (DeepMind Methodology)  
**PhD Researcher**: Lorenzo De Tomasi  
**Duration**: 10 weeks (experimental phases only)  
**Budget**: €2,000  
**Core Constraint**: OpenRouter API only, all computation local

---

## PART 0: RESEARCH PHILOSOPHY

### The Scientific Question

**How do Large Language Models perform semantic-preserving code transformations across programming languages, and can we quantify this objectively?**

**Why This Matters**:
- Current benchmarks measure generation, not transformation
- Developers spend 60-70% of time on refactoring (LaToza & Myers 2010)
- Transformation requires semantic understanding + syntactic manipulation
- Industry needs quantitative model selection guidance

### Methodological Constraints

**OpenRouter API Only**
- Rationale: Single endpoint for 200+ models, unified billing
- Alternative rejected: Individual APIs (complexity > cost savings)
- Tradeoff: 10% markup acceptable for operational simplicity

**Local Computation**
- Rationale: Reproducibility, cost predictability, data sovereignty
- Alternative rejected: Cloud (vendor lock-in, hidden costs)
- Requirement: 16GB RAM, PostgreSQL, Python 3.10+

**No External Services**
- Rationale: Dataset permanence, version control
- Alternative rejected: HF Datasets (dependency risk)
- Solution: Self-host all data

---

## PHASE 1: INFRASTRUCTURE (Week 1)

### Objective

Build reproducible research environment with database-backed experiment tracking.

### Execution Steps

**Step 1.1: Project Structure**

Create directory hierarchy:
- `data/raw/` - Immutable source data (never modify)
- `data/processed/` - Cleaned functions
- `data/test_suites/` - Generated tests
- `src/collectors/` - Web scrapers
- `src/generators/` - Test generation
- `src/evaluators/` - Transformation pipeline
- `src/metrics/` - SE calculator
- `experiments/prompts/` - Template library
- `results/raw_api_responses/` - Cache layer

**Rationale**: Separation of concerns enables reproducibility

**Step 1.2: Dependency Installation**

Use Poetry for deterministic builds:

Install:
- openai (OpenRouter compatible)
- radon (cyclomatic complexity)
- tree-sitter + language bindings (universal parser)
- apted, zss (tree edit distance)
- sqlalchemy, psycopg2 (database)
- aiohttp, asyncio-throttle (async + rate limiting)
- pandas, numpy, scipy (analysis)
- beautifulsoup4 (web scraping)
- loguru, tqdm, rich (utilities)

**Critical Choices**:
- Tree-sitter over language-specific parsers: Unified API, faster, error-tolerant
- PostgreSQL over SQLite: Concurrent writes, JSONB support, better analytics
- Async over sync: 90K calls = 25h sequential vs 6h parallel

**Step 1.3: PostgreSQL Schema**

Create 3 tables:

**`functions`** (source corpus):
- Primary key: `id` (format: {lang}_{source}_{task}_{num})
- Unique constraint: `code_hash` (SHA256 prevents duplicates)
- Metrics: cyclomatic_complexity, halstead_volume, lines_of_code
- Indexes: language, complexity_tier, domain

**`transformations`** (results):
- Primary key: `id` (auto-increment)
- Foreign key: `function_id` CASCADE delete
- Experiment: model, strategy, task, temperature
- SE components: delta_cc, preservation, diversity, effort, se_score
- Cost: cost_usd, latency_ms, tokens
- Composite index: (model, task, se_score DESC) for leaderboard

**`cost_tracking`** (budget):
- Unique: (date, model)
- Auto-updated via trigger on INSERT to transformations

**Rationale**:
- Normalization ensures integrity
- Triggers automate cost tracking
- Indexes optimize queries (leaderboard = fast GROUP BY)

### Validation

Check:
- `psql -c "\d functions"` shows all columns + indexes
- Insert test row, verify CASCADE works
- Trigger test: INSERT transformation, cost_tracking updates
- Performance: EXPLAIN ANALYZE on leaderboard query <100ms

### Expected Output

- Fully initialized Git repo
- PostgreSQL running with 3 tables
- All dependencies pass `python -c "import tree_sitter; print('OK')"`
- README documents setup for reproduction

---

## PHASE 2: DATA COLLECTION (Week 2)

### Objective

Curate 500 functions stratified by language, complexity, domain.

**Targets**:
- Languages: Python 150, Java 150, JavaScript 100, C++ 100
- Complexity: Simple 40%, Medium 40%, Complex 20%
- Domains: Algorithms 30%, Data Structures 25%, Strings 20%, Math 15%, I/O 10%

### Rationale for Sources

**Rosetta Code (260 functions)**:
- Cross-language: Same algorithm in 4+ languages
- Community-vetted: Peer-reviewed code
- Diverse: 1200+ tasks
- Licensed: GNU FDL 1.2 (open source)

**TheAlgorithms (240 functions)**:
- Modern: Python 3.10+, Java 17, ES6
- Tested: Most have test files
- Active: 100K+ stars, recent commits

### Execution Steps

**Step 2.1: Rosetta Code Scraper**

Algorithm:
1. Fetch task list from `rosettacode.org/wiki/Category:Programming_Tasks`
2. Parse HTML (BeautifulSoup + lxml)
3. Filter tasks with ≥4 language implementations
4. For each task × language:
   - Find code block (pattern: `<span id="{Language}">` → `<pre>`)
   - Extract text, clean HTML entities
   - Validate: Compile check (Python: compile(), Java: has "class")
   - Compute metrics: CC, Halstead, LOC
   - Cache HTML locally (never re-scrape)
5. Deduplicate by SHA256(code)
6. Stratify by complexity
7. Sample 260 matching distribution

**Ethical Scraping**:
- Check robots.txt (Rosetta allows)
- Rate limit: 1 req/sec (conservative)
- User-Agent: Research project + contact
- Cache all HTML

**Expected Time**: ~2 hours

**Step 2.2: TheAlgorithms Cloner**

Algorithm:
1. Clone 4 repos: `git clone --depth 1 {repo_url}`
2. Walk tree, find `.py`, `.java`, `.js`, `.cpp` files
3. Quality filters:
   - Exclude: tests, __init__, utils
   - Size: 100-5000 bytes
   - Content: Has docstring/comments, CC ≥ 3
   - Dependencies: No non-stdlib imports
4. Extract function-level code
5. Compute metrics
6. Deduplicate against Rosetta Code
7. Sample 240

**Expected Time**: ~30 minutes

**Step 2.3: Metadata Enrichment**

For each function:
1. Cyclomatic Complexity:
   - Python: radon.cc_visit()
   - Others: AST-based proxy (count if/loops + 1)

2. Halstead Volume:
   - Python: radon.raw.analyze()
   - Others: Token count approximation

3. Code Hash: SHA256(normalized_code)
   - Normalization: strip whitespace, sort imports

4. Domain: Heuristic keyword matching

5. Save as JSONL (one function per line)

### Validation

Quality Checks:
- No duplicate code_hash
- 100% compile successfully
- Distribution matches targets (chi-square test p>0.05)
- Manual review (20% sample): Quality threshold met

Database:
```sql
INSERT INTO functions (...) VALUES (...) ON CONFLICT (code_hash) DO NOTHING;
```
Expected: 500 rows, 0 conflicts

### Expected Output

- `data/processed/all_functions.jsonl`: 500 lines, ~5MB
- Database: 500 rows in `functions` table
- Distribution report:
  - By language: 150/150/100/100 ✓
  - By complexity: 200/200/100 ✓
  - Compilation: 100% success ✓
  - Avg CC: 12-15 (reasonable)

---

## PHASE 3: TEST GENERATION (Week 3)

### Objective

Generate property-based test suites achieving 80%+ branch coverage.

### Rationale

**Why LLM-Generated**:
- Scale: 500 × 47 tests/func = 23,500 tests
- Time: Human = 10 min/func (83h) vs LLM = 2h total
- Cost: €0.02/func × 500 = €10

**Why Property-Based**:
- Thoroughness: 100+ random inputs vs 5-10 manual
- Edge cases: Hypothesis explores boundaries automatically
- Reproducibility: Fixed seed ensures determinism

**Frameworks**:
- Python: pytest + hypothesis
- Java: JUnit 5 + jqwik
- JavaScript: jest + fast-check
- C++: Google Test + RapidCheck

### Execution Steps

**Step 3.1: Prompt Templates**

Create 4 templates (one per language):

Structure:
```
Generate property-based tests using {framework}.

Function:
{code}

Requirements:
1. Edge cases: empty, None/null, negatives, boundaries
2. Properties: commutativity, associativity (if applicable)
3. Invariants: output constraints, state consistency
4. Randomized: 100+ inputs via {generator}
5. Fixed seed: {seed_instruction}
6. Self-contained: No external files/network
7. Fast: <100ms total

Output ONLY test code (no explanations).
```

Language-specific:
- Python: hypothesis.strategies, @hypothesis.seed(42)
- Java: Arbitraries.*, @PropertyDefaults(seed="42")

**Step 3.2: Generation Pipeline**

Algorithm:
```
FOR each function:
    1. Load metadata (id, code, language)
    2. Render prompt template
    3. Call OpenRouter:
       - Model: anthropic/claude-3.5-sonnet (best for code)
       - Temperature: 0.2 (deterministic but flexible)
       - Max tokens: 2000
    4. Parse response (strip markdown fences)
    5. Save to data/test_suites/{id}_test.{ext}
    6. Validate (Step 3.3)
    7. If pass: UPDATE functions SET test_suite_path, validated=TRUE
    8. If fail: Retry max 3x with error feedback
```

Parallelization: Batch 10, async calls

Cost control: Monitor cumulative, switch to cheaper model if >€0.05/func

**Step 3.3: Validation**

For each test suite:

1. **Syntactic**: Compile (pytest, javac, etc.) → Exit code 0
2. **Functional**: Run tests on original → 100% pass
3. **Coverage**: `pytest --cov` → ≥80% branch
4. **Performance**: Execution time <100ms

If <80% coverage: Generate supplementary tests for uncovered branches

Retry Strategy:
- Extract error message
- Append: "Previous failed with: {error}. Fix it."
- Re-generate (max 3 retries)
- After 3: Flag for manual review

**Step 3.4: Manual Review**

Sample 100 functions (20%):
- Tests exercise meaningful properties
- Edge cases non-trivial
- No flaky tests (run 3x, all pass)

Acceptance: 95%+ judged high quality
If <95%: Revise prompt, regenerate all

### Validation

Success Metrics:
- Coverage: ≥90% have test suites
- Quality: Median coverage ≥87%
- Speed: Median <50ms
- Cost: <€15 total

Database:
```sql
SELECT COUNT(*) FROM functions WHERE validated=TRUE;
-- Expected: ≥450
```

### Expected Output

- 450-500 test files in `data/test_suites/`
- Coverage report CSV: func_id, language, coverage, test_count, time_ms
- Failed functions log: ~10-50 need manual tests
- Cost summary: "€10.23, 468 successful"

---

## PHASE 4: TRANSFORMATION EXPERIMENTS (Week 4-10)

### Objective

Execute 90,000 transformations measuring Semantic Elasticity.

**Design**:
- Functions: 500
- Models: 12
- Tasks: 3 (obfuscate, deobfuscate, refactor)
- Strategies: 5 (zero-shot, few-shot k=3/5, CoT, self-reflection)
- Total: 500 × 12 × 3 × 5 = 90,000 calls

### Rationale for Tiered Execution

**Budget Philosophy**:

Tier 1 (60% = €1,200): Exploration
- Models: Llama 8B, Mixtral, CodeGemma, DeepSeek-Coder
- Cost: ~€0.0003/call
- Purpose: Identify promising (task, strategy) pairs

Tier 2 (30% = €600): Validation
- Models: Llama 70B, DeepSeek-V3, Qwen 32B
- Cost: ~€0.002/call
- Purpose: Confirm Tier 1 findings

Tier 3 (10% = €200): SOTA
- Models: GPT-4, Claude 3.5, Gemini 1.5
- Cost: ~€0.02/call
- Purpose: Best scores for final leaderboard

**Why Tiers Work**:
- Pareto: 80% insights from 20% cost
- Early stopping: Skip expensive if unpromising
- Focused spending: Use GPT-4 only where matters

### Execution Steps

**Step 4.1: Prompt Library**

Create 15 templates (3 tasks × 5 strategies):

Examples:

1. **obfuscate_zero_shot.txt**:
```
Obfuscate this {language} function.
Preserve functionality, maximize change.
Code: {code}
Output only transformed code.
```

2. **deobfuscate_chain_of_thought.txt**:
```
Deobfuscate step-by-step:
1. Identify obfuscation patterns
2. Reverse transformations
3. Improve readability
4. Verify correctness

Code: {code}
```

3. **refactor_few_shot_k5.txt**:
```
5 refactoring examples:
[Example 1-5 with before/after]

Now refactor: {code}
```

Few-shot examples:
- Source: Manually select 15 from Rosetta Code
- Criteria: Clear improvement, verified correctness
- Storage: experiments/prompts/examples/

**Step 4.2: Experiment Configuration**

YAML config `experiments/configs/full_run.yaml`:

```yaml
experiment_id: full_run_2025_01

models:
  tier1: [llama-8b, mixtral-8x7b, ...]
  tier2: [llama-70b, deepseek-v3, ...]
  tier3: [gpt-4, claude-3.5, ...]

tasks: [obfuscate, deobfuscate, refactor]
strategies: [zero_shot, few_shot_k3, few_shot_k5, cot, self_reflect]

sampling:
  tier1: 500  # All functions
  tier2: 200  # Sample based on Tier 1
  tier3: 50   # Top performers

rate_limiting:
  max_rpm: 60
  max_concurrent: 10
  retry_max: 3

budget:
  total_usd: 2000
  daily_limit: 50
```

**Step 4.3: Execution Pipeline**

**TIER 1 (Weeks 4-6)**

Algorithm:
```
FOR each Tier 1 model:
    FOR all 500 functions:
        FOR each task:
            FOR each strategy:
                1. Load function + prompt template
                2. Render prompt (insert code, examples if few-shot)
                3. Call OpenRouter (rate-limited, async)
                4. Parse response (extract code)
                5. Run tests on transformed code
                6. Compute SE:
                   - delta_cc = |CC(orig) - CC(trans)|
                   - preservation = 1 if tests pass else 0
                   - diversity = tree_edit_distance / max_size
                   - effort = 1 / (1 + halstead_volume/1000)
                   - se_score = (delta_cc * P² * D) / E
                7. INSERT INTO transformations (...)
                8. Update cost_tracking
                9. If budget exceeded: STOP

ANALYSIS: Aggregate by (model, task, strategy)
DECISION: Top 5 combinations by avg SE → feed to Tier 2
```

**TIER 2 (Weeks 7-8)**

Same pipeline but:
- Only top 5 (task, strategy) combinations
- Sample 200 functions (stratified by complexity)
- Confirm Tier 1 patterns hold

**TIER 3 (Weeks 9-10)**

Same but:
- Best 2 combinations only
- 50 hardest functions (CC>20)
- Establish SOTA scores

**Parallelization**:
- asyncio: 10 concurrent requests
- Throttle: 60 req/min (OpenRouter limit)
- Batching: 100 functions, wait for all, next batch

**Error Handling**:

1. **Rate Limit (429)**: Exponential backoff (2^retry seconds), max 3
2. **Timeout (>120s)**: Log, skip
3. **Parse Fail**: Regex extraction, else error_type='parse_error'
4. **Test Fail**: Valid result (preservation=0, se_score=NULL)

**Caching**:

Before API call:
```sql
SELECT * FROM transformations 
WHERE function_id=%s AND model=%s AND task=%s AND strategy=%s;
```
If exists: Use cache (saves cost + time)

**Progress Monitoring**:

Terminal dashboard (rich library):
```
╔══════════════════════════════════════════╗
║   CodeTransformBench - Tier 1/3          ║
║   Model: llama-8b                        ║
║   Progress: 1,247/6,000 (20.78%)         ║
║   Success: 87.3%                         ║
║   Avg SE: 6.42 (σ=2.18)                  ║
║   Cost: $48/$1,200 (4%)                  ║
║   Rate: 42 req/min                       ║
║   ETA: 2h 15m                            ║
╚══════════════════════════════════════════╝
```

### Validation

After each tier:

1. **Coverage**: All planned experiments run?
2. **Success Rate**: 60-80% preservation (if >90% too easy, <50% too hard)
3. **Cost Accuracy**: Within budget ±10%
4. **SE Distribution**: Q1≈4, Median≈7, Q3≈12

Anomaly Detection:
- SE>50: Metric bug
- Cost>$0.10/call: Runaway tokens
- Latency>60s: Model stuck
- All tests fail: Test suite broken

### Expected Output

1. **Database**:
   - transformations: 90,000 rows
   - cost_tracking: ~360 rows

2. **Raw Data**: 
   - tier1_full.jsonl (~500MB)
   - tier2_full.jsonl (~200MB)
   - tier3_full.jsonl (~20MB)

3. **Summary CSVs**:
   - By model: model, task, strategy, count, success_rate, mean_se, median_se
   - By language
   - By complexity

4. **Cost Report**:
   - Total: $1,847 / $2,000 budget
   - Breakdown: T1=$1,104, T2=$556, T3=$187

---

## FINAL DELIVERABLES (Week 10)

### Database State

Complete PostgreSQL with:
- functions: 500 validated rows
- transformations: 90,000 experimental results
- cost_tracking: Real-time budget status

All queryable for analysis.

### Validated Dataset

- 500 functions across 4 languages
- 450+ test suites (80%+ coverage)
- 100% compilable
- Stratification verified

### Experimental Results

- Raw API responses cached
- SE scores computed
- Cost tracking detailed
- Error logs comprehensive

### Reproducible Infrastructure

- Git repo with all code
- Dependencies pinned (poetry.lock)
- Database schema documented
- YAML configs enable re-runs

---

## SUCCESS CRITERIA

### Phase 1 (Week 1)
✅ PostgreSQL with tables/indexes
✅ Dependencies installed
✅ Test suite passes

### Phase 2 (Week 2)
✅ 500 functions in database
✅ Distribution: 150/150/100/100
✅ Zero duplicates
✅ 100% compilation
✅ Manual review: quality threshold met

### Phase 3 (Week 3)
✅ ≥450 test suites
✅ Median coverage ≥87%
✅ <100ms execution
✅ <€15 cost

### Phase 4 (Weeks 4-10)

**Tier 1** (Week 6):
✅ 6,000+ transformations
✅ 60-85% success rate
✅ $600-750 spent
✅ Top 5 strategies identified

**Tier 2** (Week 8):
✅ 2,400+ transformations
✅ Tier 1 validated
✅ $500-650 spent

**Tier 3** (Week 10):
✅ 600+ transformations
✅ SOTA scores
✅ $150-200 spent
✅ Total <$2,000

---

## MINIMUM vs TARGET vs STRETCH

### Minimum Viable (Must Achieve)

By Week 10:
- ≥450 validated functions (90%)
- ≥40,000 transformations (45%)
- All 12 models evaluated
- 50-90% preservation rate
- SE scores reasonable
- Budget ≤$2,000
- Reproducible setup

### Target (Ideal)

- All 500 functions
- Full 90,000 transformations
- All tiers complete
- Clear winner models identified
- Statistical significance (p<0.05)
- Spent $1,700-1,900 (under budget)

### Stretch (If Time/Budget)

- +100 functions (600 total)
- +5th language (Rust/Go)
- +3 models
- ML model for SE prediction
- Human validation study (N=40)
- Docker container

---

## CRITICAL DECISIONS

### Week 2: Quality vs Quantity
If 400 meet quality → ACCEPT (quality > quantity)
If 500 but 30% fail → REJECT, raise bar

### Week 6: Tier 1 Results
Success >85% → Tasks too easy, harder prompts
Success <50% → Prompts too hard, simplify
Budget overspend >20% → Cut Tier 3

### Week 8: Continue Tier 3?
Tier 2 confirms Tier 1 → Proceed
Contradictory → Debug, re-run
Budget depleted → Stop, analyze

### Week 10: Extend or Finish?
Target hit → Finish
<40K transforms → Extend 2 weeks OR accept partial

---

## DAILY WORKFLOW

**Morning (09:00-12:00)**:
- Check overnight runs
- Review error logs
- Adjust configs if needed

**Afternoon (13:00-17:00)**:
- Launch new batches
- Manual validation samples
- Update documentation

**Evening (Optional)**:
- Monitor dashboards
- Calculate ETA
- Plan next day

---

## KEY FILES TO MAINTAIN

1. **CHANGELOG.md**: All decisions
   - "Week 3: Switched Claude for test gen (better quality)"

2. **EXPERIMENTS.md**: Run tracking
   - Tier 1 Run 1: 500 funcs, Llama 8B, zero-shot → 72% success, SE=5.2

3. **ERRORS.md**: Failure logging
   - Function py_001, GPT-4, timeout → Retry OK

---

## BACKUP STRATEGY

**Daily**:
```bash
pg_dump codetransform > backups/db_$(date +%Y%m%d).sql
gzip backups/db_*.sql
```
Keep last 7 days.

**Checkpoints**:
- After each tier: Full export (JSONL + SQLite)
- Before major changes: Snapshot
- Before deletions: Test on copy

---

## TROUBLESHOOTING

**Disk full**: Delete logs, compress cache
**Rate limit**: Reduce to 50 req/min, add jitter
**Invalid test code**: Retry temp=0.1 or different model
**SE all zero**: Check preservation (tests broken?), verify tree edit
**Budget exceeded**: Stop, analyze, eliminate expensive models

---

## CONCLUSION

This protocol provides complete roadmap for Weeks 1-10. Key principles:

1. **Scientific Rigor**: Every decision justified
2. **Cost Discipline**: Tiered prevents overruns
3. **Quality > Quantity**: 450 good > 500 mediocre
4. **Reproducibility**: Everything documented, cached
5. **Pragmatism**: Accept imperfection, document, forward

By Week 10, you have complete dataset answering: "How do LLMs transform code?"

Analysis and publication can follow, but empirical work will be done.

**Final Reminder**: This is research. Expect 20-30% failures. Document, learn, adapt. Goal is insight, not perfection.

Good luck!