# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CodeTransformBench** is a research benchmark to evaluate Large Language Model code transformation capabilities across 4 programming languages (Python, Java, JavaScript, C++). The project measures how well LLMs perform semantic-preserving code transformations using a custom metric called **Semantic Elasticity (SE)**.

**Scale**: 500 functions × 12 models × 3 tasks × 5 strategies = 90,000 transformations
**Budget**: €2,000 via OpenRouter API
**Database**: PostgreSQL with SQLAlchemy ORM

## Critical Constraints - READ FIRST

### Research Integrity

1. **Apply extreme scientific rigor** - This is PhD research. Every decision must be justified, every result must be verifiable, every experiment must be reproducible
2. **NEVER fabricate or invent experimental results** - If data doesn't exist, say so. Don't create fake SE scores, costs, or statistics
3. **NEVER modify raw data** - Files in `data/raw/` are immutable. Only read, never edit
4. **NEVER run experiments without explicit approval** - Each API call costs money from a limited budget
5. **NEVER delete experimental data** - Transformations, functions, logs are permanent research artifacts

### File Management

1. **AVOID creating new files unless absolutely necessary** - Prefer editing existing files
2. **NEVER create documentation files unprompted** - No README updates, CHANGELOG entries, or markdown docs unless explicitly requested
3. **NO placeholder/example files** - Don't create `example.py`, `test_example.py`, or similar
4. **Check if files exist first** - Use Read/Glob before assuming you need to create something

### Code Changes

1. **ASK before architectural changes** - Database schema, config structure, core patterns require discussion
2. **Verify assumptions by reading code** - Don't assume how something works, check the actual implementation
3. **Prefer minimal changes** - Fix the specific issue, don't refactor surrounding code
4. **NO premature abstractions** - Don't create utils/helpers for one-time operations

### Database Constraints

1. **NEVER drop or truncate tables in production** - Use explicit filters for deletes
2. **Test queries before execution** - Use SELECT before UPDATE/DELETE on production database
3. **Respect unique constraints** - Check for existing data before inserts (code_hash, experiment combinations)

### Git Operations

1. **ASK before committing** - This is PhD research, commits should be intentional
2. **NO force pushes** - Ever
3. **NO automatic gitignore updates** - Ask first

This is a **research project with real budget constraints and academic integrity requirements**. When in doubt, ask the user.

## Essential Commands

### Database Operations
```bash
# Initialize database (run once)
python database/init_db.py

# Validate configuration
python src/config.py

# Test database utilities
python src/utils/db_utils.py

# Query database directly
psql codetransform
```

### Development Workflow
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test OpenRouter API client
python src/api/openrouter_client.py

# Run individual collectors/generators
python src/collectors/rosetta_scraper.py
python src/generators/test_generator.py
```

### Data Collection
```bash
# Collect all data (automated script)
./collect_all_data.sh

# Test collection pipeline
./test_collection.sh
```

## Architecture

### Database-First Design

The project uses PostgreSQL as the single source of truth. All experimental data flows through three core tables:

1. **`functions`** - Source code corpus (500 functions)
   - Unique constraint on `code_hash` (SHA256) prevents duplicates
   - Indexed by: language, cyclomatic_complexity, domain, validated
   - Related to transformations via CASCADE delete

2. **`transformations`** - Experiment results (90K transformations)
   - Unique constraint: (function_id, model, task, strategy) prevents duplicate experiments
   - Stores SE metric components: delta_cc, preservation, diversity, effort, se_score
   - Automatically updates `cost_tracking` via PostgreSQL trigger
   - Indexed for fast leaderboard queries: (model, task, se_score DESC)

3. **`cost_tracking`** - Budget monitoring by (date, model)
   - Auto-updated via trigger on INSERT to transformations
   - Tracks: total_calls, total_cost_usd, avg_latency_ms, success/failure rates

**View**: `leaderboard` - Pre-computed aggregation of transformation stats by (model, task, strategy)

### Code Organization

```
src/
├── config.py              # Central configuration (loads .env + YAML)
├── collectors/            # Data collection from web sources
│   ├── rosetta_scraper.py      # Rosetta Code scraper
│   ├── algorithms_cloner.py    # TheAlgorithms repo cloner
│   ├── metrics_calculator.py   # CC, Halstead, LOC calculation
│   └── insert_functions.py     # Batch insert to database
├── generators/            # Test suite generation via LLMs
│   ├── test_generator.py       # Property-based test creation
│   └── test_validator.py       # Coverage + correctness validation
├── evaluators/            # Transformation experiments
│   ├── transformation_pipeline.py
│   └── semantic_elasticity.py  # SE metric calculator
├── api/                   # OpenRouter API client
│   └── openrouter_client.py    # Async client with rate limiting
└── utils/
    ├── db_utils.py        # Database connection + common queries
    ├── logger.py          # Loguru setup
    └── progress.py        # Progress tracking utilities
```

### Configuration System

**Environment Variables** (`.env`):
- `OPENROUTER_API_KEY` - Required for API access
- `DATABASE_URL` - PostgreSQL connection (default: `postgresql://localhost/codetransform`)
- Budget allocation: `BUDGET_TOTAL_USD`, `BUDGET_TIER1_PCT`, `BUDGET_TIER2_PCT`, `BUDGET_TIER3_PCT`
- Rate limits: `MAX_REQUESTS_PER_MINUTE`, `MAX_CONCURRENT_REQUESTS`

**YAML Configs** (`config/`):
- `models.yaml` - 12 LLM models organized by tier (tier1/tier2/tier3)
- `experiments.yaml` - Experiment configurations (planned)

**Model Tier System**:
- **Tier 1** (60% budget): Cheap models (Llama 8B, Mixtral, CodeGemma, DeepSeek Coder) - All 500 functions
- **Tier 2** (30% budget): Mid-tier (Llama 70B, DeepSeek V3, Qwen) - 200 sampled functions
- **Tier 3** (10% budget): SOTA (GPT-4, Claude 3.5, Gemini 1.5) - 50 hardest functions

### Semantic Elasticity (SE) Metric

**Formula**: `SE = (ΔCC × P² × D) / E`

Where:
- **ΔCC**: Absolute difference in cyclomatic complexity between original and transformed code
- **P**: Preservation (1 if tests pass, 0 if fail) - squared to heavily penalize failures
- **D**: Diversity (tree edit distance normalized by max tree size)
- **E**: Effort (inverse of Halstead volume: `1 / (1 + halstead_volume/1000)`)

**Key Properties**:
- Higher SE = better transformation (more change while preserving semantics)
- P=0 → SE=0 (any test failure results in zero score)
- Computed and stored in `transformations.se_score`

## Key Patterns and Conventions

### Database Sessions

Always use context managers for database access:

```python
from src.utils.db_utils import get_db_session

with get_db_session() as session:
    functions = session.query(Function).filter(...).all()
    # session.commit() called automatically on success
    # session.rollback() called on exception
```

### Async API Calls

The OpenRouter client uses async/await with built-in rate limiting:

```python
from src.api.openrouter_client import OpenRouterClient

client = OpenRouterClient()  # Uses config.OPENROUTER_API_KEY

# Single request
result = await client.generate(
    model='anthropic/claude-3.5-sonnet',
    prompt='...',
    temperature=0.2,
    max_tokens=2000
)

# Batch requests (respects MAX_CONCURRENT_REQUESTS)
requests = [{'model': '...', 'prompt': '...'}, ...]
results = await client.generate_batch(requests)
```

**Rate Limiting**: Automatically throttled via `asyncio-throttle` (default: 60 req/min)
**Retries**: Exponential backoff on 429 (rate limit) and timeouts (max 3 attempts)
**Cost Tracking**: All requests tracked in `client.total_cost` and database trigger

### Configuration Access

```python
from src.config import config, get_model_by_id, get_models_by_tier

# Environment variables
config.DATABASE_URL
config.BUDGET_TIER1_USD  # Computed property

# Model configs
model_info = get_model_by_id('anthropic/claude-3.5-sonnet')
tier1_models = get_models_by_tier('tier1')  # Returns list with 'tier' added
```

### Code Metrics Calculation

```python
from src.collectors.metrics_calculator import calculate_metrics

metrics = calculate_metrics(code, language='python')
# Returns: {
#     'cyclomatic_complexity': int,
#     'halstead_volume': float,
#     'lines_of_code': int,
#     'code_hash': str  # SHA256
# }
```

**Implementation Notes**:
- Python: Uses `radon` library for accurate CC and Halstead
- Other languages: AST-based proxy (tree-sitter parsing)
- Code hash normalized: strips whitespace, sorts imports

### Data Collection Patterns

1. **Web Scraping** (Rosetta Code):
   - Cache all HTML locally in `data/raw/rosetta_html/` (never re-scrape)
   - Respect robots.txt, rate limit to 1 req/sec
   - User-Agent: Identify as research project

2. **Deduplication**:
   - Database enforces UNIQUE constraint on `code_hash`
   - Use `ON CONFLICT (code_hash) DO NOTHING` when inserting

3. **Stratification**:
   - Target distribution: Python 150, Java 150, JavaScript 100, C++ 100
   - Complexity tiers: Simple (CC≤10) 40%, Medium (CC 11-30) 40%, Complex (CC>30) 20%
   - Sample using stratified random sampling to match distribution

### Budget Management

```python
from src.utils.db_utils import (
    get_remaining_budget,
    is_budget_exhausted,
    get_total_cost_by_tier
)

# Check before expensive operations
if is_budget_exhausted('tier3', threshold=0.95):
    logger.warning("Tier 3 budget 95% exhausted!")

remaining = get_remaining_budget('tier1')
print(f"Tier 1 remaining: ${remaining:.2f}")

spent = get_total_cost_by_tier('tier2')
```

**Critical**: Budget tracking happens automatically via PostgreSQL trigger. Always populate `cost_usd` in transformations table.

## Transformation Tasks and Strategies

### Tasks
- **obfuscate**: Make code harder to understand while preserving functionality
- **deobfuscate**: Clarify obfuscated code
- **refactor**: Improve code structure/readability

### Strategies
- **zero_shot**: Direct instruction
- **few_shot_k3**: 3 examples in prompt
- **few_shot_k5**: 5 examples in prompt
- **chain_of_thought**: Step-by-step reasoning
- **self_reflection**: Model critiques its own output

Prompt templates stored in `experiments/prompts/` with naming: `{task}_{strategy}.txt`

## Testing and Validation

### Test Suite Requirements
- **Framework**: pytest+hypothesis (Python), JUnit+jqwik (Java), jest+fast-check (JS), GTest+RapidCheck (C++)
- **Target Coverage**: ≥80% branch coverage
- **Performance**: <100ms execution time
- **Determinism**: Fixed seed for reproducibility

### Validation Pipeline
1. Syntactic: Code compiles
2. Functional: Tests pass on original code
3. Coverage: `pytest --cov` ≥80%
4. Speed: Execution time acceptable

Functions with validated test suites: `functions.validated = TRUE`

## Development Guidelines

### Adding New Models
1. Add to `config/models.yaml` under appropriate tier
2. Include: id, name, cost_per_1k_tokens, description
3. Verify with `python src/config.py`

### Running Experiments
1. Check budget: `get_remaining_budget(tier)`
2. Check for cached results: `check_transformation_exists(...)`
3. Execute transformation via OpenRouter client
4. Calculate SE metric components
5. Insert to `transformations` table (trigger updates `cost_tracking`)

### Querying Results
```python
from src.utils.db_utils import get_leaderboard

# Top models for a task
leaderboard = get_leaderboard(task='obfuscate', limit=10)
for entry in leaderboard:
    print(f"{entry['model']}: SE={entry['mean_se']:.2f}, Success={entry['success_rate_pct']:.1f}%")
```

## Common Pitfalls

1. **Database Sessions**: Never pass Session objects between functions - use context managers
2. **Async Contexts**: Always `await` OpenRouter client methods; use `asyncio.run()` for top-level
3. **Budget Overruns**: Check `is_budget_exhausted()` before starting expensive tier operations
4. **Duplicate Experiments**: Unique constraint on (function_id, model, task, strategy) will raise error
5. **Code Hash Collisions**: UNIQUE constraint on `code_hash` prevents duplicates during insertion
6. **Tree Edit Distance**: Requires parsed ASTs (use tree-sitter); expensive for large functions
7. **Rate Limits**: OpenRouter enforces limits; client handles this but watch for 429 errors in logs

## Research Phases

**Phase 1** (Week 1): Infrastructure setup - Database, ORM, config ✅ COMPLETED
**Phase 2** (Week 2): Data collection - 500 functions from Rosetta Code + TheAlgorithms
**Phase 3** (Week 3): Test generation - LLM-generated property-based tests
**Phase 4** (Weeks 4-10): Transformation experiments - Execute 90K transformations across tiers

See [README.md](README.md) and [requirements.md](requirements.md) for detailed research protocol.

## Critical Files

- [database/schema.sql](database/schema.sql) - Complete PostgreSQL schema with triggers
- [database/models.py](database/models.py) - SQLAlchemy ORM models with relationships
- [src/config.py](src/config.py) - Configuration loader and validation
- [config/models.yaml](config/models.yaml) - All 12 model definitions
- [src/utils/db_utils.py](src/utils/db_utils.py) - Database utilities and queries
- [src/api/openrouter_client.py](src/api/openrouter_client.py) - Async API client with rate limiting
