# CodeTransformBench

A reproducible research benchmark for evaluating Large Language Model capabilities in semantic-preserving code transformations across four programming languages.

## Overview

**Objective**: Quantify LLM performance on code transformation tasks using a novel Semantic Elasticity metric that captures both structural change and semantic preservation.

**Scale**: 240 functions × 7 state-of-the-art models × 3 transformation tasks × 3 intensity levels

**Current Progress**: Phase 4 active - Executing systematic transformations with real-time checkpoint recovery and automated quality validation

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- 16GB RAM
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

**1. Install PostgreSQL**

```bash
# macOS
brew install postgresql@15
brew services start postgresql@15

# Create database
createdb codetransform
```

**2. Clone and setup project**

```bash
cd "Adversarial Prompt Engineering for Obfuscation"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**3. Configure environment**

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenRouter API key
nano .env  # or use your preferred editor
```

Add your API key to `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

**4. Initialize database**

```bash
python database/init_db.py
```

Expected output:
```
✓ Connected to database
✓ Executed schema from database/schema.sql
✓ All tables created: {'functions', 'transformations', 'cost_tracking'}
✓ Database initialization SUCCESSFUL
```

**5. Test configuration**

```bash
python src/config.py
```

You should see:
```
✓ Configuration valid
Database URL: postgresql://localhost/codetransform
Budget: $2000.0
  Tier 1: $1200.0
  Tier 2: $600.0
  Tier 3: $200.0
```

---

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── .gitignore               # Git ignore rules
├── README.md                # This file
│
├── config/
│   ├── models.yaml          # 12 LLM model configurations
│   └── experiments.yaml     # Experiment configurations (TODO)
│
├── database/
│   ├── schema.sql           # PostgreSQL schema
│   ├── init_db.py          # Database initialization script
│   └── models.py           # SQLAlchemy ORM models
│
├── src/
│   ├── config.py           # Configuration loader
│   ├── collectors/         # Data collection scripts (TODO)
│   │   ├── rosetta_scraper.py
│   │   ├── algorithms_cloner.py
│   │   └── metrics_calculator.py
│   ├── generators/         # Test suite generation (TODO)
│   │   ├── test_generator.py
│   │   └── test_validator.py
│   ├── evaluators/         # Transformation experiments (TODO)
│   │   ├── transformation_pipeline.py
│   │   ├── semantic_elasticity.py
│   │   └── tree_edit_distance.py
│   ├── api/               # OpenRouter API client (TODO)
│   │   ├── openrouter_client.py
│   │   └── cost_tracker.py
│   └── utils/
│       ├── db_utils.py     # Database helper functions
│       ├── logger.py       # Logging setup (TODO)
│       └── progress.py     # Progress monitoring (TODO)
│
├── data/
│   ├── raw/               # Immutable source data
│   ├── processed/         # Cleaned functions
│   └── test_suites/       # Generated test files
│
├── experiments/
│   ├── prompts/           # Prompt templates (TODO)
│   └── configs/           # Experiment configs (TODO)
│
└── results/
    ├── raw_api_responses/ # Cached API responses
    └── analysis/          # Analysis scripts (TODO)
```

---

## Research Methodology

### Phase 1: Infrastructure (COMPLETED)

Established reproducible research environment with:

- PostgreSQL database with ACID guarantees and automatic triggers
- SQLAlchemy ORM for type-safe data access
- Comprehensive logging and error recovery mechanisms
- Configuration management for model parameters and API access

### Phase 2: Data Collection (COMPLETED)

Curated 240 functions from established code repositories:

- **Sources**: Rosetta Code (algorithmic implementations), TheAlgorithms (canonical solutions)
- **Languages**: Python, Java, JavaScript, C++
- **Complexity distribution**: Stratified by cyclomatic complexity (simple/medium/complex)
- **Quality control**: All functions validated for syntactic correctness and metric calculability

### Phase 3: Test Suite Generation (COMPLETED)

Generated property-based test suites using LLM-assisted methodology:

- **Coverage target**: ≥80% branch coverage
- **Frameworks**: pytest+hypothesis (Python), JUnit+jqwik (Java), Jest+fast-check (JS), GTest+RapidCheck (C++)
- **Validation**: Automated execution verification, deterministic seeding (reproducibility)
- **Output**: 240 test suites with documented edge cases and invariant properties

### Phase 4: Transformation Experiments (IN PROGRESS)

Systematic evaluation of 7 contemporary language models across transformation tasks.

**Model Selection**:

- **Tier 1** (High-throughput): Grok Code Fast 1, Gemini 2.5 Flash, DeepSeek V3.2
- **Tier 2** (Balanced): Claude Sonnet 4.5, GPT-5.2
- **Tier 3** (State-of-the-art): Claude Opus 4.5, Gemini 2.5 Pro

**Experimental Protocol**:

- **Tasks**: Obfuscation (reduce readability), deobfuscation (improve clarity), refactoring (restructure while maintaining semantics)
- **Intensity levels**: Light (±20% LOC, ±10% CC), medium (±50% LOC, 10-30% CC), heavy (max 2× LOC, 30-50%+ CC)
- **Prompting strategy**: Zero-shot with strict output format constraints (executable code only, no explanations)
- **Quality control**: Automated syntax validation, metric calculation, test execution

**Robustness Features**:

- Real-time checkpoint/resume (database-backed, zero data loss)
- Automatic retry with exponential backoff (rate limits, transient failures)
- Fatal error detection (API quota exhaustion, authentication failures)
- Parallel execution with concurrency control (5 simultaneous requests)

**Current Status** (as of last checkpoint):

- Transformation pipeline: ~2,000+ transformations completed
- SE metric calculation: Pending completion of transformation phase
- Statistical analysis: Leaderboard generation, hypothesis testing, visualization (pending)

---

## Semantic Elasticity Metric

**Formula**: `SE = (ΔCC × P² × D) / E`

Where:
- **ΔCC**: Absolute difference in cyclomatic complexity
- **P**: Preservation (1 if tests pass, 0 if fail)
- **D**: Diversity (tree edit distance normalized)
- **E**: Effort (inverse of Halstead volume)

**Interpretation**:
- Higher SE = better transformation (more change while preserving semantics)
- P=0 → SE=0 (failed transformations score zero)

---

## Database Schema

### Tables

**`functions`**: Source code corpus
- 500 functions across 4 languages
- Metrics: cyclomatic complexity, Halstead volume, LOC
- Test suite validation status

**`transformations`**: Experiment results
- 90K transformations with SE scores
- API metadata: cost, latency, tokens
- Error tracking

**`cost_tracking`**: Budget monitoring
- Daily cost aggregation by model
- Auto-updated via PostgreSQL trigger
- Success/failure rates

### Views

**`leaderboard`**: Pre-computed rankings
- Mean/median SE by model, task, strategy
- Success rates and costs
- Used for final analysis

---

## Configuration

### Environment Variables (`.env`)

```bash
# API Configuration
OPENROUTER_API_KEY=sk-or-v1-...

# Database
DATABASE_URL=postgresql://localhost/codetransform

# Rate Limiting (API throttling)
MAX_REQUESTS_PER_MINUTE=60
MAX_CONCURRENT_REQUESTS=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=codetransform.log
```

### Model Configuration (see `config/models.yaml`)

**Tier 1** (High-throughput evaluation):
- x-ai/grok-code-fast-1 - Specialized code model with rapid inference
- google/gemini-2.5-flash - Lightweight variant optimized for speed
- deepseek/deepseek-chat - Open-weights model with strong code capabilities

**Tier 2** (Balanced performance):
- anthropic/claude-sonnet-4.5 - Latest Anthropic mid-tier model
- openai/gpt-5.2 - Contemporary OpenAI flagship

**Tier 3** (State-of-the-art):
- anthropic/claude-opus-4.5 - Highest-quality Anthropic model (SWE-bench: 80.9%)
- google/gemini-2.5-pro - Advanced Google model with strong reasoning

---

## Usage Examples

### Monitor experiment progress

```bash
# Real-time monitoring (auto-refresh every 30s)
watch -n 30 ./monitor.sh

# Single snapshot
./monitor.sh
```

### Query benchmark results

```python
from src.utils.db_utils import get_leaderboard

# Get top performing models for obfuscation task
leaderboard = get_leaderboard(task='obfuscate', limit=10)

for entry in leaderboard:
    print(f"{entry['model']}: SE={entry['mean_se']:.2f}, Success={entry['success_rate_pct']:.1f}%")
```

### Calculate Semantic Elasticity metrics

```python
from src.evaluators.semantic_elasticity import SECalculator

calculator = SECalculator()

# Calculate SE for all transformations
stats = calculator.calculate_se_for_all_transformations()
print(f"Processed: {stats['success']} transformations")
print(f"Failed: {stats['failed']} transformations")
```

### Database introspection

```python
from src.utils.db_utils import get_dataset_stats

stats = get_dataset_stats()
print(f"Functions: {stats['total_functions']}")
print(f"Transformations: {stats['total_transformations']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
```

---

## Research Questions

This benchmark investigates 5 core questions about LLM code transformation capabilities:

1. **RQ1**: Which contemporary models achieve highest semantic elasticity?
   - Hypothesis: Claude Opus 4.5 and Gemini 2.5 Pro outperform all models across tasks

2. **RQ2**: How does transformation intensity affect semantic preservation?
   - Hypothesis: Heavy transformations (50%+ structural change) reduce preservation by 30-40% compared to light transformations

3. **RQ3**: Are some programming languages more amenable to transformation?
   - Hypothesis: Python achieves 40-60% higher SE than C++ due to dynamic typing and simpler syntax

4. **RQ4**: Does cyclomatic complexity limit transformation quality?
   - Hypothesis: SE degrades sharply for functions with CC > 25 (threshold effect)

5. **RQ5**: How do model capabilities scale with transformation difficulty?
   - Hypothesis: Performance gap between SOTA (Claude Opus) and high-throughput models (Grok, Gemini Flash) widens on heavy transformations

Statistical analysis: Paired t-tests, ANOVA across model tiers, correlation analysis (CC vs SE), stratified sampling validation.

---

## Contributing

This is a PhD research project. If you'd like to contribute:

1. Read [requirements.md](requirements.md) for full context
2. Check the TODO comments in code
3. Follow the existing code style
4. Write tests for new functionality
5. Update documentation

---

## License

Research project - license TBD

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{codetransformbench2025,
  author = {De Tomasi, Lorenzo},
  title = {CodeTransformBench: Evaluating LLM Code Transformation Capabilities},
  year = {2025},
  note = {Research benchmark for semantic-preserving code transformations}
}
```

---

## Acknowledgments

- Research methodology inspired by DeepMind protocols
- Semantic Elasticity metric adapted from [original paper]
- Data sources: Rosetta Code, TheAlgorithms
- API access: OpenRouter

---

## Contact

**PhD Researcher**: Lorenzo De Tomasi
**Project**: Adversarial Prompt Engineering for Obfuscation
**Duration**: 10 weeks (experimental phases)

For questions or collaboration: [contact info]

---

**Status**: Phases 1-3 Complete | Phase 4 Active (Transformation pipeline executing)
