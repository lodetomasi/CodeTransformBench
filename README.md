# CodeTransformBench

A research benchmark to evaluate Large Language Model code transformation capabilities across 4 programming languages (Python, Java, JavaScript, C++).

**Research Questions**: How do LLMs perform semantic-preserving code transformations? Can we quantify transformation quality objectively?

**Scale**: 500 functions × 12 models × 3 tasks × 5 strategies = 90,000 transformations

**Budget**: €2,000 via OpenRouter API

**Duration**: 10 weeks (experimental phases)

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

## Research Phases

### Phase 1: Infrastructure (Week 1) ✅ COMPLETED

Build reproducible research environment with database-backed experiment tracking.

**Status**: Database schema created, ORM models defined, configuration system ready.

### Phase 2: Data Collection (Week 2) 🔜 NEXT

Curate 500 functions from:
- Rosetta Code (260 functions)
- TheAlgorithms (240 functions)

Stratified by:
- **Languages**: Python 150, Java 150, JavaScript 100, C++ 100
- **Complexity**: Simple 40%, Medium 40%, Complex 20%
- **Domains**: Algorithms, Data Structures, Strings, Math, I/O

**Next steps**:
1. Implement `src/collectors/rosetta_scraper.py`
2. Implement `src/collectors/algorithms_cloner.py`
3. Run data collection
4. Validate distribution

### Phase 3: Test Generation (Week 3)

Generate property-based test suites achieving 80%+ branch coverage using LLMs.

**Frameworks**:
- Python: pytest + hypothesis
- Java: JUnit + jqwik
- JavaScript: jest + fast-check
- C++: Google Test + RapidCheck

**Budget**: ~€10-15

### Phase 4: Transformation Experiments (Weeks 4-10)

Execute 90,000 transformations measuring Semantic Elasticity (SE).

**Tiered execution**:
- **Tier 1** (60% budget): 4 cheap models, all 500 functions
- **Tier 2** (30% budget): 3 mid-tier models, 200 sampled functions
- **Tier 3** (10% budget): 5 SOTA models, 50 hardest functions

**Tasks**: obfuscate, deobfuscate, refactor
**Strategies**: zero-shot, few-shot (k=3,5), chain-of-thought, self-reflection

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
# OpenRouter API
OPENROUTER_API_KEY=sk-or-v1-...

# Database
DATABASE_URL=postgresql://localhost/codetransform

# Budget (USD)
BUDGET_TOTAL_USD=2000
BUDGET_TIER1_PCT=60
BUDGET_TIER2_PCT=30
BUDGET_TIER3_PCT=10

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60
MAX_CONCURRENT_REQUESTS=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=codetransform.log
```

### Model Tiers (see `config/models.yaml`)

**Tier 1** (Exploration): Llama 8B, Mixtral, CodeGemma, DeepSeek Coder
**Tier 2** (Validation): Llama 70B, DeepSeek V3, Qwen Coder
**Tier 3** (SOTA): GPT-4, Claude 3.5, Claude 3 Opus, Gemini 1.5 Pro, Grok 2

---

## Usage Examples

### Check database stats

```python
from src.utils.db_utils import get_dataset_stats

stats = get_dataset_stats()
print(f"Functions: {stats['total_functions']}")
print(f"Transformations: {stats['total_transformations']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Total cost: ${stats['total_cost_usd']:.2f}")
```

### Query leaderboard

```python
from src.utils.db_utils import get_leaderboard

# Get top 10 models for obfuscation task
leaderboard = get_leaderboard(task='obfuscate', limit=10)

for entry in leaderboard:
    print(f"{entry['model']}: SE={entry['mean_se']:.2f}, Success={entry['success_rate_pct']:.1f}%")
```

### Check budget status

```python
from src.utils.db_utils import get_remaining_budget, is_budget_exhausted

remaining = get_remaining_budget('tier1')
print(f"Tier 1 remaining: ${remaining:.2f}")

if is_budget_exhausted('tier1', threshold=0.9):
    print("⚠ Tier 1 budget 90% exhausted!")
```

---

## Research Questions

This benchmark answers 5 key questions:

1. **RQ1**: Which models excel at code transformation?
   - Hypothesis: GPT-4 and Claude 3.5 achieve SE >8.0

2. **RQ2**: Does prompt engineering matter?
   - Hypothesis: Few-shot k=5 improves SE by ≥2 points

3. **RQ3**: Are some languages easier to transform?
   - Hypothesis: Python SE > C++ SE by 60%

4. **RQ4**: How does complexity affect transformation?
   - Hypothesis: SE drops sharply at CC≈25

5. **RQ5**: What's the cost-quality tradeoff?
   - Hypothesis: DeepSeek-V3 matches GPT-4 at 11% cost

See [ricerca-domande.md](ricerca-domande.md) for full hypotheses and statistical tests.

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

**Status**: Phase 1 Complete ✅ | Phase 2 Ready to Start 🔜
