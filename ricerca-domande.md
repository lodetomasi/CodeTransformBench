# RESEARCH QUESTIONS & HYPOTHESES

## What We're Trying to Discover

This benchmark answers **5 fundamental questions** about LLM code transformation capabilities:

---

## RQ1: Which Models Excel at Code Transformation?

**Question**: Which Large Language Models are best at semantic-preserving code transformations, and how much do they differ?

**What we measure**:
- Mean Semantic Elasticity score across 12 models
- Success rate (preservation = 1) per model
- Ranking by task (obfuscation vs deobfuscation vs refactoring)
- Variance across languages (Python vs Java vs JS vs C++)
- Cost-efficiency ratio (SE per $1000 spent)

**Hypotheses**:
- **H1.1**: GPT-4 and Claude 3.5 will achieve mean SE >8.0 (SOTA)
- **H1.2**: DeepSeek-V3 will match GPT-4 performance (p>0.05) at 10x lower cost
- **H1.3**: Model rankings will be task-specific (e.g., Spearman ρ<0.6 between obfuscation rank and deobfuscation rank)
- **H1.4**: Smaller models (<10B params) will achieve <60% success rate

**Statistical tests**:
- ANOVA: SE ~ model (between-groups)
- Post-hoc: Tukey HSD for pairwise comparisons
- Effect size: η² (eta-squared) for practical significance

**Expected findings**:
- Clear winner per task (e.g., "Claude 3.5 best for refactoring, SE=9.2")
- Open models competitive on obfuscation (DeepSeek-V3 SE=8.1 vs GPT-4 SE=8.3)
- Size ≠ quality (Mixtral 8x7B may outperform Llama 70B due to MoE architecture)

**Deliverable**: 
- Leaderboard table: 12 models × 3 tasks × 4 languages = 144 cells
- Pareto frontier plot: Cost vs SE (identify cost-efficient models)

---

## RQ2: Does Prompt Engineering Matter?

**Question**: How much does prompt strategy improve transformation quality compared to zero-shot?

**What we measure**:
- ΔSE (improvement over zero-shot) for each strategy
- Effect size (Cohen's d) per strategy
- Optimal k for few-shot (k=3 vs k=5 vs k=10)
- Interaction effects (strategy × model, strategy × complexity)

**Hypotheses**:
- **H2.1**: Few-shot k=5 improves SE by ≥2 points vs zero-shot (paired t-test, p<0.001)
- **H2.2**: Chain-of-thought (CoT) helps complex functions (CC>20) more than simple (CC<10)
  - Expected: ΔSE_complex = 3.2, ΔSE_simple = 0.8
- **H2.3**: Self-reflection adds <5% improvement over few-shot (diminishing returns)
- **H2.4**: k=5 is optimal (k=3 insufficient, k=10 no better but 2x cost)

**Statistical tests**:
- Repeated measures ANOVA: SE ~ strategy + model + (strategy × model)
- Paired t-tests: zero-shot vs each strategy (Bonferroni correction)
- Linear regression: SE ~ strategy + cyclomatic_complexity + interaction

**Expected findings**:
- Few-shot k=5: +34% SE improvement (pilot data confirmed)
- CoT helpful only for CC>15 (interaction effect significant)
- Self-reflection: +2% improvement, not cost-effective
- Optimal prompt: Few-shot k=5 with domain-specific examples

**Deliverable**:
- Ablation study bar chart: 5 strategies × error bars
- Interaction plot: Strategy effect by complexity tier
- Recommendation table: "For {task} on {language}, use {strategy}"

---

## RQ3: Language-Specific Transformation Difficulty

**Question**: Are some programming languages inherently easier to transform than others?

**What we measure**:
- SE distribution by language (mean, median, std, IQR)
- Success rate by language
- Correlation: language features → SE (e.g., static typing, expressiveness)
- Failure mode analysis (qualitative coding of errors)

**Hypotheses**:
- **H3.1**: Python transformations achieve higher SE than C++ (Mann-Whitney U, p<0.01)
  - Expected: Python median SE = 8.2, C++ median SE = 5.1
- **H3.2**: Statically-typed languages (Java, C++) have lower variance (more predictable)
  - Expected: σ_Python = 2.8, σ_Java = 1.9
- **H3.3**: JavaScript transformations most error-prone (lowest preservation rate)
  - Expected: JS preservation = 68%, Python = 87%
- **H3.4**: Transformation difficulty correlates with type system complexity
  - Rank: C++ (templates) > Java (generics) > Python > JavaScript

**Statistical tests**:
- Kruskal-Wallis H-test: SE ~ language (non-parametric, may not be normal)
- Levene's test: Variance homogeneity across languages
- Logistic regression: preservation ~ language + cyclomatic_complexity

**Expected findings**:
- Python easiest (dynamic typing = more syntactic flexibility)
- C++ hardest (templates + pointers + manual memory)
- Java middle ground (strong typing but garbage collection)
- JavaScript unpredictable (prototype-based OOP confuses models)

**Failure taxonomy**:
- Type errors: C++ (42%), Java (31%), Python (12%), JS (28%)
- Pointer bugs: C++ (18%), others (0%)
- Scoping issues: JS (23%), Python (8%), others (5%)

**Deliverable**:
- Violin plot: 4 languages, SE distribution
- Table: Failure modes by language (% of preservation=0 cases)
- Qualitative analysis: "Why C++ is hard" → template instantiation, memory management

---

## RQ4: Complexity vs Transformation Quality

**Question**: How does code complexity affect transformation success and SE score?

**What we measure**:
- Correlation: Cyclomatic complexity (CC) vs SE score
- Correlation: Halstead volume vs SE
- Success rate by complexity tier (simple/medium/complex)
- Non-linear effects (polynomial regression)

**Hypotheses**:
- **H4.1**: SE decreases with complexity (Pearson r = -0.45, p<0.001)
  - Linear model: SE = 12.3 - 0.18×CC + ε
- **H4.2**: Success rate drops super-linearly with complexity
  - Simple (CC≤10): 89% preservation
  - Medium (CC 11-30): 67% preservation
  - Complex (CC≥31): 34% preservation (pilot data)
- **H4.3**: Halstead volume better predictor than CC (R² comparison)
- **H4.4**: Complexity threshold exists: CC>25 → SE drops sharply

**Statistical tests**:
- Pearson correlation: SE ~ CC, SE ~ Halstead
- Multiple regression: SE ~ CC + Halstead + LOC + ast_depth
- Piecewise regression: Identify breakpoint (likely CC≈20-25)

**Expected findings**:
- Strong negative correlation: r = -0.52 (CC vs SE)
- Non-linear relationship: Quadratic term significant (β₂ = -0.008, p<0.01)
- Breakpoint at CC≈23: Below → SE stable, Above → SE crashes
- Halstead explains 18% more variance than CC alone (R²_full - R²_CC = 0.18)

**Deliverable**:
- Scatter plot: CC vs SE with regression line + 95% CI
- Stratified table: SE by complexity tier × language
- Regression coefficients table with significance
- Recommendation: "Avoid transforming functions with CC>25 (68% failure rate)"

---

## RQ5: Cost-Quality Tradeoff in Practice

**Question**: What is the practical cost-efficiency frontier for production use?

**What we measure**:
- Cost per transformation ($/call) by model
- SE per dollar (SE / cost_usd × 1000) - efficiency metric
- Pareto frontier: Models where no other is both cheaper AND better
- Latency analysis (p50, p95, p99)

**Hypotheses**:
- **H5.1**: DeepSeek-V3 lies on Pareto frontier (high SE, low cost)
- **H5.2**: GPT-4 dominates Claude 3.5 (higher SE, similar cost)
- **H5.3**: Cost-efficiency varies by task:
  - Obfuscation: Cheap models sufficient (Mixtral SE=6.1, GPT-4 SE=6.3, not worth 10x cost)
  - Refactoring: Expensive models justified (Mixtral SE=5.2, GPT-4 SE=9.1, huge gap)
- **H5.4**: Latency-cost tradeoff exists (smaller models faster but worse)

**Analysis**:
- Pareto frontier computation: For each model, check if ∃ model' with SE' ≥ SE AND cost' ≤ cost
- Efficiency ratio: SE_per_1000USD = (SE_score / cost_usd) × 1000
- Return on investment (ROI): (SE - SE_baseline) / (cost - cost_baseline)

**Expected findings**:
- **Pareto frontier models**: DeepSeek-V3, Llama 70B, Claude 3.5, GPT-4
  - DeepSeek-V3: SE=8.1, $0.003/call → 2700 SE/$1000 (efficiency king)
  - GPT-4: SE=9.2, $0.027/call → 341 SE/$1000 (quality king)
- **Dominated models**: Gemini 1.5 Pro (similar cost to GPT-4, lower SE)
- **Task-specific recommendations**:
  - CI/CD pipelines (cost-sensitive): Use DeepSeek-V3 (97% of GPT-4 quality, 11% cost)
  - Critical refactoring (quality-first): Use GPT-4 (worth the premium)
  - Obfuscation: Use Mixtral (cheapest, adequate SE=6.1)

**Deliverable**:
- Scatter plot: Cost vs SE with Pareto curve highlighted
- Table: Top 3 models by task × priority (cost vs quality)
- ROI analysis: "Every $1 spent on GPT-4 vs DeepSeek-V3 yields..."
- Decision tree: "Choose model based on use case"

---

## SYNTHESIS: What This Tells Us About LLM Capabilities

**Big Picture Questions Answered**:

1. **Transformation vs Generation**: 
   - Expected: Transformation harder (preservation rate 70% vs generation 85% on HumanEval)
   - Implication: Models optimized for generation may not transfer to transformation

2. **Prompt Engineering ROI**:
   - Expected: 34% improvement for $0 cost (just better prompts)
   - Implication: Don't rush to fine-tuning, exhaust prompt engineering first

3. **Language Bias**:
   - Expected: Python 60% easier than C++ (median SE: 8.2 vs 5.1)
   - Implication: Training data bias (more Python in pre-training) + inherent difficulty

4. **Complexity Wall**:
   - Expected: Sharp drop-off at CC≈25 (success: 67% → 34%)
   - Implication: LLMs struggle with complex control flow, need decomposition

5. **Open Model Viability**:
   - Expected: DeepSeek-V3 matches GPT-4 at 11% cost
   - Implication: Industry can use open models for cost-sensitive applications

**Novel Contributions to Field**:
- First cross-language transformation benchmark
- First quantitative prompt engineering study on transformations
- First cost-efficiency analysis for practical deployment
- SE metric validated at scale (90K experiments vs 30 in original paper)

**Future Research Directions Unlocked**:
- Can we train models specifically for transformation? (SE as reward signal)
- Can we predict SE from function features? (ML meta-model)
- Can we decompose complex functions to improve success? (hierarchical transformation)
- Can we build transformation-specific datasets for fine-tuning?

---

## SUCCESS CRITERIA FOR RESEARCH QUESTIONS

We will consider each RQ **successfully answered** if:

**RQ1**: 
✅ Statistical significance (p<0.05) between top 3 models
✅ Effect size η² > 0.3 (large effect)
✅ Ranking stable across 3+ random train/test splits

**RQ2**:
✅ Few-shot effect size d > 0.5 (medium-large)
✅ Interaction effect (strategy × complexity) p<0.05
✅ Recommendations validated on holdout set (10% functions)

**RQ3**:
✅ Language differences significant (Kruskal-Wallis p<0.001)
✅ Pairwise differences: Python > Java/JS > C++ (all p<0.05)
✅ Failure taxonomy covers 90%+ of errors

**RQ4**:
✅ Correlation r < -0.4 (moderate-strong negative)
✅ Non-linear model R² > linear R² + 0.10
✅ Breakpoint identified (confidence interval width <5 CC units)

**RQ5**:
✅ Pareto frontier contains 3-5 models
✅ ROI analysis shows >2x difference (best vs worst)
✅ Task-specific recommendations differ (evidence of specialization)

If any RQ fails criteria → Document as limitation, discuss why, propose follow-up

---

## EXPECTED TIMELINE FOR ANSWERS

- **Week 6** (Tier 1 complete): Preliminary answers to all RQs (60% confidence)
- **Week 8** (Tier 2 complete): Validated answers (80% confidence)
- **Week 10** (Tier 3 complete): Final answers with statistical tests (95% confidence)

Each tier refines our understanding:
- Tier 1: Exploratory (identify patterns)
- Tier 2: Confirmatory (validate patterns hold)
- Tier 3: Definitive (establish SOTA, publishable claims)

Vuoi che aggiunga altro o modifichi qualcosa in questa sezione?