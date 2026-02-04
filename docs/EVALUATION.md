# Evaluation & Metrics Guide

This guide explains how to evaluate your RAG system with quantitative metrics and visualizations.

---

## 📊 Why Evaluation Matters

For your internship submission, having **quantitative metrics and charts** demonstrates:

✅ **Scientific Rigor** - Data-driven assessment, not just subjective claims  
✅ **Technical Depth** - Understanding of evaluation methodologies  
✅ **Professional Approach** - Industry-standard practices  
✅ **Credibility** - Concrete evidence of system performance  

---

## 🎯 Metrics Overview

### 1. Retrieval Metrics

**Keyword Coverage**
- Measures: % of expected keywords found in retrieved context
- Range: 0.0 to 1.0 (higher is better)
- Target: > 0.7

**Similarity Score**
- Measures: Cosine similarity between query and retrieved chunks
- Range: -1 to 1 (higher is better)
- Target: > 0.6

### 2. Answer Quality Metrics

**Keyword Score**
- Measures: % of expected keywords in generated answer
- Range: 0.0 to 1.0 (higher is better)
- Target: > 0.6

**Attribution Rate**
- Measures: % of answers that cite sources
- Range: 0.0 to 1.0 (higher is better)
- Target: > 0.8

**Answer Length**
- Measures: Number of words in answer
- Target: 20-200 words (comprehensive but concise)

### 3. Performance Metrics

**Latency**
- Total Query Time: Full end-to-end time
- Breakdown:
  - Embedding Time: Query vectorization
  - Retrieval Time: Vector search
  - Generation Time: LLM response
- Target: < 3000 ms total

---

## 🚀 Quick Start

### Step 1: Install Visualization Dependencies

```powershell
pip install matplotlib seaborn
```

### Step 2: Run Evaluation

```powershell
# Make sure documents are indexed first
python main.py --mode index --docs data/documents

# Run evaluation
python evaluation.py
```

This will:
- Test 10 diverse questions
- Measure retrieval quality
- Evaluate answer quality
- Measure latency
- Save results to `evaluation_results/`

### Step 3: Generate Charts

```powershell
python visualize.py
```

This creates:
- `latency_breakdown.png` - Pie chart of latency components
- `similarity_distribution.png` - Histogram of similarity scores
- `metrics_comparison.png` - Bar chart of key metrics
- `category_performance.png` - Performance by question type
- `latency_percentiles.png` - P50, P95, P99 latencies

---

## 📈 Understanding the Charts

### 1. Latency Breakdown (Pie Chart)

**What it shows:**
- Proportion of time spent in each pipeline stage

**How to interpret:**
```
Embedding: 10-20%    ← Query vectorization
Retrieval: 2-5%      ← Vector search (very fast!)
Generation: 75-85%   ← LLM call (bottleneck)
```

**Insights:**
- Generation dominates → Consider using faster models or caching
- High embedding % → Batch queries or cache embeddings
- High retrieval % → Optimize vector search or reduce index size

### 2. Similarity Distribution (Histogram)

**What it shows:**
- Distribution of similarity scores for retrieved chunks

**How to interpret:**
```
Most scores > 0.6  → Good semantic matching
Scores < 0.4       → Poor retrieval, needs improvement
Bimodal distribution → Mixed quality, investigate outliers
```

**Insights:**
- High average similarity → Effective embeddings
- Low variance → Consistent retrieval quality
- Long tail of low scores → Some questions poorly matched

### 3. Metrics Comparison (Bar Chart)

**What it shows:**
- Side-by-side comparison of key performance metrics

**How to interpret:**
```
All metrics > 0.7   → Excellent system
0.5 - 0.7           → Good, room for improvement
< 0.5               → Needs optimization
```

**Insights:**
- Keyword Coverage → Document corpus completeness
- Similarity → Embedding quality
- Keyword Score → Answer relevance
- Attribution → Source citation rate

### 4. Category Performance (Grouped Bar)

**What it shows:**
- Performance varies by question type

**How to interpret:**
```
Definitions:    High score → System good at factual answers
Comparisons:    Lower score → May need better context
Explanations:   Variable → Depends on document detail
```

**Insights:**
- Strong categories → Leverage in demo
- Weak categories → Improve documentation or chunking
- Patterns → Identify system strengths/weaknesses

### 5. Latency Percentiles (Bar Chart)

**What it shows:**
- Latency at different percentiles (P50, P95, P99)

**How to interpret:**
```
P50 (median):  Typical user experience
P95:           95% of queries faster than this
P99:           Worst-case scenarios
```

**Insights:**
- Low P50 → Fast average performance
- High P95-P99 spread → Inconsistent, investigate outliers
- All < 3000ms → Good user experience

---

## 📊 Sample Results

**Expected Performance (with 3 sample documents):**

```
LATENCY METRICS:
  Average Total:      2,500 ms
  P50 (Median):       2,300 ms
  P95:                3,200 ms
  Breakdown:
    - Embedding:      400 ms (16%)
    - Retrieval:      50 ms (2%)
    - Generation:     2,050 ms (82%)

RETRIEVAL QUALITY:
  Keyword Coverage:   75%
  Avg Similarity:     0.682
  Top-1 Similarity:   0.754

ANSWER QUALITY:
  Keyword Score:      68%
  Attribution Rate:   90%
  Avg Answer Length:  87 words
```

---

## 🎯 Improving Your Scores

### To Improve Retrieval (Keyword Coverage, Similarity)

**1. Better Chunking**
```python
# Experiment with chunk sizes
CHUNK_SIZE = 300  # Try: 300, 500, 800
CHUNK_OVERLAP = 100  # More overlap = better context
```

**2. More Documents**
- Add domain-specific documents
- Ensure comprehensive coverage
- Remove irrelevant content

**3. Better Embeddings**
```python
# Try larger embedding model
EMBEDDING_MODEL = "text-embedding-3-large"  # More expensive but better
```

### To Improve Answer Quality

**1. Better Prompts**
```python
# In rag_engine.py, modify prompt template
prompt = f"""Use ONLY the context below to answer.
Include specific details and cite sources.

Context: {context}
Question: {question}
"""
```

**2. Retrieve More Context**
```python
TOP_K_RESULTS = 5  # Instead of 3
```

**3. Adjust Temperature**
```python
temperature = 0.1  # Lower = more focused (less creative)
```

### To Improve Latency

**1. Use Faster Models**
```python
CHAT_MODEL = "gpt-3.5-turbo"  # Faster than gpt-4
EMBEDDING_MODEL = "text-embedding-3-small"  # Faster than large
```

**2. Reduce Context**
```python
TOP_K_RESULTS = 2  # Fewer chunks = faster
max_tokens = 300  # Shorter responses = faster
```

**3. Cache Results** (future enhancement)
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(question):
    return query(question)
```

---

## 📝 Creating Your Test Set

### Customize for Your Domain

**Example: Healthcare Documents**
```python
test_cases = [
    {
        "question": "What are the symptoms of diabetes?",
        "expected_keywords": ["blood sugar", "insulin", "symptoms", "glucose"],
        "category": "medical_facts"
    },
    {
        "question": "How is hypertension diagnosed?",
        "expected_keywords": ["blood pressure", "diagnosis", "measurement"],
        "category": "diagnosis"
    },
    # Add 8-10 more...
]
```

**Example: Financial Documents**
```python
test_cases = [
    {
        "question": "What is compound interest?",
        "expected_keywords": ["interest", "principal", "compound", "growth"],
        "category": "concepts"
    },
    # Add more...
]
```

### Test Set Best Practices

✅ **Diverse question types:**
- Definitions ("What is X?")
- Comparisons ("X vs Y?")
- How-to ("How does X work?")
- Specific facts ("When/Where/Who?")

✅ **Varying difficulty:**
- Easy: Direct facts in documents
- Medium: Requires connecting information
- Hard: Needs inference or synthesis

✅ **Representative queries:**
- Actual user questions
- Common use cases
- Edge cases

---

## 📊 Presenting Results

### For Your README

**Add a Results section:**

```markdown
## 📊 Performance Results

Our RAG system achieves strong performance across multiple metrics:

### Key Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Retrieval Accuracy | 75% | >70% | ✅ Excellent |
| Answer Quality | 68% | >60% | ✅ Good |
| Attribution Rate | 90% | >80% | ✅ Excellent |
| Avg Latency (P50) | 2.3s | <3s | ✅ Good |

### Performance Breakdown

![Metrics Comparison](evaluation_results/metrics_comparison.png)

### Latency Analysis

Average query latency: **2.5 seconds**

![Latency Breakdown](evaluation_results/latency_breakdown.png)

**Key Finding:** LLM generation accounts for 82% of latency, 
suggesting optimization opportunities through caching or faster models.

### Retrieval Quality

![Similarity Distribution](evaluation_results/similarity_distribution.png)

Average similarity score: **0.68** indicating effective semantic matching.

### Detailed Results

Full evaluation results available in `evaluation_results/`
```

### For Your Interview

**Be ready to discuss:**

1. **"What are your system's metrics?"**
   → Show the charts, explain each metric

2. **"How did you evaluate performance?"**
   → Explain test set, metrics, methodology

3. **"What's the biggest bottleneck?"**
   → Point to latency chart: "LLM generation at 82%"

4. **"How would you improve retrieval accuracy?"**
   → "Currently at 75%, could try larger embeddings or more docs"

5. **"Show me the data"**
   → Open `evaluation_results/summary_*.json`

---

## 🎓 Advanced Evaluation (Optional)

### Human Evaluation

Create a survey form:
```
For each answer, rate 1-5:
- Relevance: Does it answer the question?
- Accuracy: Is the information correct?
- Completeness: Is it thorough enough?
- Source Attribution: Are sources cited?
```

### A/B Testing

```python
# Compare configurations
config_a = {"chunk_size": 500, "top_k": 3}
config_b = {"chunk_size": 300, "top_k": 5}

# Run evaluation for each
results_a = evaluate(config_a)
results_b = evaluate(config_b)

# Compare metrics
```

### RAGAS Framework (Advanced)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Automated quality assessment
scores = evaluate(
    questions=questions,
    answers=answers,
    contexts=contexts,
    metrics=[faithfulness, answer_relevancy]
)
```

---

## ✅ Evaluation Checklist

Before submission, ensure you have:

- [ ] Run evaluation on test set (10+ questions)
- [ ] Generated all visualization charts
- [ ] Saved results to evaluation_results/
- [ ] Documented metrics in README
- [ ] Included charts in README or docs
- [ ] Can explain each metric
- [ ] Understand what numbers mean
- [ ] Know how to improve each metric
- [ ] Have concrete evidence of performance

---

## 🎯 Success Criteria

**Your evaluation is submission-ready when:**

✅ All metrics calculated and documented  
✅ Charts generated and look professional  
✅ Results saved and reproducible  
✅ Performance meets or exceeds targets  
✅ You can explain every chart  
✅ You know why metrics are what they are  
✅ You have improvement ideas backed by data  

---

## 📁 File Structure After Evaluation

```
assignment_rag/
├── evaluation.py              # Evaluation script
├── visualize.py              # Visualization script
└── evaluation_results/
    ├── detailed_results_20260131_143022.json
    ├── summary_20260131_143022.json
    ├── latency_breakdown.png
    ├── similarity_distribution.png
    ├── metrics_comparison.png
    ├── category_performance.png
    └── latency_percentiles.png
```

---

## 🚀 Next Steps

1. **Run evaluation:** `python evaluation.py`
2. **Generate charts:** `python visualize.py`
3. **Review results:** Check evaluation_results/
4. **Update README:** Add metrics and charts
5. **Iterate:** Improve based on findings
6. **Document learnings:** What worked, what didn't

---

**Your RAG system now has quantitative proof of performance!** 📊✨
