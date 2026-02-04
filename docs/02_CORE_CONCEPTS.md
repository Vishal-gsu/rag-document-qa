# 📖 **02 - Core Concepts Explained**

## **1. What is RAG?**

**RAG = Retrieval-Augmented Generation**

It's a 2-step process:
```
Step 1: RETRIEVAL  - Find relevant info from documents
Step 2: GENERATION - Use that info to generate answer
```

### **Without RAG (LLM Only)**
```python
Q: "Who is the author of textbook A?"
LLM: "The author is Stephen... Marsland? Maybe?"
     (No source, might be wrong)
```

### **With RAG**
```python
Q: "Who is the author of textbook A?"
   ↓
RETRIEVE: Search documents, find:
   "STEPHEN MARSLAND is the author"
   ↓
GENERATE: Use Ollama to write:
   "According to the textbook, Stephen Marsland is the author"
   (Confident + sourced!)
```

---

## **2. Vector Databases & Similarity Search**

### **What are Vectors?**

Text → Numbers (embeddings)

```
"Machine learning is AI" 
  ↓ (embedding model)
[0.23, 0.45, -0.12, 0.89, ..., -0.34]
  ↓ 384 dimensions
```

**Why numbers?**
- Computers understand numbers, not text
- Numbers capture MEANING
- Semantically similar texts have similar vectors

### **How Similarity Search Works**

```
Your Question: "What is machine learning?"
  ↓
Embedding: [0.21, 0.47, -0.11, 0.88, ...]

Compare to stored vectors:
Chunk 1: [0.22, 0.46, -0.10, 0.89, ...] → 98% similar ✅
Chunk 2: [0.15, 0.52, -0.08, 0.85, ...] → 85% similar ✅
Chunk 3: [0.45, 0.12, 0.67, 0.23, ...] → 15% similar ❌

Return: Chunks 1 & 2 (high similarity)
```

---

## **3. HNSW Algorithm (Why it's Fast)**

### **The Problem: Linear Search is SLOW**

With 4,772 vectors:
```
❌ Brute Force: Compare query to ALL 4,772 vectors
   Time: ~100ms per query
   
✅ HNSW: Navigate smart graph structure  
   Time: ~5ms per query
   Speedup: 20x faster!
```

### **How HNSW Works**

HNSW = **Hierarchical Navigable Small World**

**Hierarchical = Multiple Layers**
```
Layer 2:  O----O----O  (few vectors, long jumps)
Layer 1:  O--O--O--O  (more vectors, medium jumps)  
Layer 0:  O-O-O-O-O-O-O-O (all vectors, fine search)

Starting at top, navigate down layers to find nearest neighbors!
```

### **Search Process**

```
Query Vector: [0.21, 0.47, -0.11, ...]

1. Start at layer 2 (top):
   Find closest vector → O

2. Go to layer 1:
   From that O, find closer neighbor → O'

3. Go to layer 0 (bottom):
   From O', find all nearest neighbors
   
Final: Top 5 closest vectors found!
```

**Result: Only ~14 distance calculations instead of 4,772!**

---

## **4. Embeddings Explained**

### **What is an Embedding?**

Convert text to vector (fixed-size array of numbers)

```
Input: "Machine learning uses algorithms"
Model: sentence-transformers/all-MiniLM-L6-v2
Output: 384 numbers: [-0.012, 0.456, ..., 0.234]
```

### **Why 384 Dimensions?**

- **More = Better** (captures more meaning)
- **Less = Faster** (less computation)
- **384 = Sweet spot** (balance accuracy & speed)

```
64D:   Super fast, but loses info ❌
384D:  Fast + accurate ✅ (we use this)
1536D: Accurate but slower ⚠️
```

### **Local vs Cloud Embeddings**

| Type | Model | Speed | Cost | Privacy |
|------|-------|-------|------|---------|
| **Local** | all-MiniLM-L6-v2 | Fast ✅ | Free ✅ | Private ✅ |
| **Cloud** | OpenAI | Slower ⚠️ | $$ ⚠️ | Shared ⚠️ |

**We use LOCAL** → No API costs, privacy, instant

---

## **5. Chunking Strategy**

### **Why Chunking?**

Documents are too long! Need to split them:

```
Entire PDF: 100 pages, 50,000 characters
  ↓
Too big for embedding model!
  ↓
Solution: Split into 1000-char chunks
  ↓
Result: ~50 manageable chunks
```

### **Current Chunking Settings**

```python
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks

Example:
Chunk 1: "...learning is the process of..."
Chunk 2: "...of discovering patterns in..."
         ^^^^^^^^ (200 char overlap)
```

**Why overlap?**
- If answer spans chunk boundary, overlap captures it
- Prevents losing information at edges

---

## **6. Prompt Engineering**

### **What is a Prompt?**

Instructions to the LLM on how to behave

```
Bad Prompt:
"Answer the question"
→ LLM gets confused, makes up stuff

Good Prompt:
"Using ONLY the provided context, answer this question.
Be concise. Do not add follow-up questions."
→ LLM follows rules, gives accurate answer
```

### **Our Prompt Optimization Journey**

**Before (Hallucinations):**
```
"I cannot confirm Stephen Marsland is the author..."
(Model too cautious, hedges even when answer clear)
```

**After (Direct):**
```
"Stephen Marsland is the author of this textbook."
(Model confident when evidence is clear)
```

**Key Changes:**
- ✅ Removed: "I cannot confirm..."
- ✅ Added: "Answer directly and confidently"
- ✅ Set: Temperature=0.4 (focused, not creative)
- ✅ Limited: max_tokens=800 (prevents rambling)

---

## **7. Similarity Threshold**

### **Why Filter by Threshold?**

Not all retrieved chunks are relevant!

```
Query: "What is machine learning?"
Chunk 1: 70% similar ✅ (Keep)
Chunk 2: 65% similar ✅ (Keep)
Chunk 3: 40% similar ✅ (Keep)
Chunk 4: 25% similar ❌ (Discard - noise)
Chunk 5: 15% similar ❌ (Discard - irrelevant)

Threshold = 30%
Result: Use chunks 1-3 only
```

**Current Setting: 30% threshold**
- Filters out noise
- Prevents hallucinations from irrelevant chunks
- Still retrieves 3-5 relevant chunks

---

## **8. Temperature Parameter**

Controls how "creative" vs "focused" the LLM is:

```
Temperature = 0.0 (Deterministic)
→ Always same answer, very rigid
→ Use when: Accuracy critical

Temperature = 0.4 (Balanced) ✅ (We use this)
→ Focused but natural
→ Use when: RAG, factual answers

Temperature = 1.0 (Creative)
→ Varies answer, more creative
→ Use when: Story writing, brainstorming
```

---

## **9. Tokens**

### **What is a Token?**

A piece of text (roughly 0.75 words per token)

```
"Machine learning" = 2 tokens
"is a subset of artificial intelligence" = 7 tokens

Total sentence ≈ 9 tokens
```

### **Token Limits**

```
max_tokens=150:  ~110 words (too short)
max_tokens=800:  ~600 words (our default)
max_tokens=1500: ~1125 words (very detailed)
```

**Why limit?**
- Prevent rambling
- Limit inference cost
- Generate in reasonable time

---

## **10. Docker & Persistence**

### **What is Docker?**

Package software + dependencies in a container

```
Endee Server:
  Before: "How do I install Endee?"
  After: "docker compose up -d" ✅

Benefits:
- Same on any computer ✅
- No dependency conflicts ✅
- Data persists ✅
```

### **Our Docker Setup**

```yaml
services:
  endee:
    image: endeeio/endee-server:latest
    ports:
      - "8080:8080"
    volumes:
      - endee-data:/data  # Persistent storage!
```

**Key Point:** `endee-data` volume persists across restarts!

---

## **11. LLM Options**

### **3 Different LLM Providers**

#### **1. OpenAI API** ☁️
```
Pros:
- Best quality (GPT-3.5-turbo)
- No hardware needed
- Cloud-based

Cons:
- Costs $$
- Requires API key
- Shared servers
```

#### **2. Ollama GPU** 🚀
```
Pros:
- Free (open source)
- Fast (GPU accelerated)
- Privacy (local)

Cons:
- Needs GPU
- Slower than OpenAI
```

#### **3. Ollama CPU** 💻
```
Pros:
- Free
- Privacy
- No GPU needed

Cons:
- Slow (~2 tokens/sec vs 20 with GPU)
- High CPU usage
```

**We use Ollama GPU** (phi3 model, free, fast, private)

---

## **12. Evaluation Metrics**

### **How Do We Measure Success?**

```
Retrieval Quality:
- Similarity scores: 60-70% ✅
- Relevant chunks retrieved ✅

Generation Quality:
- No hallucinations ✅
- Sources are accurate ✅
- Answers are concise ✅

Performance:
- Query time: 2-3 seconds ✅
- Search time: <10ms ✅
```

---

## **Key Formulas**

### **Cosine Similarity**
```
Given two vectors A and B:

Similarity = (A · B) / (||A|| × ||B||)

Range: 0 (completely different) to 1 (identical)
```

### **HNSW Complexity**
```
Brute Force Search:  O(n) = comparing all n vectors
HNSW Search:         O(log n) = logarithmic!

Example with 4,772 vectors:
Brute: 4,772 comparisons
HNSW: ~log(4,772) ≈ 12 comparisons ← 400x faster!
```

---

## **Quick Recap**

| Concept | What | Why |
|---------|------|-----|
| **RAG** | Retrieve + Generate | Prevent hallucination |
| **Vectors** | Numbers from text | Enable similarity search |
| **HNSW** | Smart graph search | 100-1000x faster |
| **Embeddings** | Text → Numbers | Semantic understanding |
| **Chunking** | Split documents | Manageable size |
| **Prompting** | Instructions to LLM | Control behavior |
| **Threshold** | Similarity cutoff | Filter noise |
| **Temperature** | Creativity level | Balance accuracy/variety |
| **Tokens** | Text pieces | Count response length |
| **Docker** | Container | Reproducibility |

---

## **Next: Deep Math** 📚

Ready for the hardcore theory? Go to **THEORY.md** for:
- Vector mathematics
- HNSW algorithm details
- Information retrieval formulas
- Transformer embedding models

**Or skip to code:** Go to **04_CODE_WALKTHROUGH.md** to see it all in action!
