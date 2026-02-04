# 💻 **04 - Code Walkthrough**

This guide explains every important line of code. Read this with the actual code files open.

---

## **File 1: config.py - Configuration Hub**

**Purpose:** Central configuration for the entire system

### **Key Settings**

```python
# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

Why 384?
- Balances accuracy (better than 64D) 
- Balances speed (faster than 1536D)
- Proven for semantic search ✅
```

```python
# Document Chunking
CHUNK_SIZE = 1000           # Characters per chunk
CHUNK_OVERLAP = 200         # Characters overlapping

Why these values?
- 1000 chars ≈ 150 words (good context length)
- 200 overlap = 20% (captures info spanning boundaries)

Before optimization: CHUNK_SIZE=500, OVERLAP=50
Problem: Too small chunks, missed context
Solution: Increased to 1000/200 ✅
```

```python
# Retrieval Settings
SIMILARITY_THRESHOLD = 0.30  # Minimum similarity score
TOP_K_RESULTS = 5            # Return top-5 chunks

Why 30%?
Before: 40% threshold rejected good results
Problem: No chunks above 40%, got empty results
Solution: Lowered to 30%, now gets 3-5 relevant chunks ✅

Why top-5?
Before: 3 chunks sometimes insufficient
Solution: Increased to 5 for better context ✅
```

```python
# LLM Configuration
LLM_TEMPERATURE = 0.4        # Creativity level
LLM_MAX_TOKENS = 800         # Response length limit
LLM_TIMEOUT = 180            # 3 minutes max wait

Why 0.4 temperature?
Before: 0.7 (too creative, wandering, asking follow-ups)
       1.0 (would be even more creative)
After: 0.4 (focused, confident, concise)
Perfect for factual RAG tasks ✅

Why 800 tokens?
Before: 500 (too short, incomplete answers)
        300 (way too short)
After: 800 (≈600 words, good balance) ✅

Why 180s timeout?
Ollama on GPU: ~10-20s response
CPU: ~60-120s response
180s = safety margin, no timeout errors ✅
```

### **ENDEE Configuration**

```python
ENDEE_URL = "http://localhost:8080"
ENDEE_INDEX_NAME = "documents"
ENDEE_METRIC = "cosine"

Endee must be running!
Check: curl http://localhost:8080/status
```

---

## **File 2: embedding_engine.py - Text → Vectors**

**Purpose:** Convert text to 384-dimensional vectors

### **Key Method: embed_text()**

```python
def embed_text(self, text: str) -> list:
    """Convert text to 384D vector"""
    embedding = self.model.encode(text)
    return embedding.tolist()

Example:
Input: "Machine learning uses algorithms"
Output: [0.23, 0.45, -0.12, 0.89, ..., -0.34]
        └─ 384 numbers ─────────────────────┘

Real vectors are trained on billions of texts
So they capture semantic meaning! ✅
```

### **Key Method: embed_batch()**

```python
def embed_batch(self, texts: list) -> list:
    """Embed multiple texts (faster than one-by-one)"""
    embeddings = self.model.encode(texts)
    return embeddings.tolist()

Why batch?
Single: for text in texts: embed_text(text)  # Slow!
Batch: embed_batch(texts)  # Fast! ⚡
Speedup: 5-10x for GPU
```

### **Initialization**

```python
from sentence_transformers import SentenceTransformer

self.model = SentenceTransformer('all-MiniLM-L6-v2')

First time: Downloads model (40MB, ~1 min)
After: Instant (cached locally)

Why this model?
✅ Fast (6-12ms per embedding)
✅ Free (no API costs)
✅ Accurate (good semantic understanding)
✅ Small (40MB, fits on GPU)
✅ Popular (millions of users)
```

---

## **File 3: vector_store.py - Endee Interface**

**Purpose:** Interface to Endee vector database

### **Initialization**

```python
def __init__(self, endee_url: str = "http://localhost:8080"):
    self.client = Endee(url=endee_url)
    self.index = None
```

**What it does:**
1. Connects to Endee server
2. Creates Endee client object
3. Index initialized as None (created on first use)

### **Critical Method: _ensure_index()**

```python
def _ensure_index(self):
    """Ensure index exists, create if needed"""
    if not self.client.index_exists("documents"):
        self.client.create_index(
            name="documents",
            dimension=384,
            metric="cosine"
        )
    
    self.index = self.client.get_index("documents")
```

**Why this is critical:**
- `index_exists()` checks if "documents" index created
- `create_index()` creates it (runs once)
- `get_index()` retrieves Index object (must call after create!)

**Common Bug (Fixed in our code):**
```python
# WRONG ❌
self.index = self.client.create_index(...)  # Returns string!

# RIGHT ✅
self.client.create_index(...)
self.index = self.client.get_index(...)  # Returns Index object!
```

### **Method: add_vectors() - Batch Upload**

```python
def add_vectors(self, vectors, metadata, ids):
    """Add vectors in batches to Endee"""
    self._ensure_index()
    
    BATCH_SIZE = 1000  # Endee limit per request
    
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = {
            'vectors': vectors[i:i+BATCH_SIZE],
            'ids': ids[i:i+BATCH_SIZE],
            'metadatas': metadata[i:i+BATCH_SIZE]
        }
        self.index.upsert(**batch)

Example with 4,772 vectors:
Iteration 1: Add vectors 0-999 ✅
Iteration 2: Add vectors 1000-1999 ✅
Iteration 3: Add vectors 2000-2999 ✅
Iteration 4: Add vectors 3000-3999 ✅
Iteration 5: Add vectors 4000-4772 ✅

Total: 5 API calls (instead of failing with 1 call!)
```

**Why batching?**
- Endee has 1000 vector/request limit
- Batching prevents: "Vector batch too large" error
- Automatic handling inside our code ✅

### **Method: search() - Query Endee**

```python
def search(self, query_vector, k=5):
    """Search for top-k similar vectors"""
    self._ensure_index()
    
    results = self.index.query(
        vector=query_vector,
        k=k,
        ef=100  # HNSW accuracy parameter
    )
    
    # Convert to our format
    return [
        {
            'id': r['id'],
            'similarity': r['score'],
            'metadata': r['metadata']
        }
        for r in results
    ]

Input:
- query_vector: [0.21, 0.47, -0.11, ...] (384D)
- k: Return top-5

HNSW Process:
1. Navigate hierarchy
2. Find ~12 candidate vectors
3. Calculate distance to candidates
4. Return top-5

Output:
[
    {'id': 'doc_234', 'similarity': 0.78, 'metadata': {...}},
    {'id': 'doc_156', 'similarity': 0.72, 'metadata': {...}},
    ...
]

Performance: ~5ms (vs 100ms for brute force!) ⚡
```

---

## **File 4: rag_engine.py - RAG Pipeline**

**Purpose:** Orchestrate the full RAG workflow

### **Initialization**

```python
def __init__(self):
    self.embedding_engine = EmbeddingEngine()
    self.vector_store = VectorStore()
    self.document_processor = DocumentProcessor()
```

**Three engines initialized:**
1. **EmbeddingEngine** → Text to vectors
2. **VectorStore** → Endee interface
3. **DocumentProcessor** → Load/chunk documents

### **Method: add_documents() - Index Pipeline**

```python
def add_documents(self, document_chunks):
    """Full indexing pipeline"""
    
    # Step 1: Embed all chunks
    chunks_with_embeddings = [
        {
            **chunk,
            'embedding': self.embedding_engine.embed_text(chunk['text'])
        }
        for chunk in document_chunks
    ]
    # Result: Each chunk has 'embedding' key with 384D vector
```

**Why:** Convert text to vectors

```python
    # Step 2: Extract vectors and metadata
    vectors = [doc['embedding'] for doc in chunks_with_embeddings]
    # vectors = [[0.23, 0.45, ...], [0.12, 0.67, ...], ...]
    
    metadata = [
        {
            'text': doc['text'],
            'filename': doc['metadata'].get('filename', 'Unknown')
        }
        for doc in chunks_with_embeddings
    ]
    # metadata = [{'text': '...', 'filename': 'doc1.pdf'}, ...]
    
    ids = [f"doc_{i}" for i in range(len(chunks_with_embeddings))]
    # ids = ['doc_0', 'doc_1', ..., 'doc_4771']
```

**Why:** Format for Endee API (expects vectors array, metadata array, ids array)

```python
    # Step 3: Add to Endee
    self.vector_store.add_vectors(vectors, metadata, ids)
    # Batching happens inside! (1000 at a time)
```

**Full Pipeline Summary:**
```
PDF → Chunks → Embeddings → Formatted → Endee
          ↓
2,386 chunks → 4,772 vectors → Indexed in Endee ✅
```

### **Method: query() - Retrieval Pipeline**

```python
def query(self, question, top_k=5, similarity_threshold=0.30):
    """Full retrieval pipeline"""
    
    # Step 1: Embed the question (same model as docs!)
    query_embedding = self.embedding_engine.embed_text(question)
    # query_embedding = [0.21, 0.47, -0.11, ...] (384D)
```

**Why:** Convert question to same vector space as documents

```python
    # Step 2: Search Endee
    results = self.vector_store.search(query_embedding, k=top_k)
    # results = top-5 chunks by similarity
    # Example:
    # [
    #     {'id': 'doc_234', 'similarity': 0.78, ...},
    #     {'id': 'doc_156', 'similarity': 0.72, ...},
    #     {'id': 'doc_890', 'similarity': 0.68', ...},
    #     {'id': 'doc_45', 'similarity': 0.52, ...},
    #     {'id': 'doc_100', 'similarity': 0.48, ...}
    # ]
```

**Why:** Find semantically similar chunks

```python
    # Step 3: Filter by threshold
    relevant = [
        r for r in results 
        if r['similarity'] >= similarity_threshold
    ]
    # Before: threshold 0.40, got 0 results ❌
    # After: threshold 0.30, get 3-5 results ✅
```

**Why:** Remove low-quality matches, prevent hallucination

```python
    return relevant  # Return filtered results
```

**Query Pipeline Summary:**
```
Question → Embed (384D) → Search Endee → Top-5 → Filter → Return
                               ↓
                            ~5ms HNSW search ✅
```

### **Example Query Execution**

```python
Q: "What is machine learning?"
   ↓ (embed with all-MiniLM-L6-v2)
Query Vector: [0.19, 0.42, -0.13, 0.87, ...]

Search in Endee (HNSW):
  Navigate hierarchy → Find ~12 candidates → Calculate distances

Results:
  Chunk 1: 0.78 similarity ✅ (Kept, >0.30)
  Chunk 2: 0.72 similarity ✅ (Kept, >0.30)
  Chunk 3: 0.68 similarity ✅ (Kept, >0.30)
  Chunk 4: 0.52 similarity ✅ (Kept, >0.30)
  Chunk 5: 0.48 similarity ✅ (Kept, >0.30)

→ Pass to LLM with all 5 chunks ✅
```

---

## **File 5: llm_manager.py - LLM Interface**

**Purpose:** Abstract LLM provider (OpenAI, Ollama GPU, Ollama CPU)

### **Method: generate()**

```python
def generate(self, messages, temperature=0.4, max_tokens=800):
    """Generate response from LLM"""
    
    timeout = 180  # 3 minutes
    
    try:
        if self.provider == "openai":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
        
        elif self.provider == "ollama":
            # Connect to local Ollama
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                timeout=timeout
            )
    
    except TimeoutError:
        return "Request timed out. Try simpler query."
    
    return response.choices[0].message.content
```

**Key Parameters:**
```
temperature = 0.4:
  ✅ Focused, confident (for RAG)
  ❌ 0.7 would be too creative
  ❌ 1.0 would be very random

max_tokens = 800:
  ✅ ~600 words (detailed answer)
  ❌ 500 would cut off good answers
  ❌ 300 would be too short

timeout = 180s:
  ✅ Ollama GPU: ~10-20s
  ✅ Ollama CPU: ~60-120s
  ✅ OpenAI: ~3-5s
```

---

## **File 6: app.py - Streamlit Interface**

**Purpose:** Web UI for the entire system

### **Tab 1: Upload Documents**

```python
st.write("### 📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload files",
    accept_multiple_files=True,
    type=["pdf", "txt", "docx", "md", "csv", "json", "html"]
)

# Process uploaded files
if uploaded_files:
    for file in uploaded_files:
        chunks = processor.process_document(file)
        rag_engine.add_documents(chunks)
        st.success(f"Indexed {len(chunks)} chunks")
```

**Flow:**
```
User uploads file → Process into chunks → Embed → Index in Endee
```

### **Tab 2: Ask Questions (Main RAG)**

```python
st.write("### 🔍 Ask Questions")

question = st.text_input("Your question:")
if st.button("Ask"):
    # Retrieve
    results = rag_engine.query(question)
    
    if not results:
        st.warning("No relevant results found")
        return
    
    # Build context
    context = "\n".join([r['metadata']['text'] for r in results])
    
    # Generate answer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    answer = llm_manager.generate(messages)
    
    # Display
    st.write("### 📝 Answer")
    st.write(answer)
    
    st.write("### 📚 Sources")
    for i, result in enumerate(results, 1):
        st.write(f"**Chunk {i}** ({result['similarity']:.0%} similar)")
        st.write(result['metadata']['text'][:200] + "...")
```

**Flow:**
```
User asks question
    ↓
Retrieve from Endee (top-5 chunks)
    ↓
Build prompt with context
    ↓
Generate with Ollama (0.4 temp, 800 tokens)
    ↓
Display answer + sources + similarity scores
```

### **Key Prompts**

```python
SYSTEM_PROMPT = """You are an expert assistant...
Use ONLY the provided context.
Do NOT add follow-up questions.
Be confident in your answers."""

Why this prompt?
Before: Model would hedge ("I cannot confirm...")
Problem: Used knowledge beyond context
Solution: Explicit instruction + low temperature ✅
```

### **Tab 3: Settings**

```python
st.write("### ⚙️ Settings")

provider = st.selectbox(
    "LLM Provider",
    ["Ollama GPU", "Ollama CPU", "OpenAI"]
)

temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.4
)

similarity_threshold = st.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.30
)
```

**User can customize:**
- Which LLM to use
- How creative (temperature)
- How strict (threshold)

---

## **File 7: document_processor.py**

**Purpose:** Load and chunk various file formats

### **Process Flow**

```python
def process_document(file) -> list:
    """Load file → Chunk → Return chunks"""
    
    # Detect format and load
    if file.type == "application/pdf":
        text = extract_pdf(file)
    elif file.type == "text/plain":
        text = file.read().decode()
    elif file.type == "application/vnd.openxmlformats...":
        text = extract_docx(file)
    # ... other formats
    
    # Chunk text
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Add metadata
    chunks = [
        {
            'text': chunk,
            'metadata': {'filename': file.name}
        }
        for chunk in chunks
    ]
    
    return chunks
```

### **Chunking Logic**

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    
    return chunks

Example (simplified):
Text: "A B C D E F G H..."
Size: 3, Overlap: 1

Chunk 1: "A B C"
Chunk 2: "C D E"    ← "C" overlaps!
Chunk 3: "E F G"    ← "E" overlaps!
Chunk 4: "G H..."

Result: Boundary info preserved! ✅
```

---

## **Data Flow Visualization**

### **Indexing**

```
Upload PDF
    ↓
DocumentProcessor
    ├─ Extract text
    ├─ Split into chunks (1000 chars, 200 overlap)
    └─ Add filename metadata
           ↓
EmbeddingEngine
    ├─ Embed each chunk
    └─ Get 384D vectors
           ↓
VectorStore
    ├─ Format for Endee
    ├─ Batch into 1000-vector groups
    └─ Upsert to Endee
           ↓
Endee HNSW Index
    └─ 4,772 vectors indexed! ✅
```

### **Querying**

```
User Question: "What is machine learning?"
    ↓
EmbeddingEngine
    └─ Embed question → 384D vector
           ↓
VectorStore
    ├─ Query Endee with HNSW
    └─ Get top-5 chunks (~5ms)
           ↓
Similarity Filter
    └─ Keep chunks ≥ 0.30 similarity
           ↓
LLMManager
    ├─ Build prompt with context
    ├─ Send to Ollama (temp 0.4, max_tokens 800)
    └─ Receive answer
           ↓
Streamlit
    ├─ Display answer
    ├─ Show sources + similarity scores
    └─ User sees everything! ✅
```

---

## **Optimization Timeline**

### **Chunk Size Evolution**

```
Day 1: CHUNK_SIZE = 500, OVERLAP = 50
Problem: Too small, missing context
Retrieval: 35-40% similarity ❌

Day 2: CHUNK_SIZE = 1000, OVERLAP = 200
Result: Better context capture
Retrieval: 60-70% similarity ✅
Improvement: +25-30 percentage points! 🎯
```

### **Temperature Evolution**

```
Day 1: temperature = 0.7
Problem: Model too creative, adds follow-ups, hedges
Example: "I cannot confirm... Perhaps you might..."

Day 2: temperature = 0.5
Better but still cautious

Day 3: temperature = 0.4
Result: Direct, confident answers ✅
Example: "Stephen Marsland is the author."

Why 0.4?
- 0.0 = robotic (always same)
- 0.4 = balanced (confident + natural) ✅
- 1.0 = random (unpredictable)
```

### **Threshold Evolution**

```
Day 1: SIMILARITY_THRESHOLD = 0.40 (40%)
Problem: Top-5 chunks all < 40%, query returns nothing ❌
User sees: "No results found" 😞

Day 2: SIMILARITY_THRESHOLD = 0.30 (30%)
Result: 3-5 relevant chunks retrieved ✅
User gets: Good answers with sources 😊
```

---

## **Common Patterns**

### **Always Embed with Same Model**

```python
# ✅ CORRECT
embed_text = model.encode(text)          # Training
query_vector = model.encode(question)    # Same model!

# ❌ WRONG
embed_text = model1.encode(text)         # Model 1
query_vector = model2.encode(question)   # Model 2! ❌
# Result: Vector spaces don't match, terrible performance!
```

### **Always Batch Vectors**

```python
# ✅ CORRECT
for i in range(0, len(vectors), 1000):
    index.upsert(vectors[i:i+1000], ...)  # Batch

# ❌ WRONG
index.upsert(vectors, ...)  # All 4,772 at once!
# Result: API error "Vector batch too large" ❌
```

### **Always Check Index Exists**

```python
# ✅ CORRECT
if not client.index_exists("documents"):
    client.create_index(...)
index = client.get_index("documents")

# ❌ WRONG
index = client.create_index(...)  # Always creates
# Result: Error if runs twice ❌
```

---

## **Next Steps**

1. **Deep dive by component:** → **05_COMPONENT_GUIDE.md**
2. **When things break:** → **06_TROUBLESHOOTING.md**
3. **Run in production:** → **07_DEPLOYMENT.md**

