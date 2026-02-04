# 🔌 **03 - Endee Integration Guide**

## **1. Why Endee? Why Not Something Else?**

### **The Challenge**
You need to store 4,772 embedding vectors and search through them quickly.

### **Option 1: Brute Force**
```python
# Search all vectors
for vector in all_vectors:
    distance = calculate_distance(query_vector, vector)
    
# Problem: 4,772 calculations per query ❌
# Time: ~100ms per search (slow!)
```

### **Option 2: Vector Database (Endee)**
```python
# HNSW graph does smart navigation
result = index.query(query_vector, k=5)

# Benefit: ~12 calculations (log n) per query ✅
# Time: ~5ms per search (20x faster!)
```

**That's why we use Endee!**

---

## **2. What is Endee?**

**Endee = Vector Database with HNSW**

```
┌─────────────────────────────────────┐
│       Endee Server (Docker)         │
│  ┌─────────────────────────────────┐│
│  │    Index (HNSW Graph)           ││
│  │   Layer 2: O--O--O              ││
│  │   Layer 1: O-O-O-O-O            ││
│  │   Layer 0: O-O-O-O-O-O-O-O-O    ││
│  └─────────────────────────────────┘│
│       4,772 vectors indexed         │
│   Persistent storage (Docker vol)   │
└─────────────────────────────────────┘

Listen on: localhost:8080
```

---

## **3. Endee Server Setup**

### **Start Endee with Docker**

```bash
# Navigate to project
cd e:\project\assignment_rag

# Start Endee server
docker compose up -d

# Verify it's running
curl http://localhost:8080/status

# Check logs
docker logs endee -f

# Stop Endee
docker compose down

# Stop but keep data
docker compose stop
```

### **Docker Compose File**

```yaml
version: "3.8"

services:
  endee:
    image: endeeio/endee-server:latest
    container_name: endee
    ports:
      - "8080:8080"
    volumes:
      - endee-data:/data
    environment:
      - ENDEE_PORT=8080
      - ENDEE_HOST=0.0.0.0
    restart: always

volumes:
  endee-data:
    driver: local
```

**Key Points:**
- `ports: 8080:8080` → Access from localhost:8080
- `volumes: endee-data:/data` → Data persists!
- `restart: always` → Auto-restart on failure

---

## **4. Endee Python API**

### **Import and Initialize**

```python
from endee import Endee

# Connect to running Endee server
client = Endee(url="http://localhost:8080")

# Create or get index
index = client.create_index(
    name="documents",
    dimension=384,  # Match embedding dimension!
    metric="cosine"  # Similarity metric
)
```

### **Index Lifecycle**

```python
# 1. Create index (first time)
index = client.create_index(
    name="documents",
    dimension=384,
    metric="cosine"
)

# 2. Get index (already exists)
index = client.get_index("documents")

# 3. Check if exists
if client.index_exists("documents"):
    index = client.get_index("documents")
else:
    index = client.create_index(...)
```

**Important:** `create_index()` returns string, must call `get_index()` to get Index object!

---

## **5. Adding Vectors (Upsert)**

### **Single Vector**

```python
# Add one vector
vector = [0.23, 0.45, -0.12, 0.89, ..., -0.34]  # 384 dims
metadata = {"text": "Machine learning is...", "filename": "doc1.pdf"}

index.upsert(
    vectors=[vector],
    ids=["vec_1"],
    metadatas=[metadata]
)
```

### **Batch Insert (Recommended)**

```python
vectors = [vec1, vec2, vec3, ..., vec4772]
ids = ["vec_1", "vec_2", ..., "vec_4772"]
metadatas = [meta1, meta2, ..., meta4772]

# Our code does batching automatically:
BATCH_SIZE = 1000

for i in range(0, len(vectors), BATCH_SIZE):
    batch_vectors = vectors[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]
    
    index.upsert(
        vectors=batch_vectors,
        ids=batch_ids,
        metadatas=batch_metadatas
    )
```

**Why batch?**
- Endee has 1000 vector per-request limit
- Batch handling prevents API errors
- More efficient

---

## **6. Searching Vectors (Query)**

### **Basic Search**

```python
# Search for top-5 similar vectors
query_vector = [0.21, 0.47, -0.11, ...]  # 384 dims

results = index.query(
    vector=query_vector,
    k=5,         # Return top-5
    ef=100       # HNSW search accuracy
)

# Returns list of results
for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']}")  # Similarity 0-1
    print(f"Metadata: {result['metadata']}")
```

### **HNSW Parameters**

```
ef = Search expansion factor

ef=50:   Fast, less accurate
ef=100:  Balanced ✅ (we use this)
ef=200:  Slow, more accurate

Recommendation: Start with ef=100
```

---

## **7. How Our Code Uses Endee**

### **File: vector_store.py**

```python
from endee import Endee

class VectorStore:
    def __init__(self, endee_url: str = "http://localhost:8080"):
        self.client = Endee(url=endee_url)
        self.index = None
    
    def _ensure_index(self):
        """Create or get index"""
        if not self.client.index_exists("documents"):
            self.client.create_index(
                name="documents",
                dimension=384,
                metric="cosine"
            )
        
        # Get the Index object
        self.index = self.client.get_index("documents")
    
    def add_vectors(self, vectors, metadata, ids):
        """Add vectors in batches"""
        self._ensure_index()
        
        BATCH_SIZE = 1000
        
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = {
                'vectors': vectors[i:i+BATCH_SIZE],
                'ids': ids[i:i+BATCH_SIZE],
                'metadatas': metadata[i:i+BATCH_SIZE]
            }
            
            self.index.upsert(**batch)
    
    def search(self, query_vector, k=5):
        """Search for top-k similar vectors"""
        self._ensure_index()
        
        results = self.index.query(
            vector=query_vector,
            k=k,
            ef=100
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
```

### **File: rag_engine.py (Using VectorStore)**

```python
def add_documents(self, document_chunks):
    """Index documents in Endee"""
    
    # Embed all chunks
    chunks_with_embeddings = [
        {**chunk, 'embedding': self.embedding_engine.embed_text(chunk['text'])}
        for chunk in document_chunks
    ]
    
    # Prepare for Endee
    vectors = [doc['embedding'] for doc in chunks_with_embeddings]
    metadata = [
        {
            'text': doc['text'],
            'filename': doc['metadata'].get('filename', 'Unknown')
        }
        for doc in chunks_with_embeddings
    ]
    ids = [f"doc_{i}" for i in range(len(chunks_with_embeddings))]
    
    # Add to Endee (batching handled inside)
    self.vector_store.add_vectors(
        vectors=vectors,
        metadata=metadata,
        ids=ids
    )

def query(self, question, top_k=5, similarity_threshold=0.30):
    """Retrieve relevant chunks from Endee"""
    
    # Embed question (same model as documents!)
    query_embedding = self.embedding_engine.embed_text(question)
    
    # Search Endee
    results = self.vector_store.search(query_embedding, k=top_k)
    
    # Filter by threshold
    relevant = [
        r for r in results 
        if r['similarity'] >= similarity_threshold
    ]
    
    return relevant
```

---

## **8. Current Endee Status**

### **Our Index Statistics**

```
Index Name: documents
Dimension: 384
Metric: cosine
Total Vectors: 4,772
Data Source: 5 PDFs/MDs
Storage: Docker volume (endee-data)
Persisted: Yes (survives restarts)

Sample Query:
Q: "What is machine learning?"
Query Time: ~5ms
Results Returned: 5 chunks
Similarity Scores: 70%, 68%, 62%, 58%, 52%
```

### **Storage Location**

```
Docker Volume: endee-data
Physical Location (Windows): 
  %ProgramData%\Docker\volumes\endee-data\_data\

To backup:
  docker volume inspect endee-data
  # Note the Mountpoint
  # Backup that directory
```

---

## **9. Endee HNSW Parameters Deep Dive**

### **When Creating Index**

```python
index = client.create_index(
    name="documents",
    dimension=384,        # Embedding dimension
    metric="cosine",      # Similarity metric
    max_m=16,            # Connections per node
    ef_construction=200  # Quality of graph building
)
```

**max_m (Connections):**
```
max_m=5:   Sparse, fast, less accurate
max_m=16:  Balanced ✅ (default)
max_m=48:  Dense, slow, more accurate

Rule: 5-48 is typical range
```

**ef_construction (Build Quality):**
```
ef_construction=100:  Fast building, basic accuracy
ef_construction=200:  Balanced ✅ (default)
ef_construction=400:  Slow building, high accuracy

Higher = better index quality, slower to build
```

---

## **10. Troubleshooting Endee**

### **Issue: Connection Refused**

```
Error: ConnectionError: Failed to connect to http://localhost:8080

Solution:
1. Check Endee is running: docker ps
2. If not: docker compose up -d
3. Wait 5 seconds for startup
4. Test: curl http://localhost:8080/status
```

### **Issue: 1000 Vector Limit**

```
Error: Vector batch too large

Solution:
# Already handled in our code!
# But manually, do this:

vectors = [v1, v2, ..., v5000]

for i in range(0, len(vectors), 1000):
    index.upsert(
        vectors=vectors[i:i+1000],
        ids=ids[i:i+1000],
        metadatas=metadata[i:i+1000]
    )
```

### **Issue: Dimension Mismatch**

```
Error: vector dimension 384 != index dimension 1536

Solution:
# Make sure embedding model matches!

# Check embedding dimension:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(model.get_sentence_embedding_dimension())  # 384

# Create index with same dimension:
index = client.create_index(
    dimension=384  # ← Must match!
)
```

### **Issue: Data Lost After Restart**

```
If using: docker compose down

Solution: Use: docker compose stop
↓
docker compose start (data persists!)

Or add to docker-compose.yml:
  restart: always
```

---

## **11. Endee vs Alternatives**

| Feature | Endee | Pinecone | Weaviate | Milvus |
|---------|-------|----------|----------|--------|
| **Self-Hosted** | ✅ | ❌ | ✅ | ✅ |
| **Free** | ✅ | ❌ | ✅ | ✅ |
| **Setup** | Easy (Docker) | API key | Medium | Hard |
| **Speed** | Fast | Fast | Medium | Fast |
| **HNSW** | ✅ | ✅ | No | ✅ |
| **Best For** | Quick setup | Production | Enterprise | Advanced |

**We chose Endee because:**
- ✅ Self-hosted (no vendor lock-in)
- ✅ Free and open-source
- ✅ Easy Docker setup
- ✅ HNSW = fast search
- ✅ Perfect for RAG

---

## **12. Performance Comparison**

### **Search Speed**

```
Brute Force:   100ms per query (4,772 comparisons)
Endee HNSW:    ~5ms per query (~12 comparisons)

Speedup: 20x faster! ⚡

With 10 queries/day over 1 year:
Brute Force: 3,650 queries × 100ms = 6.1 minutes
Endee:       3,650 queries × 5ms = 18 seconds
Savings: 5m 40s per year!
```

### **Index Building**

```
Index Creation: ~500ms
Upsert 4,772 vectors: ~2 seconds (5 batches of 1000)
Total Setup: ~3 seconds

Fast enough for production!
```

---

## **13. Scaling Considerations**

### **With Current Setup**

```
Max Vectors: 1 million+ (memory permitting)
Query Latency: ~5-50ms (depends on index size)
Memory per Million: ~2GB (rough estimate)
```

### **If You Scale to 1 Million Vectors**

```python
# Still use batching!
BATCH_SIZE = 1000
for i in range(0, 1000000, BATCH_SIZE):
    index.upsert(vectors[i:i+1000], ...)

# Query time still ~50ms (logarithmic growth!)
# HNSW remains efficient at scale
```

---

## **14. Production Checklist**

- [ ] Endee running with `restart: always`
- [ ] Data backed up regularly
- [ ] Index dimension matches embeddings (384)
- [ ] Batching implemented (1000 at a time)
- [ ] Similarity threshold set (0.30)
- [ ] Top-K configured (5)
- [ ] Monitor query times
- [ ] Monitor server memory

---

## **Next Steps**

1. **See it in code:** → **04_CODE_WALKTHROUGH.md**
2. **Deploy it:** → **07_DEPLOYMENT.md**
3. **Fix issues:** → **06_TROUBLESHOOTING.md**

---

## **Quick Reference**

```python
# Initialize
from endee import Endee
client = Endee(url="http://localhost:8080")
index = client.create_index("docs", 384, "cosine")
index = client.get_index("docs")

# Add vectors
index.upsert(vectors=[...], ids=[...], metadatas=[...])

# Search
results = index.query(vector=[...], k=5, ef=100)

# Check result
for r in results:
    print(r['id'], r['score'], r['metadata'])
```

