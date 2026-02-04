# 🛠️ **06 - Troubleshooting Guide**

Solutions to every error we encountered and how to fix them.

---

## **Endee & Docker Issues**

### **Issue 1: Endee Connection Refused**

**Error:**
```
ConnectionError: Failed to connect to http://localhost:8080
```

**Diagnosis:**
```bash
# Check if Endee container is running
docker ps | grep endee

# If not listed, it's down
```

**Solutions:**

```bash
# Solution 1: Start Endee
docker compose up -d

# Wait for startup (takes 5-10 seconds)
sleep 5

# Solution 2: Verify it's running
curl http://localhost:8080/status
# Should return: {"status": "ok"}

# Solution 3: Check logs
docker logs endee -f

# Solution 4: If still broken, restart
docker compose restart endee

# Solution 5: Nuclear option - fresh start
docker compose down
docker volume rm endee-data  # ⚠️ DELETES INDEXED VECTORS!
docker compose up -d
```

**Prevention:**
- Add `restart: always` to docker-compose.yml
- Run `docker compose up -d` at boot
- Check health regularly

---

### **Issue 2: Port Already in Use**

**Error:**
```
Error response from daemon: Bind for 0.0.0.0:8080 failed: port is already allocated
```

**Diagnosis:**
```bash
# Find what's using port 8080
netstat -ano | findstr :8080  # Windows

# Kill process (if safe)
taskkill /PID <PID> /F
```

**Solutions:**

```bash
# Solution 1: Use different port
# Edit docker-compose.yml
ports:
  - "8081:8080"  # Use 8081 instead

# Solution 2: Stop whatever's using 8080
# Check what it is first!
docker ps  # If Docker container, stop it
docker stop <container_name>

# Solution 3: Check for multiple Endee instances
docker ps -a | grep endee
# If multiple, remove old ones:
docker rm endee-old
```

---

### **Issue 3: Data Lost After Restart**

**Symptom:**
```
Restarted Docker, all indexed vectors gone! 😱
```

**Root Cause:**
```bash
# Used this (WRONG):
docker compose down  # Deletes containers AND volumes

# Should use this (RIGHT):
docker compose stop  # Just stops, keeps data
```

**Prevention:**

```bash
# Ensure volume persists in docker-compose.yml:
volumes:
  - endee-data:/data  # Named volume (persists) ✅
  
# NOT this:
volumes:
  - /tmp/endee:/data  # Temp directory (deleted!)

# Verify volume exists:
docker volume ls | grep endee-data

# Backup volume regularly:
docker volume inspect endee-data
# Note the Mountpoint and backup that directory
```

**Recovery:**
```bash
# If you have backups:
1. Stop Endee: docker compose stop
2. Restore backup to volume mount point
3. Restart: docker compose up -d

# If no backup:
1. Re-index documents
2. More careful next time!
```

---

## **Vector & Embedding Issues**

### **Issue 4: Dimension Mismatch**

**Error:**
```
ValueError: vector dimension 384 != index dimension 1536
```

**Root Cause:**
```python
# ❌ WRONG - embedding model outputs 384D
embedding = SentenceTransformer('all-MiniLM-L6-v2').encode(text)
# Result: 384 dimensions

# But index expects 1536D
client.create_index(dimension=1536)
# ↑ Mismatch! ❌
```

**Solution:**

```bash
# Step 1: Check your embedding dimension
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f'Dimension: {model.get_sentence_embedding_dimension()}')
"
# Output: Dimension: 384

# Step 2: Recreate index with correct dimension
python -c "
from endee import Endee
client = Endee()
if client.index_exists('documents'):
    # Delete old index first
    # (Endee doesn't have delete_index, so we're stuck)
    # Option: Recreate with docker volume rm
    pass
else:
    client.create_index(name='documents', dimension=384)
"

# Step 3: Verify config.py matches
# config.py should have:
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

**Prevention:**
- Always verify dimension before creating index
- Use config.py for all dimensions
- Match embedding model everywhere

---

### **Issue 5: Vector Batch Too Large**

**Error:**
```
BadRequest: Vector batch too large
```

**Root Cause:**
```python
# ❌ WRONG - trying to add all 4,772 at once
vectors = [v1, v2, ..., v4772]
index.upsert(vectors=vectors)  # ❌ Endee limit: 1000

# ✅ RIGHT - batch into chunks of 1000
for i in range(0, len(vectors), 1000):
    batch = vectors[i:i+1000]
    index.upsert(vectors=batch)
```

**Solution:**

```python
# Already implemented in vector_store.py!
# But if you get this error:

def add_vectors_safe(self, vectors, metadata, ids):
    BATCH_SIZE = 1000
    
    for i in range(0, len(vectors), BATCH_SIZE):
        batch_vectors = vectors[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_metadata = metadata[i:i+BATCH_SIZE]
        
        try:
            self.index.upsert(
                vectors=batch_vectors,
                ids=batch_ids,
                metadatas=batch_metadata
            )
        except Exception as e:
            print(f"Batch {i//BATCH_SIZE} failed: {e}")
            continue
```

**Prevention:**
- Always batch at 1000 vector max
- Check `add_vectors()` implementation

---

## **LLM & Generation Issues**

### **Issue 6: Ollama Not Running**

**Error:**
```
ConnectionError: Failed to connect to http://localhost:11434
```

**Diagnosis:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If error, Ollama is down
```

**Solution:**

```bash
# Windows: Start Ollama
"C:\Users\<YourUser>\AppData\Local\Ollama\ollama.exe" serve

# Or if installed as service:
# Just run `ollama serve` in terminal

# Verify it's working:
ollama list  # Should show phi3, mistral, etc.

# If no models:
ollama pull phi3    # Download model
ollama pull mistral # Download model

# Test generation:
ollama run phi3 "What is machine learning?"
```

**Prevention:**
- Start Ollama before running app
- Use systemd/Windows service to auto-start
- Keep models cached (don't delete!)

---

### **Issue 7: Model Too Slow**

**Symptom:**
```
Waiting 60+ seconds for response...
```

**Diagnosis:**
```python
# Check which LLM is active:
print(config.LLM_PROVIDER)  # "ollama_gpu"?

# If using CPU:
# Ollama CPU: ~2 tokens/second
# Ollama GPU: ~20 tokens/second
# Speedup: 10x slower!
```

**Solution:**

```python
# Option 1: Ensure GPU is enabled
# Check: Does your GPU have CUDA support?
nvidia-smi  # If shows GPU info, CUDA available

# Option 2: Use faster model
ollama pull mistral  # Faster than phi3
# config.py: LLM_MODEL = "mistral"

# Option 3: Reduce tokens
config.LLM_MAX_TOKENS = 300  # Instead of 800
# Faster but shorter answers

# Option 4: Increase temperature
config.LLM_TEMPERATURE = 0.7  # Instead of 0.4
# Weird but makes model less careful = faster
# NOT RECOMMENDED for accuracy

# Option 5: Use OpenAI (fastest!)
# If you have API key:
export OPENAI_API_KEY="sk-..."
config.LLM_PROVIDER = "openai"
```

**Example Timing:**
```
Ollama GPU (phi3):    ~10-20 seconds per response ✅
Ollama CPU (phi3):    ~60-120 seconds per response ⚠️
Ollama CPU (mistral): ~30-60 seconds per response ⚠️ (faster)
OpenAI (gpt-3.5):     ~3-5 seconds per response ✅ (fastest)
```

---

### **Issue 8: Model Hallucinating**

**Symptom:**
```
Q: "Who is the author?"
A: "I believe it might be Stephen Marsland, 
    although I'm not entirely certain. 
    Perhaps if you had more information..."
```

**Root Cause:**
```python
# ❌ WRONG - Temperature too high
temperature = 0.7  # Creative, wandering

# ✅ RIGHT - Temperature lower
temperature = 0.4  # Focused, confident
```

**Solutions:**

```python
# Solution 1: Decrease temperature
config.LLM_TEMPERATURE = 0.4  # From 0.7

# Solution 2: Improve prompt
# OLD:
"Answer the user's question"

# NEW:
"Using ONLY the provided context, answer directly.
Be confident. Do NOT add follow-up questions or caveats."

# Solution 3: Improve retrieval
# Ensure similarity scores are high (60%+)
config.SIMILARITY_THRESHOLD = 0.30
config.TOP_K_RESULTS = 5

# Solution 4: Use better model
# Mistral > Phi3 > TinyLlama for accuracy
ollama pull mistral
```

**Prevent:** Use consistent temperature (0.4) for all queries

---

### **Issue 9: Request Timed Out**

**Error:**
```
TimeoutError: Request timed out waiting for response
```

**Root Cause:**
```python
# Timeout too short for slow model
timeout = 30  # Ollama GPU needs 15+ seconds

# Or GPU is overloaded
# Or model is stuck
```

**Solution:**

```python
# Solution 1: Increase timeout
config.LLM_TIMEOUT = 180  # 3 minutes (already default)

# Solution 2: Check if GPU is available
nvidia-smi
# If not showing GPU info, only CPU available

# Solution 3: Reduce concurrent requests
# Only run 1 Streamlit session at a time

# Solution 4: Restart Ollama
ollama serve  # Kill and restart

# Solution 5: Use simpler query
# Complex questions might take longer
Q: "Who is the author?"          # Fast
Q: "What is the history of..."  # Slower (longer context)
```

---

## **Retrieval Issues**

### **Issue 10: No Results Found**

**Symptom:**
```
"No relevant results found"
```

**Diagnosis:**
```python
# Check similarity scores:
# Results might be below threshold

# Add debug output:
results = rag_engine.query(question)
for r in results:
    print(f"Similarity: {r['similarity']}")
# If all < 0.30, problem is retrieval
```

**Solutions:**

```python
# Solution 1: Lower threshold
config.SIMILARITY_THRESHOLD = 0.20  # From 0.30

# Solution 2: Increase TOP_K
config.TOP_K_RESULTS = 10  # From 5
# Get more candidates to filter

# Solution 3: Check if documents indexed
# Verify:
print(f"Index contains {count_vectors()} vectors")
# Should be 2,386+ vectors

# Solution 4: Increase chunk size
# Larger chunks = better embeddings = higher similarity
config.CHUNK_SIZE = 2000  # From 1000
# Then re-index all documents

# Solution 5: Try rephrasing question
Q: "What is machine learning?"    # Good
Q: "ML"                           # Too short
Q: "Tell me about the history..." # Long but clear
```

**Debug Code:**
```python
# Add to app.py to see all results:
if st.checkbox("Show debug info"):
    st.write("All results:")
    for r in results:
        st.write(f"{r['id']}: {r['similarity']:.2%} similarity")
```

---

### **Issue 11: Wrong Results**

**Symptom:**
```
Q: "What is machine learning?"
A: [Results about deep learning, neural networks, etc.]
# Related but not exact!
```

**Root Cause:**
- Chunk size too small (context lost)
- Similarity threshold too low (noisy results)
- Embedding model limitations

**Solution:**

```python
# Solution 1: Increase chunk size
config.CHUNK_SIZE = 1500  # More context

# Solution 2: Increase overlap
config.CHUNK_OVERLAP = 300  # Less info loss

# Solution 3: Increase threshold
config.SIMILARITY_THRESHOLD = 0.50  # More selective

# Solution 4: Use better embedding model
# Current: all-MiniLM-L6-v2 (good balance)
# Better: all-mpnet-base-v2 (more accurate, slower)
# In embedding_engine.py:
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Then re-index all documents

# Solution 5: Improve document chunking
# Add better chunk boundaries (section headers, paragraphs)
# Not just character-based splitting
```

---

## **App & UI Issues**

### **Issue 12: Export Data Not Working**

**Error:**
```
NameError: name 'export_data' is not defined
```

**Root Cause:**
```python
# ❌ WRONG - Indentation
if st.button("Export"):
    st.write("Exporting...")
st.download_button(...)  # Outside if block ❌

# ✅ RIGHT - Proper indentation
if st.button("Export"):
    export_data = ...
    st.download_button(...)  # Inside if block ✅
```

**Solution:**
- Already fixed in app.py
- Check indentation if error reappears

---

### **Issue 13: App Crashes on Upload**

**Error:**
```
Unexpected error while running the app
```

**Diagnosis:**
```python
# Add error handling:
try:
    chunks = processor.process_document(file)
    rag_engine.add_documents(chunks)
except Exception as e:
    st.error(f"Upload failed: {str(e)}")
    print(f"Full error: {e}")  # Check console
```

**Solutions:**

```python
# Solution 1: Check file format
# Some PDFs are images (OCR needed)
# Check if file is actually a PDF, not image

# Solution 2: Check file size
# Very large files might timeout
config.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Solution 3: Check encoding
# Text files might be UTF-16, not UTF-8

# Solution 4: Update dependencies
pip install --upgrade pdf2image PyPDF2

# Solution 5: Restart Streamlit
# Stop app and run again
streamlit run app.py
```

---

### **Issue 14: Streamlit Port Already in Use**

**Error:**
```
Error: Address already in use (port 8501)
```

**Solution:**

```bash
# Solution 1: Use different port
streamlit run app.py --server.port 8502

# Solution 2: Kill existing Streamlit
# Find the process:
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill it:
kill <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Solution 3: Wait a minute
# Streamlit sometimes takes time to release port
```

---

## **Python & Environment Issues**

### **Issue 15: Module Not Found**

**Error:**
```
ModuleNotFoundError: No module named 'endee'
```

**Solution:**

```bash
# Install missing package
pip install endee==0.1.8

# Verify installation
python -c "import endee; print(endee.__version__)"

# List all installed packages
pip list | grep endee

# If still missing, try with python directly
python -m pip install endee==0.1.8

# Check virtual environment
which python  # Should point to venv/bin/python
```

### **Issue 16: Version Conflicts**

**Error:**
```
The following packages have unmet dependencies...
```

**Solution:**

```bash
# Install exact versions we tested
pip install -r requirements.txt

# If requirements.txt missing, install manually:
pip install endee==0.1.8
pip install ollama==0.0.7
pip install streamlit==1.31.0
pip install sentence-transformers==2.2.2
pip install pytorch (GPU version if needed)

# Clear pip cache if weird errors
pip cache purge
pip install --no-cache-dir endee==0.1.8
```

---

## **Performance Issues**

### **Issue 17: Indexing Too Slow**

**Symptom:**
```
Indexing 2,386 chunks taking 5+ minutes...
```

**Solution:**

```python
# Solution 1: Use batch embedding
# Current: Already uses batch! ✅
# If rewriting:
vectors = embedding_engine.embed_batch(texts)  # Fast ✅
# NOT:
vectors = [embedding_engine.embed_text(t) for t in texts]  # Slow

# Solution 2: Reduce chunk size
# Smaller chunks = fewer vectors = faster
config.CHUNK_SIZE = 500  # From 1000
# But worse retrieval quality

# Solution 3: Increase GPU memory
# If using GPU, ensure enough VRAM allocated

# Solution 4: Use streaming
# Stream results instead of batch all at once
```

**Expected Timing:**
```
Embedding 2,386 chunks:  ~1-2 seconds (GPU) ✓
Uploading to Endee:      ~2-3 seconds (5 batches)
Total indexing:          ~3-5 seconds ✓
```

---

## **Debug Checklist**

When something breaks, check these in order:

```
1. ✅ Is Endee running?
   curl http://localhost:8080/status

2. ✅ Is Ollama running? (if using)
   curl http://localhost:11434/api/tags

3. ✅ Are documents indexed?
   Query Endee directly for vector count

4. ✅ Check similarity scores
   Add debug output to app.py

5. ✅ Check LLM response
   Test Ollama directly:
   ollama run phi3 "test"

6. ✅ Check logs
   docker logs endee -f
   streamlit run app.py  (see console output)

7. ✅ Restart everything
   docker compose down && docker compose up -d
   streamlit run app.py
```

---

## **Still Broken?**

### **Nuclear Options**

```bash
# Clean slate - delete all data
docker compose down
docker volume rm endee-data
rm -rf __pycache__
pip install --upgrade pip setuptools wheel
pip install --force-reinstall endee==0.1.8

# Restart
docker compose up -d
streamlit run app.py
```

### **Get Help**

1. Run debug script:
```python
# debug.py
from endee import Endee
from sentence_transformers import SentenceTransformer

print("1. Checking Endee...")
try:
    client = Endee("http://localhost:8080")
    print(f"   ✅ Connected. Indexes: {client.index_list()}")
except Exception as e:
    print(f"   ❌ Endee error: {e}")

print("2. Checking embedding model...")
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"   ✅ Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"   ❌ Model error: {e}")

print("3. Checking Ollama...")
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags")
    print(f"   ✅ Ollama running. Models: {len(r.json()['models'])}")
except Exception as e:
    print(f"   ❌ Ollama error: {e}")
```

2. Share output of this script
3. Check logs: `docker logs endee`

---

## **Next Steps**

- ✅ Issue fixed? Great! Continue with **07_DEPLOYMENT.md**
- ❓ Still broken? Check the **Nuclear Options** section
- 📞 Need help? Create GitHub issue with debug script output

