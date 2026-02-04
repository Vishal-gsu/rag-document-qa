# 🚀 **07 - Deployment & Production Guide**

How to set up and run the system end-to-end.

---

## **Quick Start (5 minutes)**

### **Prerequisites**
- Windows/Mac/Linux
- Docker installed
- Python 3.8+
- NVIDIA GPU (optional, for speed)

### **Step 1: Start Endee (30 seconds)**

```bash
cd e:\project\assignment_rag

# Start Endee server
docker compose up -d

# Verify
curl http://localhost:8080/status
# Should show: {"status": "ok"}
```

### **Step 2: Install Python Dependencies (2 minutes)**

```bash
# Create virtual environment
python -m venv venv

# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install packages
pip install -r requirements.txt
# OR manually:
pip install endee==0.1.8
pip install ollama==0.0.7
pip install streamlit==1.31.0
pip install sentence-transformers==2.2.2
pip install PyPDF2 python-docx beautifulsoup4
```

### **Step 3: Start Ollama (1 minute)**

```bash
# Start Ollama server (if not already running)
ollama serve

# In another terminal, pull a model:
ollama pull phi3  # ~2GB, first time only
```

### **Step 4: Start Web App (30 seconds)**

```bash
# New terminal, venv activated
streamlit run app.py

# Opens in browser at: http://localhost:8501
```

**Done!** 🎉

---

## **Full Deployment Flow**

### **Architecture**

```
┌─────────────────────────────────────────────────────┐
│           Streamlit Web UI                           │
│         (http://localhost:8501)                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Upload | Ask Question | Settings               │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────┬──────────────────────────────────────┘
               │
   ┌───────────┴───────────┬──────────────┐
   │                       │              │
   ▼                       ▼              ▼
┌─────────────┐    ┌──────────────┐  ┌────────────┐
│ Embedding   │    │ RAG Engine   │  │ LLM Manager│
│ Engine      │    │              │  │            │
│             │    │ • Chunk      │  │ • Ollama   │
│ • embed_    │    │ • Embed      │  │ • OpenAI   │
│   text()    │    │ • Search     │  │            │
└─────────────┘    │ • Score      │  └────────────┘
                   └──────────────┘
                          │
                   ┌──────┴───────────┐
                   │                  │
                   ▼                  ▼
            ┌─────────────┐    ┌──────────────┐
            │Vector Store │    │Doc Processor │
            │(Endee i/f)  │    │              │
            │             │    │ • PDF        │
            │ • upsert()  │    │ • DOCX       │
            │ • query()   │    │ • TXT, MD... │
            └──────┬──────┘    └──────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Endee Server (Docker)   │
        │                          │
        │  HNSW Index (4,772 vecs) │
        │  http://localhost:8080   │
        │                          │
        └──────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Persistent Volume        │
        │ (endee-data)             │
        │ Data survives restarts!  │
        └──────────────────────────┘
```

---

## **Detailed Deployment Steps**

### **Step 1: Environment Setup**

```bash
# Create project directory
mkdir -p e:\project\assignment_rag
cd e:\project\assignment_rag

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows Command Prompt
# OR
venv\Scripts\Activate.ps1  # Windows PowerShell (might need: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned)

# macOS/Linux:
source venv/bin/activate
```

**Verify venv is active:**
```bash
python --version
which python  # Should show venv path
```

### **Step 2: Install Dependencies**

**Option A: From requirements.txt (Recommended)**

```bash
# Create requirements.txt if it doesn't exist:
echo "endee==0.1.8" > requirements.txt
echo "ollama==0.0.7" >> requirements.txt
echo "streamlit==1.31.0" >> requirements.txt
echo "sentence-transformers==2.2.2" >> requirements.txt
echo "torch==2.0.0" >> requirements.txt  # CPU version by default
echo "PyPDF2==3.0.1" >> requirements.txt
echo "python-docx==0.8.11" >> requirements.txt
echo "beautifulsoup4==4.12.0" >> requirements.txt
echo "lxml==4.9.2" >> requirements.txt

# Install all
pip install -r requirements.txt
```

**Option B: Manual Installation**

```bash
pip install endee==0.1.8
pip install ollama==0.0.7
pip install streamlit==1.31.0
pip install sentence-transformers==2.2.2
pip install torch
pip install PyPDF2
pip install python-docx
pip install beautifulsoup4
```

**For GPU Support (CUDA/NVIDIA):**

```bash
# Remove CPU torch first
pip uninstall torch

# Install GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU:
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### **Step 3: Docker & Endee Setup**

```bash
# Verify Docker is installed
docker --version
# Should show: Docker version XX.XX

# Create docker-compose.yml if it doesn't exist:
cat > docker-compose.yml << 'EOF'
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
EOF

# Start Endee
docker compose up -d

# Wait 5 seconds for startup
sleep 5

# Verify it's running
curl http://localhost:8080/status
# Should show: {"status":"ok"}

# Check logs (if needed)
docker logs endee

# Stop Endee later
docker compose stop

# Start again (data persists)
docker compose start
```

### **Step 4: Ollama Setup**

```bash
# Download Ollama from https://ollama.ai
# Or install via package manager

# Start Ollama server
ollama serve

# In another terminal, pull models:
ollama pull phi3      # ~2GB (recommended)
ollama pull mistral   # ~4GB (better quality)
ollama pull neural-chat  # ~4GB

# Verify models installed
ollama list

# Test generation
ollama run phi3 "What is machine learning?"

# For GPU support:
# Ollama auto-detects CUDA/Metal support
# Check: nvidia-smi should show Ollama process
```

### **Step 5: Run Application**

```bash
# Make sure venv is activated
# Make sure Endee is running (docker ps)
# Make sure Ollama is running (ollama serve in another terminal)

# Run Streamlit app
streamlit run app.py

# Opens browser at: http://localhost:8501
# Stop with: Ctrl+C
```

---

## **Production Considerations**

### **1. Auto-Start Services**

**Docker (auto-restart):**
```yaml
# Already in docker-compose.yml:
restart: always
# Endee restarts if crashes
```

**Ollama auto-start (Windows):**
```bash
# Create batch file: start_ollama.bat
@echo off
cd "C:\Users\<YourUser>\AppData\Local\Ollama"
ollama.exe serve

# Schedule with Task Scheduler to run at startup
```

**Streamlit auto-start:**
```bash
# Create batch file: start_app.bat
@echo off
cd e:\project\assignment_rag
call venv\Scripts\activate
streamlit run app.py --server.headless true
```

### **2. Resource Management**

**CPU/Memory Limits:**
```yaml
# Limit Endee memory in docker-compose.yml:
services:
  endee:
    mem_limit: 4g        # Max 4GB RAM
    cpus: '2.0'          # Max 2 CPU cores
    volumes: ...
```

**GPU Memory:**
```bash
# Check available VRAM
nvidia-smi

# Limit GPU usage for Ollama
# Set in environment:
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 only
ollama serve
```

### **3. Monitoring**

```bash
# Check service health
docker ps  # Is Endee running?
curl http://localhost:8080/status

# Check Ollama
ollama list  # Models available?
curl http://localhost:11434/api/tags

# Check Streamlit processes
ps aux | grep streamlit  # Linux
tasklist | findstr streamlit  # Windows
```

### **4. Logging**

```bash
# Endee logs
docker logs endee -f
docker logs endee --tail 100  # Last 100 lines

# Streamlit logs
# Appear in terminal when running

# Application logging (optional)
# Add to app.py:
import logging
logging.basicConfig(filename='rag.log', level=logging.DEBUG)
logging.info("App started")
```

### **5. Backup Strategy**

```bash
# Backup Endee data
docker volume inspect endee-data
# Note the Mountpoint

# Linux/macOS:
tar czf endee-backup.tar.gz /var/lib/docker/volumes/endee-data/_data

# Windows PowerShell:
Copy-Item -Path "C:\ProgramData\Docker\volumes\endee-data\_data" -Destination "E:\backups\endee-data" -Recurse

# Schedule weekly backups (task scheduler)
```

---

## **Configuration for Different Scenarios**

### **Scenario 1: Development (Your Setup)**

```python
# config.py
LLM_PROVIDER = "ollama_gpu"
LLM_MODEL = "phi3"
SIMILARITY_THRESHOLD = 0.30
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 800
```

**Pros:** Free, fast, works offline
**Cons:** Requires GPU, responses slower than cloud

### **Scenario 2: Production with OpenAI**

```python
# config.py
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-3.5-turbo"
SIMILARITY_THRESHOLD = 0.25
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1000

# Environment variable:
export OPENAI_API_KEY="sk-..."
```

**Pros:** Fastest, best quality
**Cons:** Costs $$, requires internet

### **Scenario 3: Edge Device (Raspberry Pi)**

```python
# config.py
LLM_PROVIDER = "ollama_cpu"  # No GPU
LLM_MODEL = "neural-chat"    # Lighter model
SIMILARITY_THRESHOLD = 0.40
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 300         # Shorter responses

# Reduce chunk size for speed
CHUNK_SIZE = 500
```

**Pros:** Runs anywhere, offline
**Cons:** Very slow responses (30+ seconds)

### **Scenario 4: High-Volume Production**

```python
# config.py
LLM_PROVIDER = "openai"  # Most reliable
LLM_MODEL = "gpt-3.5-turbo"
SIMILARITY_THRESHOLD = 0.35
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1000

# Increase concurrency
BATCH_SIZE = 5000  # Larger batches

# Add caching
# Implement query caching for repeated questions
```

---

## **Performance Tuning**

### **Speed Optimization**

```python
# Current settings (balanced):
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# For 10x speed increase (quality trade-off):
CHUNK_SIZE = 2000       # Fewer chunks
TOP_K_RESULTS = 3       # Return fewer
SIMILARITY_THRESHOLD = 0.40  # Stricter filter
LLM_MAX_TOKENS = 300    # Shorter responses

# Timings:
# Default: ~2-3 seconds per query
# Optimized: ~0.5-1 second per query
```

### **Quality Optimization**

```python
# For best possible answers (slower):
CHUNK_SIZE = 500        # More chunks, better context
CHUNK_OVERLAP = 250     # Heavy overlap
TOP_K_RESULTS = 10      # Get more candidates
SIMILARITY_THRESHOLD = 0.20  # Keep more results
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Better embeddings
LLM_TEMPERATURE = 0.2   # Very focused

# Timings:
# Optimized: ~5-10 seconds per query
```

---

## **Scaling Guide**

### **Scaling Vectors (Indexing more documents)**

```
Current: 4,772 vectors (2,386 chunks)
Potential: 1,000,000+ vectors

Bottleneck: VRAM

Solution 1: Increase server RAM
# docker-compose.yml
mem_limit: 16g  # 16GB instead of 4GB

Solution 2: Use smaller embedding dimension
# config.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384D (current)
EMBEDDING_MODEL = "all-distilroberta-v1"  # 384D but faster

Solution 3: Distributed indexing
# Split documents across multiple Endee instances
# Use load balancer to route queries
```

### **Scaling Queries (More concurrent users)**

```
Current: Single user
Problem: One user at a time, Streamlit single-threaded

Solution 1: Add Streamlit Sharing (cloud)
# Deploy to streamlit.io cloud

Solution 2: Use Streamlit Server API
# Run behind proxy with connection pooling

Solution 3: Rewrite with FastAPI
# Multi-threaded Python backend
# Replace Streamlit with React frontend
```

---

## **Deployment Checklist**

Before going to production:

```
✅ Endee running with restart: always
✅ Docker volume persisted (endee-data)
✅ Regular backups scheduled (weekly)
✅ Ollama models cached locally
✅ API keys secured (environment variables)
✅ Firewall configured (block port 8080 externally)
✅ Logging enabled (docker logs, app logs)
✅ Health checks implemented
✅ Error handling tested
✅ Performance monitored
✅ Documentation updated
✅ Team trained on operations
```

---

## **Troubleshooting Common Deployment Issues**

### **App won't start**

```bash
# Check dependencies
pip list

# Check venv activation
which python

# Check Python version
python --version

# Try clean install
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

### **Endee not persisting data**

```bash
# Check volume is named volume (not temp)
docker volume ls | grep endee

# Check docker-compose.yml has:
volumes:
  - endee-data:/data

# NEVER use:
volumes:
  - /tmp/endee:/data  # Lost after restart!
```

### **Models keep downloading**

```bash
# Models should be cached
# Check Ollama model directory
ls ~/.ollama/models  # macOS/Linux
dir %USERPROFILE%\.ollama\models  # Windows

# Pre-download models on setup
ollama pull phi3
ollama pull mistral
```

---

## **Network Security**

### **Running Behind Firewall**

```bash
# Allow only local access (default)
# http://localhost:8501  # Streamlit
# http://localhost:8080  # Endee (internal only)
# http://localhost:11434 # Ollama (internal only)

# Expose to network (careful!):
# docker-compose.yml
ports:
  - "0.0.0.0:8080:8080"  # ⚠️ Opens to network

# Better: Use nginx reverse proxy
```

### **API Key Security**

```bash
# NEVER hardcode keys!
# Use environment variables:

export OPENAI_API_KEY="sk-..."
export ENDEE_AUTH_TOKEN="..."

# In code:
import os
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

# In production: Use secrets manager
# AWS Secrets Manager, Azure Key Vault, etc.
```

---

## **Next Steps After Deployment**

1. ✅ System running? Verify in browser
2. 📊 Upload test documents
3. 🔍 Test queries
4. 📈 Monitor performance
5. 🔧 Tune parameters as needed
6. 📚 Document your setup
7. 🎓 Train team on operation

---

## **Quick Reference Commands**

```bash
# Endee
docker compose up -d           # Start
docker compose stop            # Stop (keeps data)
docker compose down            # Stop (deletes containers)
docker logs endee -f           # View logs
curl http://localhost:8080/status  # Health check

# Ollama
ollama serve                   # Start server
ollama pull phi3               # Download model
ollama list                    # Show models
ollama run phi3 "prompt"       # Test generation

# Streamlit
streamlit run app.py           # Start app
streamlit config show          # Show config
streamlit cache clear          # Clear cache

# Python venv
python -m venv venv            # Create
source venv/bin/activate       # Activate (macOS/Linux)
venv\Scripts\activate          # Activate (Windows)
pip install -r requirements.txt # Install

# Useful
ps aux | grep <service>        # Check if running (Linux)
tasklist | findstr <service>   # Check if running (Windows)
```

---

## **Success Indicators**

When your system is correctly deployed:

✅ **Endee:**
- Container running: `docker ps` shows endee
- Health check passes: `curl http://localhost:8080/status`
- Data persists: Restart and vectors still there

✅ **Ollama:**
- Server responding: `curl http://localhost:11434/api/tags`
- Model cached: `ollama list` shows phi3
- Generation works: Can ask questions

✅ **Streamlit:**
- App loads: Browser shows interface
- Upload works: Can upload documents
- Query works: Can ask questions and get answers
- Timing acceptable: 2-3 seconds per query

**Congratulations!** 🎉 Your RAG system is production-ready!

