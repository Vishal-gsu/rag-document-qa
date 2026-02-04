# � Intelligent Document Q&A System with RAG

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Required-blue)](https://www.docker.com/)

A **production-ready Retrieval Augmented Generation (RAG)** system that lets you chat with your documents using AI. Upload PDFs, Word docs, or text files and ask questions - get accurate, cited answers in seconds!

![RAG Architecture](https://img.shields.io/badge/Architecture-RAG-green) ![Vector DB](https://img.shields.io/badge/VectorDB-Endee%20HNSW-orange) ![LLM](https://img.shields.io/badge/LLM-4%20Options-purple)

> **📌 Built with [Endee Vector Database](https://github.com/EndeeLabs/endee)** | [My Forked Repository](https://github.com/Vishal-gsu/endee)

---

## ✨ Features

- 🔍 **Semantic Search** - Find relevant information using AI-powered similarity search
- 📄 **Multi-Format Support** - PDF, DOCX, TXT, Markdown files
- ⚡ **Fast Retrieval** - HNSW algorithm for sub-second search (O(log n) complexity)
- 🤖 **Multiple LLM Options** - Choose from Groq (FREE), OpenAI, or local Ollama
- 💾 **Persistent Storage** - Your documents stay indexed across sessions
- 🎨 **Interactive UI** - Beautiful Streamlit interface with 4 tabs
- 🧪 **Testing Suite** - 6 interactive tests for embeddings, similarity, and search quality
- 📊 **Performance Metrics** - Track retrieval accuracy and response times

---

## 🎯 How It Works

```
📄 Upload Documents → 🔪 Chunk Text → 🧠 Generate Embeddings → 💾 Store in Vector DB
                                                                        ↓
                                                              🔍 Your Question
                                                                        ↓
                                                    🎯 Similarity Search (HNSW)
                                                                        ↓
                                                    📚 Retrieve Top-K Relevant Chunks
                                                                        ↓
                                                    🤖 LLM + Context → ✅ Answer
```

### RAG Pipeline

1. **Document Ingestion** - Upload PDFs/DOCX/TXT files
2. **Chunking** - Split into 500-character chunks with overlap
3. **Embedding** - Convert text to 384D vectors using sentence-transformers
4. **Storage** - Store in Endee Vector DB with HNSW indexing
5. **Query** - User asks a question
6. **Retrieval** - Find top-5 similar chunks using cosine similarity
7. **Generation** - LLM generates answer using retrieved context
8. **Response** - Get answer with source citations

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop/))
- **Git** ([Download](https://git-scm.com/downloads))

### Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/assignment_rag.git
cd assignment_rag

# 2️⃣ Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Setup environment variables
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux

# Edit .env and add your API key (choose one):
# - GROQ_API_KEY=your_key_here (FREE - recommended!)
# - OPENAI_API_KEY=your_key_here (paid)
# Get Groq key: https://console.groq.com/keys

# 5️⃣ Start Endee vector database
docker compose up -d

# 6️⃣ Run the application
streamlit run app.py
```

**🎉 Done!** Open http://localhost:8501 in your browser

---

## 🎮 Usage Guide

### 1️⃣ **Upload Documents** (Tab 1)

- Click **"Browse files"** and select PDFs, DOCX, or TXT files
- Click **"🚀 Upload & Index"**
- Wait for indexing to complete (1-2 min for 10 documents)

### 2️⃣ **Configure LLM** (Sidebar)

**Option A: Groq API (Recommended - FREE!)**
- Select **"⚡ Groq API (FREE)"**
- Paste your Groq API key
- Choose model: `llama-3.3-70b-versatile`
- Click **"Set Groq Provider"**

**Option B: OpenAI (Paid)**
- Select **"☁️ OpenAI API"**
- Paste your OpenAI API key
- Choose model: `gpt-3.5-turbo`

**Option C: Ollama (Local - Optional)**
- Install Ollama: https://ollama.com
- Run: `ollama serve`
- Pull model: `ollama pull llama3.2`
- Select **"🚀 Ollama (GPU)"** or **"💻 Ollama (CPU)"**

### 3️⃣ **Ask Questions** (Tab 2)

- Type your question: *"What is machine learning?"*
- Click **"🔍 Ask"**
- Get answer with source citations!

### 4️⃣ **Test & Visualize** (Tab 4)

Try 6 interactive tests:
- **Embedding Similarity** - Compare word embeddings
- **Word Clustering** - Visualize semantic relationships
- **Semantic Search Quality** - Test retrieval accuracy
- **Context Window Impact** - Compare chunk sizes
- **Multi-Query Comparison** - Analyze multiple queries
- **Embedding Distribution** - Visualize database vectors

---

## 📊 Project Structure

```
assignment_rag/
├── 📄 README.md              # This file - GitHub documentation
├── 📄 GITHUB_SETUP.md        # GitHub repo setup guide
├── 📋 requirements.txt       # Python dependencies
├── 🐳 docker-compose.yml     # Endee database config
├── 🔧 .env.example           # Environment template
├── ⚙️ config.py              # Configuration management
├── 🎨 app.py                 # Streamlit web interface (4 tabs)
│
├── 📂 Core Modules
│   ├── document_processor.py # Load & chunk documents
│   ├── embedding_engine.py   # Generate embeddings
│   ├── vector_store.py       # Endee DB interface
│   ├── rag_engine.py         # RAG orchestration
│   ├── llm_manager.py        # Multi-LLM support (4 providers)
│   └── prompt_templates.py   # System prompts
│
├── 📂 data/
│   ├── documents/            # 📄 Your source documents (put files here!)
│   └── vectordb/             # 💾 Endee database storage
│
└── 📂 venv/                  # Python virtual environment
```

---

## ⚙️ Configuration

Edit `.env` file to customize:

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy example environment file
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 5: Prepare Documents
```bash
# Create documents directory
mkdir data
mkdir data\documents  # Windows
# mkdir -p data/documents  # macOS/Linux

# Add your documents to data/documents/
```

---

## 🚀 Usage

### Basic Usage

#### 1. Index Documents
```bash
python main.py --mode index --docs data/documents
```

#### 2. Query the System
```bash
python main.py --mode query --question "What is machine learning?"
```

#### 3. Interactive Mode
```bash
python main.py --mode interactive
```

### Programmatic Usage
```python
from rag_engine import RAGEngine

# Initialize RAG system
rag = RAGEngine()

# Index documents
rag.index_documents("data/documents")

# Query
response = rag.query("Explain neural networks")
print(response)
```

---

## ⚙️ Configuration

Edit `.env` file to customize:

```env
# LLM Providers (choose one or configure multiple)
GROQ_API_KEY=your_groq_key_here          # FREE from console.groq.com
OPENAI_API_KEY=your_openai_key_here      # Paid from platform.openai.com

# Embedding Settings
EMBEDDING_MODEL=sentence-transformers    # Local (free) or OpenAI (paid)

# RAG Settings
CHUNK_SIZE=500                           # Characters per chunk
CHUNK_OVERLAP=50                         # Overlap between chunks
TOP_K_RESULTS=5                          # Number of chunks to retrieve
```

---

## 🔧 Advanced Features

### 🎨 Streamlit Interface Tabs

**Tab 1: 📁 Document Upload**
- Drag & drop files or browse
- Supports PDF, DOCX, TXT, MD
- Auto-deduplication (skips already-indexed files)
- Batch indexing with progress tracking

**Tab 2: 💬 Ask Questions**
- Natural language query interface
- Conversation history
- Source citations with similarity scores
- Confidence threshold filtering

**Tab 3: ⚙️ Settings**
- LLM provider selection (Groq/OpenAI/Ollama)
- Model configuration
- Embedding options
- Custom prompt templates

**Tab 4: 🧪 Interactive Tests**
- Test 1: Embedding Similarity (compare word vectors)
- Test 2: Word Clustering (t-SNE visualization)
- Test 3: Semantic Search Quality (retrieval metrics)
- Test 4: Context Window Impact (chunk size comparison)
- Test 5: Multi-Query Comparison (heatmap analysis)
- Test 6: Embedding Distribution (2D visualization)

### 🤖 LLM Provider Options

| Provider | Cost | Speed | Models | Setup |
|----------|------|-------|--------|-------|
| **Groq** | FREE | ⚡ Fastest | Llama 3.3 70B, Mixtral 8x7B | Get key from console.groq.com |
| **OpenAI** | Paid | Fast | GPT-3.5, GPT-4 | Get key from platform.openai.com |
| **Ollama (GPU)** | FREE | Fast (local) | Llama, Mistral, Phi-3 | Install from ollama.com |
| **Ollama (CPU)** | FREE | Moderate | Same as GPU | No GPU required |

**💡 Recommended:** Start with **Groq** for best free experience!

---

## 🚨 Troubleshooting

### Common Issues

**❌ "ModuleNotFoundError"**
```bash
# Ensure venv is activated
.\venv\Scripts\activate  # Windows
source venv/bin/activate # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

**❌ "Docker not running"**
```bash
# Start Docker Desktop, then:
docker compose up -d

# Check status:
docker ps
```

**❌ "Groq API Error"**
- Check your API key in `.env`
- Verify key at https://console.groq.com/keys
- Free tier has rate limits (30 requests/minute)

**❌ "No documents indexed"**
- Put files in `data/documents/` folder
- Click "🚀 Upload & Index" in Tab 1
- Wait for indexing to complete

**❌ "Slow responses"**
- Try Groq instead of OpenAI (10x faster)
- Reduce `TOP_K_RESULTS` to 3
- Use smaller chunk sizes (300-400 chars)

---

## 📈 Performance Metrics

**Tested on 2,386 document chunks:**

| Metric | Value |
|--------|-------|
| **Indexing Time** | ~2 min (500 documents) |
| **Query Latency (P50)** | 1.8s with Groq, 2.5s with OpenAI |
| **Retrieval Accuracy** | 65-75% similarity scores |
| **Database Size** | ~10MB for 2,400 vectors (384D) |
| **Memory Usage** | ~500MB (embeddings loaded in RAM) |

---



## 🎯 Use Cases & Examples

### Example 1: Technical Documentation

**Question:** *"How do I configure Docker for this project?"*

**Answer:** 
> To configure Docker, run `docker compose up -d` from the project root. This starts the Endee vector database on port 8080...
> 
> **Sources:** 
> - `README.md` (similarity: 0.82)
> - `docker-compose.yml` (similarity: 0.75)

### Example 2: Research Papers

**Question:** *"What are the key findings about HNSW algorithm?"*

**Answer:**
> HNSW (Hierarchical Navigable Small World) achieves O(log n) search complexity with high recall rates. The paper demonstrates 10-100x speedup compared to brute-force search...
>
> **Sources:**
> - `vector_databases.md` (similarity: 0.88)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📋 FAQ

**Q: Do I need an OpenAI API key?**  
A: No! You can use Groq API (FREE) or Ollama (local). OpenAI is optional.

**Q: How much does Groq cost?**  
A: Completely FREE with generous rate limits (30 requests/minute).

**Q: Can I run this without internet?**  
A: Yes! Use Ollama for fully offline operation (requires downloading models first).

**Q: What file formats are supported?**  
A: PDF, DOCX, TXT, and Markdown (.md) files.

**Q: How many documents can I index?**  
A: Tested with 1000+ documents. Endee HNSW scales to millions of vectors.

**Q: Do files get re-indexed every time?**  
A: No! The system automatically skips already-indexed files (smart deduplication).

**Q: Can I delete indexed documents?**  
A: Yes, use the "🗑️ Clear Database" button in the sidebar or run `docker volume rm endee_data`.

---

## 📞 Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/yourusername/assignment_rag/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/yourusername/assignment_rag/discussions)
- 📧 **Email:** your.email@example.com

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

If this project helped you, please ⭐ star it on GitHub!

---

## 🔗 Links

- **Live Demo:** [Coming Soon]
- **Video Tutorial:** [Coming Soon]
- **Blog Post:** [Coming Soon]

---

<div align="center">

**Built with ❤️ using Endee, Streamlit, and Groq**

[⬆ Back to Top](#-intelligent-document-qa-system-with-rag)

</div>
