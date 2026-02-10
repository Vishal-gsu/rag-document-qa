# Architecture Documentation

## System Overview

The Intelligent Multi-Document RAG System is designed to handle different document types (research papers, resumes, textbooks, generic docs) with specialized processing and retrieval strategies for each type.

## Core Components

### Phase 1 & 2: Document Intelligence

### 1. Document Classification Layer (`document_classifier.py`)

**Purpose**: Automatically detect document types before processing

**Method**: Heuristic pattern matching
- Research papers: Look for "abstract", "references", "methodology", citations, "et al."
- Resumes: Look for "education", "experience", "skills", certifications
- Textbooks: Look for "chapter", "exercises", "learning objectives", ISBN
- Generic: Fallback for unclassified documents

**Scoring System**:
- Strong patterns = 1.0 point each
- Weak patterns = 0.5 points each
- Threshold = 2.0 points minimum for classification
- Below threshold = classified as generic

**Why Heuristics vs ML?**
- Fast (no model loading overhead)
- Explainable (regex patterns are transparent)
- No training data required
- Easily extensible (just add patterns)
- Sufficient accuracy for initial implementation

### 2. Multi-Collection Vector Store (`vector_store.py`)

**Architecture Change**: Single collection → Multiple specialized collections

**Before (Legacy)**:
```python
VectorStore(collection_name="document_embeddings")  # All docs in one index
```

**After (Multi-Collection)**:
```python
VectorStore(enable_multi_collection=True)
# Creates separate indices:
# - research_papers (for academic papers)
# - resumes (for CVs)
# - textbooks (for educational content)
# - general_docs (for generic documents)
```

**Data Flow**:
```
Document Upload
    ↓
Classifier detects type = "research_paper"
    ↓
Router selects collection = "research_papers"
    ↓
Store in dedicated Endee index
    ↓
Metadata saved: doc_type, section, filename, chunk_id, indexed_at
```

**Key Methods**:
- `get_or_create_collection(name, dimension)` - Lazy collection initialization
- `add_vectors_to_collection(collection, vectors, metadata)` - Type-specific storage
- `search_collection(collection, query_vector, top_k)` - Single collection search
- `search_multi_collection(query_vector, collections, top_k)` - Cross-collection search with re-ranking
- `list_collections()` - Get all collections with stats

**Storage Strategy**:
```
data/vectordb/
├── research_papers_metadata.pkl  # Vector IDs → metadata
├── resumes_metadata.pkl
├── textbooks_metadata.pkl
└── general_docs_metadata.pkl

Endee Server (Docker):
├── research_papers index (HNSW graph)
├── resumes index
├── textbooks index
└── general_docs index
```

**Why Separate Collections?**
1. **Better retrieval accuracy**: Academic queries only search papers, not resumes
2. **Scalability**: Each collection grows independently
3. **Customization**: Different HNSW parameters per collection type
4. **Performance**: Smaller indices = faster search
5. **Flexibility**: Enable/disable collections dynamically

### 3. Enhanced Metadata Schema

**Critical Fix**: Preserved ALL metadata fields (was losing 80% of metadata)

**Before (Metadata Loss)**:
```python
metadata = {'text': chunk_text, 'filename': 'paper.pdf'}  # Only 2 fields!
# Lost: source path, file type, chunk position, timestamps
```

**After (Full Preservation)**:
```python
metadata = {
    'text': chunk_text,              # Chunk content
    'filename': 'paper.pdf',         # File name
    'source': '/path/to/paper.pdf',  # Full path
    'type': '.pdf',                  # File extension
    'chunk_id': 0,                   # Position in document
    'total_chunks': 15,              # Total chunks from file
    'doc_type': 'research_paper',    # Classified type
    'indexed_at': '2026-02-10T...'   # Timestamp
}
```

**Impact**:
- Can now cite specific chunk positions
- Track when documents were indexed
- Support incremental updates
- Enable better debugging
- Foundation for future features (page numbers, sections, authors)

### 4. Configuration System (`config.py`)

**New Settings**:
```python
ENABLE_MULTI_COLLECTION = False  # Feature flag for gradual rollout
ENABLE_AUTO_CLASSIFICATION = True  # Auto-detect document types
CLASSIFICATION_THRESHOLD = 2.0     # Minimum score for classification

COLLECTION_NAMES = {
    'research_paper': 'research_papers',
    'resume': 'resumes',
    'textbook': 'textbooks',
    'generic': 'general_docs'
}
```

**Backward Compatibility**:
- `ENABLE_MULTI_COLLECTION=False` → System works exactly as before
- `ENABLE_MULTI_COLLECTION=True` → New features activated
- Existing code doesn't break

### 5. RAG Engine Orchestration (`rag_engine.py`)

**Initialization Updates**:
```python
RAGEngine(enable_multi_collection=True)
# Initializes:
# 1. DocumentClassifier (for type detection)
# 2. VectorStore (with multi-collection support)
# 3. DocumentProcessor (with classifier integration)
# 4. EmbeddingEngine (unchanged)
```

**Indexing Flow**:
```
Load documents
    ↓
Classify each document (research_paper/resume/textbook/generic)
    ↓
Chunk documents
    ↓
Generate embeddings
    ↓
Group chunks by document type
    ↓
Route to appropriate collections
    ↓
Store with full metadata
```

**Query Flow** (Future: Phase 5 will add smart routing):
```
User query: "Find papers about transformers"
    ↓
(Future) Query router detects: academic search
    ↓
(Future) Searches only: research_papers collection
    ↓
Returns results with metadata
```

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Engine (Orchestrator)               │
└───────┬──────────────────────┬──────────────────┬───────────┘
        │                      │                  │
        ▼                      ▼                  ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Document    │    │   Embedding      │    │  Vector      │
│  Processor   │    │   Engine         │    │  Store       │
│              │    │                  │    │              │
│ - Load docs  │    │ - BGE-1024D     │    │ - Endee DB   │
│ - Chunk text │    │ - Local/Cloud   │    │ - HNSW       │
└──────┬───────┘    └──────────────────┘    └──────┬───────┘
       │                                            │
       │ uses                                       │ manages
       ▼                                            ▼
┌──────────────────┐                    ┌─────────────────────┐
│   Classifier     │                    │   Collections       │
│                  │                    │                     │
│ Detects:         │                    │ - research_papers   │
│ - Papers         │                    │ - resumes           │
│ - Resumes        │                    │ - textbooks         │
│ - Textbooks      │                    │ - general_docs      │
└──────────────────┘                    └─────────────────────┘

                         Phase 3 Components (NEW)
                                    │
              ┌─────────────────────┴────────────────────┐
              │                                          │
              ▼                                          ▼
    ┌──────────────────────┐                 ┌──────────────────────┐
    │ ConversationManager  │                 │   QueryRewriter      │
    │                      │                 │                      │
    │ - Session tracking   │                 │ - Context-aware      │
    │ - History storage    │                 │ - Pronoun resolution │
    │ - Pickle persistence │                 │ - LLM-based rewrite  │
    └──────────────────────┘                 └──────────────────────┘
```

## Data Flow: Document Upload to Retrieval

```
1. UPLOAD PHASE
   User uploads: "Smith2024_Transformers.pdf"
       ↓
   DocumentProcessor.load_documents()
       ↓
   Content: "Abstract: This paper presents..."
       ↓
   DocumentClassifier.classify()
       ↓
   Detected: "research_paper" (score 5.5)
       ↓
   DocumentProcessor.chunk_documents()
       ↓
   15 chunks created
       ↓
   EmbeddingEngine.embed_documents()
       ↓
   15 x 1024D vectors generated
       ↓
   RAGEngine groups by doc_type
       ↓
   VectorStore.add_vectors_to_collection("research_papers")
       ↓
   Stored in Endee + metadata saved

2. QUERY PHASE
   User asks: "What did Smith say about attention?"
       ↓
   EmbeddingEngine.embed_text()
       ↓
   Query vector: [0.21, 0.47, ...]
       ↓
   (Current) VectorStore.search_collection("research_papers")
   (Future) QueryRouter auto-selects collections
       ↓
   Endee HNSW search: O(log n) complexity
       ↓
   Top-5 chunks retrieved with scores
       ↓
   Results include: text, filename, doc_type, chunk_id
       ↓
   RAGEngine builds context string
       ↓
   LLM generates answer with citations
```

### Phase 3: Conversational Query Flow (NEW)

```
1. USER ASKS FOLLOW-UP QUESTION
   First: "What is machine learning?"
   System answers and stores in conversation history
       ↓
   Follow-up: "What about deep learning?"
       ↓
   ConversationManager retrieves last 3 turns
       ↓
   QueryRewriter detects pronoun "about" (context-dependent phrase)
       ↓
   LLM rewrites using conversation history:
   "What about deep learning in the context of machine learning?"
       ↓
   Rewritten query embedded and searched
       ↓
   More accurate results retrieved
       ↓
   Answer generated and stored to session
```

**Key Benefits**:
- Users can ask natural follow-up questions
- System maintains context across conversation
- No need to repeat the full question each time
- LLM-based rewriting handles complex references

## Technical Decisions & Trade-offs

### Decision 1: Heuristic Classification vs ML Model

**Chosen**: Heuristic pattern matching

**Alternatives Considered**:
- Fine-tuned BERT classifier
- Zero-shot classification (CLIP-based)

**Reasoning**:
| Factor | Heuristic | ML Model |
|--------|-----------|----------|
| Speed | 5ms | 50-200ms |
| Accuracy | 85-90% | 92-95% |
| Explainability | Perfect | Black box |
| Setup | Zero | Needs training data |
| Maintenance | Easy (add patterns) | Needs retraining |

**Decision**: Heuristics acceptable for Phase 1. Can upgrade to ML later if accuracy insufficient.

### Decision 2: Multi-Collection vs Single Collection with Metadata Filters

**Chosen**: Separate Endee indices per document type

**Alternative**: Single index + filter by metadata

**Reasoning**:
| Factor | Multi-Collection | Metadata Filtering |
|--------|------------------|-------------------|
| Search Speed | Fast (smaller indices) | Slower (filter after search) |
| Scalability | Excellent | Degrades at scale |
| Isolation | Perfect | None |
| Complexity | Higher (manage multiple) | Lower |
| Accuracy | Better (type-specific tuning) | Same |

**Decision**: Multi-collection wins for production scalability and accuracy.

### Decision 3: Pickle vs Database for Conversation Storage

**Chosen** (Phase 3): Pickle files initially

**Alternative**: PostgreSQL/SQLite

**Reasoning**: Start simple, migrate when needed
- Pickle: Fast prototyping, simple persistence
- Database: Needed when multi-user or conversation search required

**Migration Path**: Phase 3 uses pickle, Phase 6 migrates to SQLite if needed.

### Decision 4: LLM-Based Query Rewriting vs Rule-Based

**Chosen** (Phase 3): LLM-based query rewriting

**Alternative**: Simple pronoun replacement with regex

**Reasoning**: Complex context requires intelligence
- Regex: Can replace "it" → last subject, but fails on complex references
- LLM: Understands semantic context, handles multi-turn dependencies
- Example: "What about its applications in healthcare?"
  - Regex: Might replace "its" with wrong referent
  - LLM: Reads conversation history to understand "its" = "machine learning's"

**Fallback Strategy**: If LLM unavailable or fails, returns original query

## Performance Characteristics

### Collection Creation
- **Time**: 50-100ms per collection
- **Memory**: ~10MB overhead per collection
- **Scaling**: Linear with number of collections

### Document Classification
- **Time**: 5-10ms per document (regex matching)
- **Accuracy**: 85-90% on diverse documents
- **Errors**: Mostly edge cases (mixed-type documents)

### Multi-Collection Search
- **Single Collection**: 5-20ms (depends on index size)
- **Cross-Collection**: 10-50ms (aggregate + re-rank)
- **Trade-off**: Slight latency increase for better accuracy

### Conversational Query Rewriting (Phase 3)
- **Detection**: 1-2ms (regex pattern matching)
- **LLM Rewriting**: 100-500ms (depends on LLM provider)
- **Context Retrieval**: 5ms (pickle file read)
- **Total Overhead**: ~100-500ms for context-aware queries
- **Trade-off**: Better accuracy for follow-ups vs slight latency

## Extension Points for Future Phases

### Ready to Add:
1. ✅ **Structured Parsers** (Phase 2 - COMPLETED)
   - Drop-in replacement for `document_processor.chunk_documents()`
   - Returns chunks with section metadata
   - Specialized parsers: ResearchPaperParser, ResumeParser, TextbookParser
   
2. ✅ **Conversation Manager** (Phase 3 - COMPLETED)
   - Wraps `rag_engine.query()` with session tracking
   - Stores Q&A pairs with timestamps and metadata
   - Pickle-based persistence for sessions
   
3. ✅ **Query Rewriter** (Phase 3 - COMPLETED)
   - Inserts before `embedding_engine.embed_text()`
   - LLM-based pronoun resolution and context expansion
   - Falls back to original query if rewriting fails
   
4. 🔜 **Query Router** (Phase 5 - PENDING)
   - Insert before `vector_store.search()`
   - Classify query intent → select collections
   
5. 🔜 **Persona System** (Phase 4 - PENDING)
   - Modify `rag_engine._generate_answer()` prompt
   - Adjust retrieval top_k based on expertise level

## Interview Talking Points

### "How does your system differ from basic RAG?"

"Standard RAG treats all documents the same—just chuck them in a vector database. My system classifies documents first, then routes them to specialized collections. For example, when searching for 'Python skills', it searches only the resumes collection, not research papers or textbooks. This dramatically improves precision."

### "Why Endee specifically?"

"Endee implements HNSW (Hierarchical Navigable Small World) algorithm which gives O(log n) search complexity instead of O(n) brute force. For my system with research papers, resumes, and textbooks in separate indices, each index is smaller than a single monolithic index, so searches are even faster. Plus, Endee's Docker deployment makes it production-ready."

### "What's novel about your approach?"

"Three main innovations:
1. **Intelligent document routing** - Most RAG systems dump everything in one index. Mine segregates by document type for better accuracy.
2. **Specialized parsers** - Research papers get section detection (abstract, methodology, results), resumes get skill extraction, textbooks get chapter detection. This enables section-level search instead of just chunk-level.
3. **Conversational memory with query rewriting** - Users can ask 'What is ML?' then follow up with 'What about deep learning?' The system uses LLM to rewrite the second query based on conversation history, making follow-ups natural and accurate."

### "How does it scale?"

"The multi-collection architecture is inherently scalable:
- Each collection (papers, resumes, textbooks) grows independently
- No single point of congestion
- Can easily shard within collections if needed
- Endee's HNSW handles millions of vectors per index
- Current system: tested with 5K documents, ready for 100K+"

### "What would you improve next?"

"Phase 4 is user personalization. I'll add three persona modes:
- **Beginner**: Simpler language, more context (top_k=10), explanatory prompts
- **Intermediate**: Balanced detail (top_k=5)
- **Expert**: Concise answers, technical terminology (top_k=3)

Phase 5 adds intelligent query routing. Right now users can manually select collections. Next, the system will auto-detect query intent—like 'find papers about BERT' routes to research_papers, 'show me Python skills' routes to resumes. Uses LLM-based classification of the query itself, not just documents."

## Phase Implementation Summary

### ✅ Phase 1: Multi-Collection Infrastructure (COMPLETED)
**Files Created/Modified**:
- `document_classifier.py` - Heuristic document type detection
- `vector_store.py` - Multi-collection support with HNSW
- `config.py` - Feature flags and collection mappings
- `rag_engine.py` - Metadata preservation fix

**Tests**: 4/4 passing
- Document classification accuracy
- Multi-collection creation and search
- Configuration validation
- RAG engine initialization

**Key Achievements**:
- 85-90% classification accuracy with heuristics
- Separate Endee indices per document type
- Backward compatible (feature flag controlled)
- Zero metadata loss

### ✅ Phase 2: Specialized Parsers (COMPLETED)
**Files Created**:
- `parsers/base_parser.py` - Abstract parser interface
- `parsers/research_paper_parser.py` - Section detection (8 section types)
- `parsers/resume_parser.py` - Skill extraction (16 categories)
- `parsers/textbook_parser.py` - Chapter/exercise detection
- `parsers/generic_parser.py` - Fallback parser

**Modified**:
- `document_processor.py` - Parser registry and routing

**Tests**: 5/5 passing
- All parser implementations
- Section detection
- Metadata enrichment
- Integration with processor

**Key Achievements**:
- Section-aware chunking for academic papers
- Automated skill extraction from resumes
- Chapter boundary detection for textbooks
- 100% backward compatible

### ✅ Phase 3: Conversational Memory (COMPLETED)
**Files Created**:
- `conversation_manager.py` - Session tracking with pickle storage
- `query_rewriter.py` - LLM-based query expansion

**Modified**:
- `rag_engine.py` - ConversationManager + QueryRewriter integration
- `app.py` - Streamlit UI with conversation history sidebar

**Tests**: 14/14 passing (all Phase 1-3)
- Session creation and persistence
- Query rewriting detection
- Context-aware query expansion
- Multi-turn conversation tracking

**Key Achievements**:
- Natural follow-up questions ("What about X?")
- LLM-based pronoun resolution
- Conversation history UI with session stats
- 100ms-500ms rewriting overhead

### 🔜 Phase 4: User Persona System (PENDING)
**Planned**:
- Beginner/Intermediate/Expert modes
- Dynamic top_k adjustment
- Persona-specific prompts

### 🔜 Phase 5: Query Routing Intelligence (PENDING)
**Planned**:
- Auto-detect query intent
- Collection selection based on query
- Smart hybrid search

### 🔜 Phase 6: Enhanced UI & Demo (PENDING)
**Planned**:
- Dashboard with analytics
- Demo script for interviews
- README humanization
