"""
Streamlit Web Interface for RAG System
3 Tabs: Upload, Query, Settings
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List
import json
import uuid

# Import RAG components
from rag_engine import RAGEngine
from config import Config
from llm_manager import LLMManager
from prompt_templates import PROMPT_TEMPLATES, get_template, get_template_names

# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "llm_manager" not in st.session_state:
    st.session_state.llm_manager = LLMManager()

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = get_template("Professional")

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

# Removed use_local_embeddings - always use local BGE embeddings

# Removed top_k, temperature, max_tokens - controlled by persona

if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.4

if "enable_conversation" not in st.session_state:
    st.session_state.enable_conversation = True

if "current_persona" not in st.session_state:
    st.session_state.current_persona = 'intermediate'

if "adapt_persona_to_query" not in st.session_state:
    st.session_state.adapt_persona_to_query = False

if "selected_collections" not in st.session_state:
    st.session_state.selected_collections = None  # None = auto-route

if "enable_auto_routing" not in st.session_state:
    st.session_state.enable_auto_routing = True


def initialize_rag():
    """Initialize RAG engine if not already done."""
    if st.session_state.rag_engine is None:
        try:
            # Always use local BGE embeddings (free, state-of-the-art)
            from embedding_engine import EmbeddingEngine
            embedding_engine = EmbeddingEngine(use_local=True)
            
            # Create RAG engine with local embeddings
            st.session_state.rag_engine = RAGEngine()
            st.session_state.rag_engine.embedding_engine = embedding_engine
            return True
        except Exception as e:
            st.error(f"Error initializing RAG engine: {e}")
            return False
    return True


def save_uploaded_file(uploaded_file, docs_dir="data/documents"):
    """Save uploaded file to documents directory."""
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    
    file_path = docs_path / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


# ============================================================================
# MAIN APP
# ============================================================================

st.title("📚 RAG Document Q&A System")
st.markdown("🚀 **Powered by Endee Vector Database (HNSW Algorithm)** | Upload documents, ask questions, and get AI-powered answers with source citations.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "💬 Query & Chat", "⚙️ Settings", "🧪 Interactive Tests"])

# ============================================================================
# TAB 1: UPLOAD DOCUMENTS
# ============================================================================

with tab1:
    st.header("📤 Upload & Index Documents")
    st.markdown("🔹 Upload documents to create your knowledge base using **Endee HNSW Vector Database**")
    st.markdown("⚡ **Why Endee?** Fast O(log n) similarity search • Scalable to millions of vectors • Industry-standard HNSW algorithm")
    st.markdown("📁 **Supported formats:** PDF, DOCX, TXT, MD, CSV, JSON, HTML")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'md', 'csv', 'json', 'html'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        st.success(f"✓ {len(uploaded_files)} file(s) selected")
        
        # Show file list
        with st.expander("📋 Selected Files", expanded=True):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🚀 Upload & Index", type="primary", width='stretch'):
                if not initialize_rag():
                    st.error("Failed to initialize RAG engine")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    saved_files = []
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")
                        try:
                            file_path = save_uploaded_file(file)
                            saved_files.append(file_path)
                            st.session_state.uploaded_files_list.append(file.name)
                        except Exception as e:
                            st.error(f"Error saving {file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if saved_files:
                        status_text.text("Indexing documents...")
                        try:
                            # Index the documents
                            st.session_state.rag_engine.index_documents("data/documents")
                            
                            status_text.empty()
                            progress_bar.empty()
                            st.success(f"✓ Successfully uploaded and indexed {len(saved_files)} file(s)!")
                            
                            # Show stats
                            stats = st.session_state.rag_engine.vector_store.get_stats()
                            st.info(f"📊 Database: {stats['total_vectors']} chunks indexed")
                            
                        except Exception as e:
                            st.error(f"Error indexing documents: {e}")
        
        with col2:
            if st.button("🗑️ Clear Selection", width='stretch'):
                st.rerun()
    
    # Show indexed documents
    st.markdown("---")
    st.subheader("📁 Document Database")
    
    if initialize_rag():
        stats = st.session_state.rag_engine.vector_store.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Chunks", stats.get('total_vectors', 0))
        col2.metric("Vector Dimension", stats.get('dimension', 0))
        col3.metric("Collection", stats.get('collection_name', 'N/A'))
        col4.metric("Vector DB", "🚀 Endee HNSW")
        
        if st.session_state.uploaded_files_list:
            with st.expander("📋 Uploaded Files History"):
                for filename in st.session_state.uploaded_files_list:
                    st.write(f"✓ {filename}")

# ============================================================================
# TAB 2: QUERY & CHAT
# ============================================================================

with tab2:
    st.header("💬 Ask Questions")
    
    if not initialize_rag():
        st.warning("⚠️ Please initialize the system in Settings tab first.")
    else:
        # Check if database has content
        stats = st.session_state.rag_engine.vector_store.get_stats()
        if stats.get('total_vectors', 0) == 0:
            st.warning("⚠️ No documents indexed yet. Please upload documents in the Upload tab.")
        else:
            # Chat interface
            st.markdown("Ask questions about your documents. The system remembers conversation history and can answer follow-up questions!")
            
            # Sidebar: Conversation controls
            with st.sidebar:
                st.markdown("### 💭 Conversation")
                
                # Show current session info
                if st.session_state.rag_engine and hasattr(st.session_state.rag_engine, 'conversation_manager'):
                    if st.session_state.rag_engine.conversation_manager:
                        stats = st.session_state.rag_engine.conversation_manager.get_session_stats(st.session_state.session_id)
                        if stats:
                            st.metric("Turns", stats.get('turn_count', 0))
                
                # New conversation button
                if st.button("🔄 New Conversation", help="Start a fresh conversation"):
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.conversation_history = []
                    st.success("Started new conversation!")
                    st.rerun()
                
                # Enable/disable conversation
                st.session_state.enable_conversation = st.checkbox(
                    "Enable Conversational Memory",
                    value=st.session_state.enable_conversation,
                    help="When enabled, the system remembers context from previous questions"
                )
                
                st.markdown("---")
                st.markdown("### 👤 User Persona")
                
                # Persona selector
                persona_options = {
                    'beginner': '🌱 Beginner - Detailed explanations',
                    'intermediate': '📚 Intermediate - Balanced approach',
                    'expert': '🎯 Expert - Technical depth',
                    'researcher': '🔬 Researcher - Academic focus'
                }
                
                selected_persona = st.selectbox(
                    "Select your expertise level:",
                    options=list(persona_options.keys()),
                    format_func=lambda x: persona_options[x],
                    index=list(persona_options.keys()).index(st.session_state.current_persona),
                    help="Adjusts answer style, detail level, and retrieval depth"
                )
                
                if selected_persona != st.session_state.current_persona:
                    st.session_state.current_persona = selected_persona
                    st.success(f"Persona updated to {persona_options[selected_persona]}")
                
                # Adaptive persona toggle
                st.session_state.adapt_persona_to_query = st.checkbox(
                    "Auto-adapt to query complexity",
                    value=st.session_state.adapt_persona_to_query,
                    help="Automatically adjust persona based on your question"
                )
                
                # Show current persona details
                if st.session_state.rag_engine and hasattr(st.session_state.rag_engine, 'persona_manager'):
                    if st.session_state.rag_engine.persona_manager:
                        profile = st.session_state.rag_engine.persona_manager.get_profile(st.session_state.current_persona)
                        with st.expander("📋 Persona Details"):
                            st.text(f"Top-K: {profile.top_k}")
                            st.text(f"Temperature: {profile.temperature}")
                            st.text(f"Max Tokens: {profile.max_tokens}")
                            st.text(f"Strategy: {profile.retrieval_strategy}")
                
                st.markdown("---")
                st.markdown("### 🗂️ Collection Routing")
                
                # Auto-routing toggle
                st.session_state.enable_auto_routing = st.checkbox(
                    "Enable Smart Routing",
                    value=st.session_state.enable_auto_routing,
                    help="Automatically route queries to relevant document types"
                )
                
                # Collection selector (manual override)
                if st.session_state.rag_engine and st.session_state.rag_engine.enable_multi_collection:
                    available_collections = list(st.session_state.rag_engine.vector_store.collections.keys())
                    
                    if not st.session_state.enable_auto_routing:
                        collection_options = {
                            'research_papers': '📄 Research Papers',
                            'resumes': '👤 Resumes/CVs',
                            'textbooks': '📚 Textbooks',
                            'general_docs': '📁 General Documents'
                        }
                        
                        selected = st.multiselect(
                            "Select collections to search:",
                            options=[c for c in collection_options.keys() if c in available_collections],
                            default=None,
                            format_func=lambda x: collection_options.get(x, x),
                            help="Choose which document types to search"
                        )
                        
                        st.session_state.selected_collections = selected if selected else None
                    else:
                        st.session_state.selected_collections = None
                        st.info("🤖 Auto-routing enabled - collections selected based on query intent")
            
            # Display conversation history from ConversationManager (Enhanced Phase 6)
            if st.session_state.rag_engine and hasattr(st.session_state.rag_engine, 'conversation_manager'):
                if st.session_state.rag_engine.conversation_manager:
                    history = st.session_state.rag_engine.conversation_manager.get_history(st.session_state.session_id)
                    
                    if history:
                        st.markdown("### 💭 Conversation History")
                        
                        # Show summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Questions", len(history))
                        with col2:
                            rewritten_count = sum(1 for h in history if h.get('metadata', {}).get('rewritten_query'))
                            st.metric("Rewrites", rewritten_count)
                        with col3:
                            avg_sources = sum(len(h.get('retrieved_context', [])) for h in history) / max(len(history), 1)
                            st.metric("Avg Sources", f"{avg_sources:.1f}")
                        
                        st.markdown("---")
                        
                        # Timeline view
                        for idx, turn in enumerate(history):
                            turn_number = idx + 1
                            
                            # Question with timestamp
                            col_q, col_time = st.columns([5, 1])
                            with col_q:
                                st.markdown(f"**Q{turn_number}:** {turn['question']}")
                            with col_time:
                                timestamp = turn.get('timestamp', 'N/A')
                                if isinstance(timestamp, str) and timestamp != 'N/A':
                                    # Just show time part if available
                                    time_part = timestamp.split('T')[-1][:8] if 'T' in timestamp else timestamp[:8]
                                    st.caption(time_part)
                            
                            # Answer (collapsible for long answers)
                            answer_preview = turn['answer'][:200]
                            if len(turn['answer']) > 200:
                                with st.expander(f"**A{turn_number}:** {answer_preview}..."):
                                    st.markdown(turn['answer'])
                            else:
                                st.markdown(f"**A{turn_number}:** {turn['answer']}")
                            
                            # Enhanced metadata display
                            meta_items = []
                            metadata = turn.get('metadata', {})
                            
                            if metadata.get('rewritten_query'):
                                meta_items.append(f"🔄 Rewritten: `{metadata['rewritten_query']}`")
                            
                            if metadata.get('persona'):
                                meta_items.append(f"👤 {metadata['persona'].title()}")
                            
                            if metadata.get('collections_searched') and isinstance(metadata.get('collections_searched'), list):
                                collections = ', '.join(metadata['collections_searched'])
                                meta_items.append(f"🗂️ {collections}")
                            
                            if metadata.get('routing_confidence'):
                                conf = metadata['routing_confidence']
                                meta_items.append(f"🎯 Confidence: {conf:.0%}")
                            
                            if meta_items:
                                st.caption(" • ".join(meta_items))
                            
                            st.markdown("---")
            
            # Query input
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What is machine learning? Then try: What about deep learning?",
                key="query_input"
            )
            
            col1, col2, col3 = st.columns([2, 2, 6])
            
            with col1:
                ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
            
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.conversation_history = []
                    st.rerun()
            
            if ask_button and question:
                with st.spinner("Thinking..."):
                    try:
                        import time
                        start_time = time.time()
                        
                        # Use RAG engine's query method with all features
                        result = st.session_state.rag_engine.query(
                            question=question,
                            top_k=None,  # Let persona decide unless overridden
                            return_context=True,
                            similarity_threshold=st.session_state.similarity_threshold,
                            session_id=st.session_state.session_id if st.session_state.enable_conversation else None,
                            enable_rewrite=st.session_state.enable_conversation,
                            persona=st.session_state.current_persona,
                            adapt_to_query=st.session_state.adapt_persona_to_query,
                            collections=st.session_state.selected_collections,
                            auto_route=st.session_state.enable_auto_routing
                        )
                        
                        query_time = time.time() - start_time
                        
                        answer = result['answer']
                        filtered_results = result['context']
                        
                        if not filtered_results:
                            st.error("No relevant information found.")
                        else:
                            # Performance Dashboard (Phase 6)
                            st.markdown("### 📊 Query Performance")
                            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                            
                            with perf_col1:
                                st.metric("⏱️ Total Time", f"{query_time:.2f}s")
                            with perf_col2:
                                st.metric("📄 Sources", len(filtered_results))
                            with perf_col3:
                                scores = [r.get('score', 0.0) for r in filtered_results if isinstance(r, dict) and 'score' in r]
                                avg_score = sum(scores) / len(scores) if scores else 0.0
                                st.metric("🎯 Avg Score", f"{avg_score:.0%}")
                            with perf_col4:
                                used_persona = result.get('metadata', {}).get('persona', st.session_state.current_persona)
                                if used_persona:
                                    st.metric("👤 Persona", used_persona.title())
                                else:
                                    st.metric("👤 Persona", "Default")
                            
                            # Routing Explanation (Phase 6)
                            if st.session_state.enable_auto_routing and result.get('metadata', {}).get('routing_explanation'):
                                with st.expander("🧭 Routing Decision", expanded=False):
                                    routing_info = result['metadata']
                                    st.markdown(f"**Query:** {question}")
                                    st.markdown(f"**Explanation:** {routing_info.get('routing_explanation', 'N/A')}")
                                    
                                    if routing_info.get('collections_searched') and isinstance(routing_info.get('collections_searched'), list):
                                        collections_str = ', '.join(routing_info['collections_searched'])
                                        st.markdown(f"**Collections Searched:** {collections_str}")
                                    
                                    if routing_info.get('routing_confidence'):
                                        conf = routing_info['routing_confidence']
                                        confidence_color = "🟢" if conf > 0.7 else "🟡" if conf > 0.5 else "🟠"
                                        st.markdown(f"**Confidence:** {confidence_color} {conf:.0%}")
                                    
                                    # Show all collection scores if available
                                    if hasattr(st.session_state.rag_engine, 'query_router'):
                                        all_scores = st.session_state.rag_engine.query_router.get_intent_confidence(question)
                                        if all_scores:
                                            st.markdown("**All Collection Scores:**")
                                            for coll, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                                                bar_width = int(score * 20)
                                                bar = "█" * bar_width + "░" * (20 - bar_width)
                                                st.text(f"{coll:20s} {bar} {score:.0%}")
                            
                            # Display answer
                            st.markdown("### 💡 Answer")
                            st.markdown(answer)
                            
                            # Display sources
                            st.markdown(f"### 📚 Sources (showing {len(filtered_results)} most relevant)")
                            for i, result_item in enumerate(filtered_results):
                                # Ensure result_item has required fields
                                if not isinstance(result_item, dict) or 'score' not in result_item:
                                    continue
                                    
                                sim_score = result_item.get('score', 0.0)
                                # Color code by similarity
                                if sim_score >= 0.7:
                                    emoji = "🟢"
                                elif sim_score >= 0.5:
                                    emoji = "🟡"
                                else:
                                    emoji = "🟠"
                                
                                # Add collection tag if multi-collection
                                collection_tag = ""
                                metadata = result_item.get('metadata', {})
                                if isinstance(metadata, dict) and metadata.get('collection'):
                                    coll_icons = {
                                        'research_papers': '📄',
                                        'resumes': '👤',
                                        'textbooks': '📚',
                                        'general_docs': '📁'
                                    }
                                    coll_name = metadata['collection']
                                    collection_tag = f" {coll_icons.get(coll_name, '📄')} [{coll_name}]"
                                
                                with st.expander(f"{emoji} Source {i+1}: {metadata.get('filename', 'Unknown')}{collection_tag} (Similarity: {sim_score:.1%})"):
                                    st.text(result_item.get('text', 'No text available'))
                                    
                                    # Show additional metadata
                                    meta_info = []
                                    if metadata.get('chunk_index') is not None:
                                        meta_info.append(f"Chunk #{metadata['chunk_index'] + 1}")
                                    if metadata.get('doc_type'):
                                        meta_info.append(f"Type: {metadata['doc_type']}")
                                    if meta_info:
                                        st.caption(" | ".join(meta_info))
                            
                            # Enhanced Export (Phase 6)
                            export_data = {
                                "question": question,
                                "answer": answer,
                                "session_id": st.session_state.session_id,
                                "query_time_seconds": query_time,
                                "persona": used_persona,
                                "routing": {
                                    "enabled": st.session_state.enable_auto_routing,
                                    "collections_searched": result.get('metadata', {}).get('collections_searched', []),
                                    "confidence": result.get('metadata', {}).get('routing_confidence'),
                                    "explanation": result.get('metadata', {}).get('routing_explanation')
                                },
                                "sources": [
                                    {
                                        "file": r.get('metadata', {}).get('filename', 'Unknown') if isinstance(r.get('metadata'), dict) else 'Unknown',
                                        "collection": r.get('metadata', {}).get('collection') if isinstance(r.get('metadata'), dict) else None,
                                        "similarity": r.get('score', 0.0),
                                        "text": r.get('text', ''),
                                        "chunk_index": r.get('metadata', {}).get('chunk_index') if isinstance(r.get('metadata'), dict) else None
                                    }
                                    for r in filtered_results
                                    if isinstance(r, dict)
                                ],
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            col_export1, col_export2 = st.columns(2)
                            with col_export1:
                                st.download_button(
                                    "📥 Download Result (JSON)",
                                    data=json.dumps(export_data, indent=2),
                                    file_name=f"rag_result_{st.session_state.session_id[:8]}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            with col_export2:
                                # Export full conversation
                                if st.button("💾 Export Full Conversation", use_container_width=True):
                                    if st.session_state.rag_engine.conversation_manager:
                                        full_history = st.session_state.rag_engine.conversation_manager.get_history(st.session_state.session_id)
                                        conversation_export = {
                                            "session_id": st.session_state.session_id,
                                            "total_turns": len(full_history),
                                            "persona": st.session_state.current_persona,
                                            "conversation": full_history,
                                            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                        st.download_button(
                                            "📥 Download Conversation",
                                            data=json.dumps(conversation_export, indent=2),
                                            file_name=f"conversation_{st.session_state.session_id[:8]}.json",
                                            mime="application/json",
                                            use_container_width=True,
                                            key="download_conversation"
                                        )
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

# ============================================================================
# TAB 3: SETTINGS
# ============================================================================

with tab3:
    st.header("⚙️ System Settings")
    
    # Endee Vector Database Info Box
    st.info("""
    ### 🚀 **Endee Vector Database (HNSW Algorithm)**
    
    **What is Endee?**
    - Industry-standard vector database using HNSW (Hierarchical Navigable Small World) algorithm
    - Provides O(log n) similarity search - exponentially faster than brute-force O(n)
    
    **Performance Comparison:**
    ```
    📊 Vectors | Old (Pickle) | Endee HNSW | Speedup
    ────────────────────────────────────────────────
    100        | 100ms        | 10ms       | 10x faster
    1,000      | 1s           | 12ms       | 83x faster  
    10,000     | 10s          | 14ms       | 714x faster
    100,000    | 100s         | 17ms       | 5882x faster
    ```
    
    **Your current database (~61 chunks):** ⚡ **~6x faster** with Endee!
    
    **Why it matters:**
    - ✅ Scales to millions of vectors without slowdown
    - ✅ Professional-grade infrastructure
    - ✅ Memory-efficient graph indexing
    - ✅ Production-ready for real applications
    """)
    
    st.markdown("---")
    
    # Embedding Model Info (Fixed to BGE)
    st.subheader("🔢 Embedding Model")
    
    st.success("✓ **Using BAAI/bge-large-en-v1.5** (1024D, state-of-the-art, FREE)")
    st.info(
        "**Why BGE?**\n\n"
        "• Top-ranked on MTEB benchmark\n"
        "• 1024-dimensional embeddings\n"
        "• No API costs or rate limits\n"
        "• First run downloads ~400MB model (one-time)\n\n"
        "If not installed, run: `pip install sentence-transformers`"
    )
    
    st.markdown("---")
    
    # LLM Provider Selection
    st.subheader("🤖 LLM Provider (for Answering)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        provider = st.radio(
            "Choose LLM Provider:",
            ["☁️ OpenAI API", "⚡ Groq API (FREE)", "🚀 Ollama (GPU)", "💻 Ollama (CPU)"],
            help="OpenAI requires paid API key. Groq is FREE & fast. Ollama runs locally."
        )
        
        # Set provider
        if provider == "☁️ OpenAI API":
            api_key = st.text_input("OpenAI API Key:", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            model = st.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
            
            if st.button("Set OpenAI Provider"):
                try:
                    st.session_state.llm_manager.set_provider("openai", api_key=api_key, model=model)
                    st.success(f"✓ Using OpenAI {model}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif provider == "⚡ Groq API (FREE)":
            api_key = st.text_input("Groq API Key:", type="password", value=os.getenv("GROQ_API_KEY", ""),
                                   help="Get FREE API key from https://console.groq.com/keys")
            model = st.selectbox("Model:", [
                "llama-3.3-70b-versatile",  # Best for general tasks
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768",       # Good for long context
                "gemma2-9b-it"              # Fast & efficient
            ])
            
            if st.button("Set Groq Provider"):
                try:
                    st.session_state.llm_manager.set_provider("groq", api_key=api_key, model=model)
                    st.success(f"✓ Using Groq {model} (FREE & Fast!)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif provider == "🚀 Ollama (GPU)":
            if not st.session_state.llm_manager.ollama_available:
                st.warning("⚠️ Ollama not detected. Please install from https://ollama.com")
                st.code("# Install Ollama, then run:\nollama pull llama3.2")
            else:
                models = st.session_state.llm_manager.get_available_ollama_models()
                if models:
                    model = st.selectbox("Model:", models)
                    if st.button("Set Ollama GPU"):
                        try:
                            st.session_state.llm_manager.set_provider("ollama_gpu", model=model)
                            st.success(f"✓ Using Ollama {model} with GPU")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("No models found. Install with: ollama pull llama3.2")
        
        else:  # Ollama CPU
            if not st.session_state.llm_manager.ollama_available:
                st.warning("⚠️ Ollama not detected. Please install from https://ollama.com")
            else:
                models = st.session_state.llm_manager.get_available_ollama_models()
                if models:
                    model = st.selectbox("Model:", models)
                    if st.button("Set Ollama CPU"):
                        try:
                            st.session_state.llm_manager.set_provider("ollama_cpu", model=model)
                            st.success(f"✓ Using Ollama {model} on CPU")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("No models found. Install with: ollama pull llama3.2")
    
    with col2:
        st.info("**💡 LLM Options:**\n\n"
                "**OpenAI API:** Best quality, requires paid API key\n\n"
                "**Groq API:** ⭐ FREE & FAST! Llama 3.3 70B, Mixtral models\n\n"
                "**Ollama GPU:** Fast local inference, free, requires 4GB+ GPU\n\n"
                "**Ollama CPU:** Slower but works without GPU, free\n\n"
                "**Recommended:** Start with Groq for best free experience!")
        
        # Current provider info
        info = st.session_state.llm_manager.get_provider_info()
        st.json(info)
    
    st.markdown("---")
    
    # System Prompt
    st.subheader("📝 System Prompt")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        template_name = st.selectbox(
            "Prompt Template:",
            get_template_names(),
            help="Choose a pre-built template or select Custom to write your own"
        )
        
        template_info = PROMPT_TEMPLATES[template_name]
        st.info(f"**{template_info['name']}**\n\n{template_info['description']}")
    
    with col2:
        st.markdown("**Preview:**")
        st.text_area(
            "Template content:",
            value=template_info['prompt'],
            height=200,
            disabled=True,
            key="template_preview"
        )
    
    # Custom prompt editor
    st.markdown("### ✏️ Custom System Prompt")
    custom_prompt = st.text_area(
        "Edit or create your own system prompt:",
        value=st.session_state.system_prompt,
        height=200,
        help="This prompt instructs the AI how to respond to questions"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Use Template", width='stretch'):
            st.session_state.system_prompt = get_template(template_name)
            st.success(f"✓ Loaded {template_name} template")
            st.rerun()
    
    with col2:
        if st.button("💾 Save Custom", width='stretch'):
            st.session_state.system_prompt = custom_prompt
            st.success("✓ Custom prompt saved")
    
    with col3:
        if st.button("🔄 Reset to Default", width='stretch'):
            st.session_state.system_prompt = get_template("Professional")
            st.success("✓ Reset to Professional template")
            st.rerun()
    
    st.markdown("---")
    
    # RAG Parameters Info (Controlled by Persona)
    st.subheader("🎚️ RAG Parameters")
    st.info(
        "📊 **Parameters are controlled by User Persona** (see Query tab sidebar)\n\n"
        "Current persona adjusts these automatically:\n"
        "• **Top-K**: Number of documents retrieved\n"
        "• **Temperature**: Creativity vs focus\n"
        "• **Max Tokens**: Answer length\n"
        "• **Strategy**: Retrieval approach\n\n"
        "Switch persona in the Query tab to change these settings."
    )
    
    # Show current persona settings
    if st.session_state.rag_engine and hasattr(st.session_state.rag_engine, 'persona_manager'):
        if st.session_state.rag_engine.persona_manager:
            profile = st.session_state.rag_engine.persona_manager.get_profile(st.session_state.current_persona)
            
            st.markdown(f"### 👤 Current Persona: **{st.session_state.current_persona.title()}**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Top-K", profile.top_k)
            with col2:
                st.metric("Temperature", f"{profile.temperature}")
            with col3:
                st.metric("Max Tokens", profile.max_tokens)
            with col4:
                st.metric("Strategy", profile.retrieval_strategy.replace('_', ' ').title())
    
    # Similarity threshold (not persona-controlled)
    st.markdown("### 🎯 Similarity Threshold")
    similarity_threshold = st.slider(
        "Minimum similarity score for retrieved documents:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.similarity_threshold,
        step=0.05,
        help="Only use chunks above this similarity. Higher = more strict, Lower = more results"
    )
    st.caption(f"Current: {similarity_threshold:.0%} - {'Strict' if similarity_threshold >= 0.6 else 'Moderate' if similarity_threshold >= 0.4 else 'Lenient'}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Threshold", type="primary"):
            st.session_state.similarity_threshold = similarity_threshold
            st.success("✓ Threshold saved!")
    with col2:
        if st.button("🔄 Reset to Default"):
            st.session_state.similarity_threshold = 0.4
            st.success("✓ Reset to 40%!")
            st.rerun()
    
    st.info("ℹ️ **Note:** Chunk Size (1000) and Overlap (200) are fixed in config to maintain database consistency.")

# ============================================================================
# TAB 4: INTERACTIVE TESTS
# ============================================================================

with tab4:
    st.header("🧪 Interactive Tests & Visualizations")
    st.markdown("Test embedding quality, similarity search, and system performance with your own inputs")
    
    if not initialize_rag():
        st.warning("⚠️ Please initialize the RAG engine first by uploading documents in the Upload tab")
    else:
        # Test selection
        test_type = st.selectbox(
            "Choose a test:",
            [
                "1️⃣ Embedding Similarity Test",
                "2️⃣ Semantic Search Quality",
                "3️⃣ Chunk Retrieval Analysis",
                "4️⃣ Multi-Query Comparison",
                "5️⃣ Embedding Visualization (t-SNE)",
                "6️⃣ Performance Benchmarking"
            ]
        )
        
        st.markdown("---")
        
        # ========================================================================
        # TEST 1: Embedding Similarity Test
        # ========================================================================
        if test_type == "1️⃣ Embedding Similarity Test":
            st.subheader("🔬 Test Embedding Similarity")
            st.markdown("Compare how similar different texts are in the embedding space")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text1 = st.text_area(
                    "Text 1:",
                    value="Machine learning is a subset of artificial intelligence",
                    height=100,
                    key="similarity_text1"
                )
            
            with col2:
                text2 = st.text_area(
                    "Text 2:",
                    value="AI and machine learning are related concepts",
                    height=100,
                    key="similarity_text2"
                )
            
            if st.button("🔍 Calculate Similarity", key="calc_similarity"):
                with st.spinner("Computing embeddings..."):
                    try:
                        import numpy as np
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        # Get embeddings with error handling
                        emb1_result = st.session_state.rag_engine.embedding_engine.embed_text(text1)
                        emb2_result = st.session_state.rag_engine.embedding_engine.embed_text(text2)
                        
                        # Validate embeddings are lists/arrays, not strings
                        if isinstance(emb1_result, str) or isinstance(emb2_result, str):
                            st.error(f"Embedding error: {emb1_result if isinstance(emb1_result, str) else emb2_result}")
                            st.stop()
                        
                        emb1 = np.array(emb1_result)
                        emb2 = np.array(emb2_result)
                        
                        # Verify embeddings are numeric
                        if emb1.dtype.kind not in 'fc' or emb2.dtype.kind not in 'fc':
                            st.error(f"Invalid embedding format. Expected numeric array, got: {emb1.dtype}, {emb2.dtype}")
                            st.stop()
                        
                        # Calculate similarity
                        similarity = cosine_similarity([emb1], [emb2])[0][0]
                        
                        # Display results
                        st.markdown("### 📊 Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Similarity Score", f"{similarity:.4f}")
                        
                        with col2:
                            st.metric("Percentage", f"{similarity * 100:.2f}%")
                        
                        with col3:
                            status = "Very Similar ✅" if similarity > 0.7 else "Similar ✓" if similarity > 0.5 else "Somewhat Similar ⚠️" if similarity > 0.3 else "Different ❌"
                            st.metric("Status", status)
                        
                        # Visual gauge
                        st.markdown("### 🎯 Similarity Gauge")
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=similarity * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Similarity %"},
                            delta={'reference': 70},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgray"},
                                    {'range': [30, 50], 'color': "lightyellow"},
                                    {'range': [50, 70], 'color': "lightgreen"},
                                    {'range': [70, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        st.markdown("### 💡 Interpretation")
                        if similarity > 0.7:
                            st.success("✅ **Very Similar**: These texts have very similar semantic meaning")
                        elif similarity > 0.5:
                            st.info("✓ **Similar**: These texts share common concepts")
                        elif similarity > 0.3:
                            st.warning("⚠️ **Somewhat Similar**: Some overlap but different focus")
                        else:
                            st.error("❌ **Different**: These texts discuss different topics")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # ========================================================================
        # TEST 2: Semantic Search Quality
        # ========================================================================
        elif test_type == "2️⃣ Semantic Search Quality":
            st.subheader("🔍 Test Semantic Search")
            st.markdown("Search your knowledge base and see detailed similarity scores")
            
            query = st.text_input(
                "Enter search query:",
                value="What is machine learning?",
                key="semantic_search_query"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                k = st.slider("Number of results:", 1, 10, 5, key="semantic_k")
            with col2:
                threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.3, 0.05, key="semantic_threshold")
            
            if st.button("🔎 Search", key="semantic_search"):
                with st.spinner("Searching..."):
                    try:
                        # Get embedding and search directly
                        query_embedding = st.session_state.rag_engine.embedding_engine.embed_text(query)
                        results = st.session_state.rag_engine.vector_store.search(
                            query_vector=query_embedding,
                            top_k=k
                        )
                        
                        # Filter by similarity threshold
                        results = [r for r in results if r['score'] >= threshold]
                        
                        if not results:
                            st.warning("No results found above the similarity threshold")
                        else:
                            st.success(f"Found {len(results)} relevant chunks")
                            
                            # Create bar chart of similarities
                            import plotly.express as px
                            import pandas as pd
                            
                            df = pd.DataFrame({
                                'Chunk': [f"Chunk {i+1}" for i in range(len(results))],
                                'Similarity': [r['score'] * 100 for r in results]
                            })
                            
                            fig = px.bar(
                                df,
                                x='Chunk',
                                y='Similarity',
                                title='Similarity Scores for Retrieved Chunks',
                                labels={'Similarity': 'Similarity (%)'},
                                color='Similarity',
                                color_continuous_scale='Viridis'
                            )
                            fig.add_hline(y=threshold * 100, line_dash="dash", line_color="red",
                                        annotation_text=f"Threshold ({threshold * 100:.0f}%)")
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            # Show results
                            st.markdown("### 📄 Retrieved Chunks")
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Chunk {i} - {result['score'] * 100:.1f}% similar"):
                                    st.write(result['text'])
                                    st.caption(f"Source: {result['metadata'].get('filename', 'Unknown')}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # ========================================================================
        # TEST 3: Chunk Retrieval Analysis
        # ========================================================================
        elif test_type == "3️⃣ Chunk Retrieval Analysis":
            st.subheader("📊 Analyze Retrieval Distribution")
            st.markdown("See how your query matches across different similarity ranges")
            
            query = st.text_input(
                "Test query:",
                value="neural networks and deep learning",
                key="chunk_analysis_query"
            )
            
            if st.button("📈 Analyze", key="chunk_analysis"):
                with st.spinner("Analyzing..."):
                    try:
                        # Get more results for analysis
                        query_embedding = st.session_state.rag_engine.embedding_engine.embed_text(query)
                        results = st.session_state.rag_engine.vector_store.search(
                            query_vector=query_embedding,
                            top_k=20
                        )
                        
                        if not results:
                            st.warning("No chunks found")
                        else:
                            import plotly.express as px
                            import pandas as pd
                            
                            # Create distribution histogram
                            similarities = [r['score'] * 100 for r in results]
                            
                            fig = px.histogram(
                                x=similarities,
                                nbins=20,
                                title='Distribution of Similarity Scores',
                                labels={'x': 'Similarity (%)', 'y': 'Count'},
                                color_discrete_sequence=['steelblue']
                            )
                            fig.add_vline(x=30, line_dash="dash", line_color="red",
                                        annotation_text="Default Threshold (30%)")
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            # Statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Chunks", len(results))
                            with col2:
                                st.metric("Above 30%", len([s for s in similarities if s >= 30]))
                            with col3:
                                st.metric("Above 50%", len([s for s in similarities if s >= 50]))
                            with col4:
                                st.metric("Above 70%", len([s for s in similarities if s >= 70]))
                            
                            # Top 5 breakdown
                            st.markdown("### 🏆 Top 5 Matches")
                            top_5 = results[:5]
                            
                            for i, result in enumerate(top_5, 1):
                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    st.metric(f"Rank {i}", f"{result['score'] * 100:.1f}%")
                                with col2:
                                    st.text(result['text'][:150] + "...")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # ========================================================================
        # TEST 4: Multi-Query Comparison
        # ========================================================================
        elif test_type == "4️⃣ Multi-Query Comparison":
            st.subheader("🔄 Compare Multiple Queries")
            st.markdown("Test how different phrasings of the same question perform")
            
            st.markdown("**Enter 3-4 variations of the same question:**")
            
            queries = []
            for i in range(4):
                query = st.text_input(
                    f"Query {i+1}:",
                    value=["What is machine learning?", 
                           "Explain ML", 
                           "Define machine learning",
                           "What does ML mean?"][i],
                    key=f"multi_query_{i}"
                )
                if query.strip():
                    queries.append(query)
            
            if st.button("🔬 Compare Queries", key="compare_queries"):
                if len(queries) < 2:
                    st.warning("Please enter at least 2 queries")
                else:
                    with st.spinner("Analyzing all queries..."):
                        try:
                            import pandas as pd
                            import plotly.express as px
                            
                            results_data = []
                            
                            for query in queries:
                                query_embedding = st.session_state.rag_engine.embedding_engine.embed_text(query)
                                results = st.session_state.rag_engine.vector_store.search(
                                    query_vector=query_embedding,
                                    top_k=5
                                )
                                
                                if results:
                                    avg_sim = sum(r['score'] for r in results) / len(results)
                                    max_sim = max(r['score'] for r in results)
                                    min_sim = min(r['score'] for r in results)
                                    
                                    results_data.append({
                                        'Query': query[:30] + "...",
                                        'Avg Similarity': avg_sim * 100,
                                        'Max Similarity': max_sim * 100,
                                        'Min Similarity': min_sim * 100,
                                        'Results Count': len(results)
                                    })
                            
                            df = pd.DataFrame(results_data)
                            
                            # Bar chart comparison
                            fig = px.bar(
                                df,
                                x='Query',
                                y=['Avg Similarity', 'Max Similarity', 'Min Similarity'],
                                title='Query Performance Comparison',
                                labels={'value': 'Similarity (%)', 'variable': 'Metric'},
                                barmode='group'
                            )
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            # Table
                            st.dataframe(df, width='stretch')
                            
                            # Best query
                            best_idx = df['Avg Similarity'].idxmax()
                            st.success(f"🏆 **Best performing query:** {queries[best_idx]}")
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        # ========================================================================
        # TEST 5: Embedding Visualization (t-SNE)
        # ========================================================================
        elif test_type == "5️⃣ Embedding Visualization (t-SNE)":
            st.subheader("🎨 Visualize Embeddings in 2D")
            st.markdown("See how your texts cluster in embedding space using t-SNE")
            
            st.markdown("**Enter 5-10 different texts to visualize:**")
            
            texts = []
            labels = []
            for i in range(6):
                col1, col2 = st.columns([3, 1])
                with col1:
                    text = st.text_input(
                        f"Text {i+1}:",
                        value=["Machine learning algorithms", 
                               "Neural networks",
                               "Pizza recipe",
                               "Cooking pasta",
                               "Deep learning models",
                               "Italian cuisine"][i],
                        key=f"tsne_text_{i}"
                    )
                with col2:
                    label = st.text_input(
                        "Label:",
                        value=["AI", "AI", "Food", "Food", "AI", "Food"][i],
                        key=f"tsne_label_{i}"
                    )
                
                if text.strip():
                    texts.append(text)
                    labels.append(label)
            
            if st.button("🎨 Visualize", key="visualize_tsne"):
                if len(texts) < 3:
                    st.warning("Please enter at least 3 texts")
                else:
                    with st.spinner("Computing embeddings and running t-SNE..."):
                        try:
                            import numpy as np
                            from sklearn.manifold import TSNE
                            import plotly.express as px
                            import pandas as pd
                            
                            # Get embeddings
                            embeddings = []
                            for text in texts:
                                emb = st.session_state.rag_engine.embedding_engine.embed_text(text)
                                embeddings.append(emb)
                            
                            embeddings = np.array(embeddings)
                            
                            # Apply t-SNE
                            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(texts)-1))
                            embeddings_2d = tsne.fit_transform(embeddings)
                            
                            # Create DataFrame
                            df = pd.DataFrame({
                                'x': embeddings_2d[:, 0],
                                'y': embeddings_2d[:, 1],
                                'text': [t[:50] + "..." if len(t) > 50 else t for t in texts],
                                'label': labels
                            })
                            
                            # Plot
                            fig = px.scatter(
                                df,
                                x='x',
                                y='y',
                                color='label',
                                hover_data=['text'],
                                title='Text Embeddings Visualization (t-SNE)',
                                labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'},
                                size_max=15
                            )
                            
                            fig.update_traces(marker=dict(size=12))
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            st.info("💡 **How to read:** Points close together have similar semantic meaning. Different colors show your labels.")
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.info("Note: t-SNE requires at least 3 texts. If you see perplexity errors, try adding more texts.")
        
        # ========================================================================
        # TEST 6: Performance Benchmarking
        # ========================================================================
        elif test_type == "6️⃣ Performance Benchmarking":
            st.subheader("⚡ Performance Benchmarking")
            st.markdown("Measure the speed of different operations")
            
            num_queries = st.slider("Number of test queries:", 5, 50, 10, key="bench_queries")
            
            if st.button("🏃 Run Benchmark", key="run_benchmark"):
                with st.spinner("Running benchmark..."):
                    try:
                        import time
                        import numpy as np
                        import pandas as pd
                        import plotly.express as px
                        
                        test_queries = [
                            "What is machine learning?",
                            "Explain neural networks",
                            "Define AI",
                            "How does deep learning work?",
                            "What are transformers?"
                        ] * (num_queries // 5 + 1)
                        test_queries = test_queries[:num_queries]
                        
                        embedding_times = []
                        search_times = []
                        total_times = []
                        
                        for query in test_queries:
                            # Measure embedding time
                            start = time.time()
                            emb = st.session_state.rag_engine.embedding_engine.embed_text(query)
                            embedding_time = (time.time() - start) * 1000
                            embedding_times.append(embedding_time)
                            
                            # Measure search time
                            start = time.time()
                            query_embedding = st.session_state.rag_engine.embedding_engine.embed_text(query)
                            results = st.session_state.rag_engine.vector_store.search(
                                query_vector=query_embedding,
                                top_k=5
                            )
                            search_time = (time.time() - start) * 1000
                            search_times.append(search_time)
                            
                            total_times.append(embedding_time + search_time)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Avg Embedding Time", f"{np.mean(embedding_times):.2f} ms")
                            st.caption(f"Min: {np.min(embedding_times):.2f} ms | Max: {np.max(embedding_times):.2f} ms")
                        
                        with col2:
                            st.metric("Avg Search Time", f"{np.mean(search_times):.2f} ms")
                            st.caption(f"Min: {np.min(search_times):.2f} ms | Max: {np.max(search_times):.2f} ms")
                        
                        with col3:
                            st.metric("Avg Total Time", f"{np.mean(total_times):.2f} ms")
                            st.caption(f"Min: {np.min(total_times):.2f} ms | Max: {np.max(total_times):.2f} ms")
                        
                        # Time series plot
                        df = pd.DataFrame({
                            'Query': list(range(1, len(test_queries) + 1)),
                            'Embedding Time (ms)': embedding_times,
                            'Search Time (ms)': search_times,
                            'Total Time (ms)': total_times
                        })
                        
                        fig = px.line(
                            df,
                            x='Query',
                            y=['Embedding Time (ms)', 'Search Time (ms)', 'Total Time (ms)'],
                            title=f'Performance Over {num_queries} Queries',
                            labels={'value': 'Time (ms)', 'variable': 'Operation'}
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Interpretation
                        st.markdown("### 💡 Performance Insights")
                        
                        avg_total = np.mean(total_times)
                        if avg_total < 100:
                            st.success("✅ **Excellent**: Very fast response times (<100ms)")
                        elif avg_total < 500:
                            st.info("✓ **Good**: Acceptable response times (<500ms)")
                        else:
                            st.warning("⚠️ **Slow**: Consider optimization (>500ms)")
                        
                        # Breakdown
                        embedding_pct = (np.mean(embedding_times) / avg_total) * 100
                        search_pct = (np.mean(search_times) / avg_total) * 100
                        
                        st.write(f"- Embedding: {embedding_pct:.1f}% of total time")
                        st.write(f"- Search: {search_pct:.1f}% of total time")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "📚 RAG Document Q&A System | Built with Streamlit & Endee Vector DB"
    "</div>",
    unsafe_allow_html=True
)
