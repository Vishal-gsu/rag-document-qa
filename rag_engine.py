"""
RAG Engine Module
Orchestrates the complete RAG pipeline: retrieval + generation.
Supports multi-collection architecture for document type segregation.
Includes conversational memory, query rewriting, user personas, and query routing.
"""
from typing import List, Dict, Optional
from datetime import datetime
from config import Config
from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from vector_store import VectorStore
from document_classifier import DocumentClassifier
from conversation_manager import ConversationManager
from query_rewriter import QueryRewriter
from user_persona import UserPersona
from query_router import QueryRouter


class RAGEngine:
    """
    Retrieval Augmented Generation Engine.
    Combines document retrieval with LLM generation.
    """
    
    def __init__(self, 
                 db_path: str = None,
                 collection_name: str = None,
                 chat_model: str = None,
                 enable_multi_collection: bool = None,
                 enable_conversation: bool = True,
                 enable_personas: bool = True,
                 default_persona: str = 'intermediate',
                 enable_query_routing: bool = True,
                 use_llm_routing: bool = False):
        """
        Initialize RAG engine.
        
        Args:
            db_path: Path to vector database
            collection_name: Name of the default collection
            chat_model: OpenAI chat model to use
            enable_multi_collection: Enable multi-collection mode
            enable_conversation: Enable conversational memory
            enable_personas: Enable user persona system
            default_persona: Default persona (beginner/intermediate/expert/researcher)
            enable_query_routing: Enable intelligent query routing
            use_llm_routing: Use LLM for query routing (more accurate but slower)
        """
        # Validate configuration
        Config.validate()
        
        # Determine multi-collection mode
        if enable_multi_collection is None:
            enable_multi_collection = Config.ENABLE_MULTI_COLLECTION
        
        self.enable_multi_collection = enable_multi_collection
        self.enable_conversation = enable_conversation
        self.enable_personas = enable_personas
        self.enable_query_routing = enable_query_routing and enable_multi_collection  # Only meaningful in multi-collection mode
        
        # Initialize components
        self.embedding_engine = EmbeddingEngine()
        
        # Document classifier (for multi-collection mode)
        self.classifier = None
        if enable_multi_collection and Config.ENABLE_AUTO_CLASSIFICATION:
            self.classifier = DocumentClassifier()
            print("✓ Document classifier enabled")
        
        self.vector_store = VectorStore(
            db_path=db_path or Config.ENDEE_DB_PATH,
            collection_name=collection_name or Config.COLLECTION_NAME,
            enable_multi_collection=enable_multi_collection
        )
        self.document_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            classifier=self.classifier,
            enable_specialized_parsing=Config.ENABLE_SPECIALIZED_PARSING
        )
        
        # Multi-collection mode status
        if enable_multi_collection:
            print("✓ Multi-collection mode enabled")
        
        # LLM manager (handles OpenAI, Groq, Ollama)
        from llm_manager import LLMManager
        self.llm_manager = LLMManager()
        # Use the actual model from LLMManager, not config default
        self.chat_model = self.llm_manager.model if self.llm_manager.model else (chat_model or Config.CHAT_MODEL)
        
        # Conversation components (Phase 3)
        self.conversation_manager = None
        self.query_rewriter = None
        if enable_conversation:
            self.conversation_manager = ConversationManager()
            self.query_rewriter = QueryRewriter(llm_manager=self.llm_manager)
            print("✓ Conversational memory enabled")
        
        # User persona system (Phase 4)
        self.persona_manager = None
        if enable_personas:
            self.persona_manager = UserPersona(default_persona=default_persona)
            print(f"✓ User persona system enabled (default: {default_persona})")
        
        # Query routing system (Phase 5)
        self.query_router = None
        if enable_query_routing and enable_multi_collection:
            self.query_router = QueryRouter(
                use_llm=use_llm_routing,
                llm_manager=self.llm_manager if use_llm_routing else None
            )
            print(f"✓ Query routing enabled ({'LLM-based' if use_llm_routing else 'heuristic'})")
        
        print("✓ RAG Engine initialized")
    
    def index_documents(self, directory: str):
        """
        Index all documents from a directory into the vector store.
        Automatically skips already-indexed files.
        
        Args:
            directory: Path to directory containing documents
        """
        print(f"\n{'='*60}")
        print("📚 INDEXING DOCUMENTS")
        print(f"{'='*60}\n")
        
        # Step 0: Check already indexed files
        indexed_files = self.vector_store.get_indexed_files()
        if indexed_files:
            print(f"📋 Already indexed: {len(indexed_files)} files")
            for filename in sorted(indexed_files):
                print(f"  ✓ {filename}")
            print()
        
        # Step 1: Load documents
        print("Step 1: Loading documents...")
        all_documents = self.document_processor.load_documents(directory)
        
        if not all_documents:
            print("⚠️ No documents found to index")
            return
        
        # Filter out already-indexed documents
        documents = [doc for doc in all_documents if doc['metadata'].get('filename') not in indexed_files]
        
        if not documents:
            print(f"✓ All {len(all_documents)} documents already indexed! Nothing to do.")
            return
        
        print(f"📄 New documents to index: {len(documents)}")
        for doc in documents:
            print(f"  → {doc['metadata'].get('filename', 'Unknown')}")
        
        # Step 2: Chunk documents
        print("\nStep 2: Chunking documents...")
        chunks = self.document_processor.chunk_documents(documents)
        
        # Step 3: Generate embeddings
        print("\nStep 3: Generating embeddings...")
        chunks_with_embeddings = self.embedding_engine.embed_documents(chunks)
        
        # Step 4: Store in vector database
        print("Step 4: Storing in vector database...")
        
        # CRITICAL FIX: Preserve ALL metadata fields, not just text and filename
        vectors = [doc['embedding'] for doc in chunks_with_embeddings]
        
        # Enhanced metadata with all fields + indexing timestamp
        metadata = []
        for doc in chunks_with_embeddings:
            meta = {
                'text': doc['text'],
                'filename': doc['metadata'].get('filename', 'Unknown'),
                'source': doc['metadata'].get('source', ''),
                'type': doc['metadata'].get('type', ''),
                'chunk_id': doc['metadata'].get('chunk_id', 0),
                'total_chunks': doc['metadata'].get('total_chunks', 1),
                'indexed_at': datetime.now().isoformat()
            }
            
            # Add doc_type if in multi-collection mode
            if self.enable_multi_collection and 'doc_type' in doc['metadata']:
                meta['doc_type'] = doc['metadata']['doc_type']
            
            metadata.append(meta)
        
        # If multi-collection mode, route to appropriate collections
        if self.enable_multi_collection:
            # Group chunks by document type
            chunks_by_type = {}
            for i, doc in enumerate(chunks_with_embeddings):
                doc_type = doc['metadata'].get('doc_type', 'generic')
                if doc_type not in chunks_by_type:
                    chunks_by_type[doc_type] = {
                        'vectors': [],
                        'metadata': [],
                        'ids': []
                    }
                
                chunks_by_type[doc_type]['vectors'].append(vectors[i])
                chunks_by_type[doc_type]['metadata'].append(metadata[i])
            
            # Add to appropriate collections
            for doc_type, data in chunks_by_type.items():
                collection_name = Config.get_collection_name(doc_type)
                
                # Generate IDs for this collection
                if collection_name in self.vector_store.collections:
                    start_id = self.vector_store.collections[collection_name]['vector_count']
                else:
                    start_id = 0
                
                data['ids'] = [f"{collection_name}_{start_id + i}" for i in range(len(data['vectors']))]
                
                print(f"  → Adding {len(data['vectors'])} vectors to '{collection_name}' collection")
                self.vector_store.add_vectors_to_collection(
                    collection_name=collection_name,
                    vectors=data['vectors'],
                    metadata=data['metadata'],
                    ids=data['ids']
                )
        else:
            # Legacy single-collection mode
            start_id = self.vector_store.vector_count
            ids = [f"doc_{start_id + i}" for i in range(len(chunks_with_embeddings))]
            
            self.vector_store.add_vectors(vectors=vectors, metadata=metadata, ids=ids)
        
        print(f"\n{'='*60}")
        print("✓ INDEXING COMPLETE")
        print(f"{'='*60}\n")
        
        # Show stats
        stats = self.vector_store.get_stats()
        print(f"📊 Collection: {stats['collection_name']}")
        print(f"📊 Total vectors: {stats['total_vectors']}")
        print(f"📊 Dimension: {stats['dimension']}")
    
    def query(self, 
              question: str, 
              top_k: int = None,
              return_context: bool = False,
              similarity_threshold: float = 0.0,
              session_id: str = None,
              enable_rewrite: bool = True,
              persona: str = None,
              adapt_to_query: bool = False,
              collections: List[str] = None,
              auto_route: bool = True) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve (overrides persona default)
            return_context: Whether to return retrieved context
            similarity_threshold: Minimum similarity score (0.0 to 1.0) for results
            session_id: Conversation session ID for context tracking
            enable_rewrite: Enable query rewriting based on conversation history
            persona: User persona (beginner/intermediate/expert/researcher)
            adapt_to_query: Dynamically adapt persona based on query complexity
            collections: Specific collections to search (None = auto-route or all)
            auto_route: Auto-route query to relevant collections (only if collections=None)
            
        Returns:
            Generated answer (and context if return_context=True)
        """
        if not question or not question.strip():
            return "Please provide a valid question."
        
        # Apply persona settings (Phase 4)
        persona_profile = None
        if self.enable_personas and self.persona_manager:
            if adapt_to_query:
                persona_profile = self.persona_manager.adapt_to_query_complexity(question, persona)
            else:
                persona_profile = self.persona_manager.get_profile(persona)
            
            # Use persona's top_k if not explicitly overridden
            if top_k is None:
                top_k = persona_profile.top_k
        else:
            top_k = top_k or Config.TOP_K_RESULTS
        
        # Determine which collections to search (Phase 5)
        search_collections = collections  # Use explicit collections if provided
        routing_confidence = None
        routing_explanation = None
        
        if search_collections is None and self.enable_query_routing and self.query_router:
            # Auto-route based on query intent
            if auto_route:
                available = list(self.vector_store.collections.keys()) if self.enable_multi_collection else None
                search_collections = self.query_router.route_query(question, available)
                print(f"  Routing: {', '.join(search_collections)}")
                
                # Get routing confidence scores
                all_scores = self.query_router.get_intent_confidence(question)
                if all_scores and search_collections:
                    # Calculate confidence as average of selected collections
                    routing_confidence = sum(all_scores.get(c, 0) for c in search_collections) / len(search_collections)
                    routing_explanation = f"Query routed to {len(search_collections)} collection(s) based on keyword/phrase matching"
        
        original_question = question
        
        print(f"\n{'='*60}")
        print(f"🤔 Question: {question}")
        print(f"{'='*60}\n")
        
        # Step 0: Query rewriting (if conversation enabled)
        if self.enable_conversation and session_id and self.query_rewriter and enable_rewrite:
            history = self.conversation_manager.get_history(session_id, last_n=3)
            if history:
                print("Step 0: Rewriting query with conversation context...")
                question = self.query_rewriter.rewrite_with_context(
                    current_query=question,
                    conversation_history=history,
                    enable_rewrite=enable_rewrite
                )
                if question != original_question:
                    print(f"  Original: {original_question}")
                    print(f"  Rewritten: {question}")
                print()
        
        # Step 1: Embed the question (using rewritten query if applicable)
        print("Step 1: Embedding question...")
        question_embedding = self.embedding_engine.embed_text(question)
        
        # Step 2: Retrieve relevant chunks
        print(f"Step 2: Retrieving top-{top_k} relevant chunks...")
        
        # Use multi-collection search if collections specified and multi-collection enabled
        if search_collections and self.enable_multi_collection:
            results = self.vector_store.search_multi_collection(
                query_vector=question_embedding,
                collections=search_collections,
                top_k=top_k
            )
        else:
            # Default single-collection search
            results = self.vector_store.search(
                query_vector=question_embedding,
                top_k=top_k
            )
        
        # Filter by similarity threshold if specified
        if similarity_threshold > 0.0:
            results = [r for r in results if r['score'] >= similarity_threshold]
        
        if not results:
            return "I don't have enough context to answer that question. Please index some documents first."
        
        # Display retrieved context
        print("\n📝 Retrieved Context:")
        for i, result in enumerate(results):
            print(f"\n  [{i+1}] Similarity: {result['score']:.4f}")
            print(f"      Source: {result['metadata'].get('filename', 'Unknown')}")
            print(f"      Text: {result['text'][:100]}...")
        
        # Step 3: Build context
        context = self._build_context(results)
        
        # Step 4: Generate answer using LLM
        provider_name = self.llm_manager.provider or 'LLM'
        model_name = self.llm_manager.model or self.chat_model
        print(f"\nStep 3: Generating answer with {provider_name.upper()} ({model_name})...")
        if persona_profile:
            print(f"  Using persona: {persona_profile.name}")
            answer = self._generate_answer(
                question, 
                context,
                system_prompt=persona_profile.system_prompt,
                temperature=persona_profile.temperature,
                max_tokens=persona_profile.max_tokens
            )
        else:
            answer = self._generate_answer(question, context)
        
        print(f"\n{'='*60}")
        print("💡 Answer:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
        # Build metadata for return
        result_metadata = {
            'rewritten_query': question if question != original_question else None,
            'top_k': top_k,
            'model': self.chat_model,
            'num_results': len(results),
            'persona': persona_profile.name if persona_profile else None,
            'collections_searched': search_collections if search_collections else None,
            'routing_enabled': self.enable_query_routing and auto_route,
            'routing_confidence': routing_confidence,
            'routing_explanation': routing_explanation
        }
        
        # Step 5: Save conversation turn (if conversation enabled)
        if self.enable_conversation and session_id and self.conversation_manager:
            self.conversation_manager.add_turn(
                session_id=session_id,
                question=original_question,  # Store original question, not rewritten
                answer=answer,
                retrieved_context=[{'text': r['text'][:200], 'source': r['metadata'].get('filename')} for r in results],
                metadata=result_metadata
            )
        
        if return_context:
            return {
                'answer': answer,
                'context': results,
                'metadata': result_metadata
            }
        
        return answer
    
    def _build_context(self, results: List[Dict]) -> str:
        """
        Build context string from search results.
        
        Args:
            results: Search results from vector store
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results):
            source = result['metadata'].get('filename', 'Unknown')
            text = result['text']
            context_parts.append(f"[Source {i+1}: {source}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, 
                        question: str, 
                        context: str,
                        system_prompt: str = None,
                        temperature: float = 0.3,
                        max_tokens: int = 500) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context
            system_prompt: Custom system prompt (uses default if None)
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer
        """
        # Use default prompt if none provided
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that provides accurate answers based on given context."
        
        # Construct prompt
        prompt = f"""Context:
{context}

Question: {question}

Instructions:
1. Answer the question using the information from the context above
2. If the context doesn't contain enough information to answer, say so
3. Be accurate and follow the guidelines in the system prompt
4. Cite which source(s) you used when relevant (e.g., "According to Source 1...")

Answer:"""
        
        try:
            # Use LLM manager instead of direct OpenAI client
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            return self.llm_manager.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def interactive_mode(self):
        """Run interactive Q&A session."""
        print("\n" + "="*60)
        print("🤖 RAG INTERACTIVE MODE")
        print("="*60)
        print("Type your questions (or 'quit' to exit)")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("❓ You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    stats = self.vector_store.get_stats()
                    print(f"\n📊 Collection Stats:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    print()
                    continue
                
                # Query the system
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize RAG engine
        rag = RAGEngine()
        
        # Show current stats
        print("\n📊 Current Collection Stats:")
        stats = rag.vector_store.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Example query (will only work if documents are indexed)
        if stats['total_vectors'] > 0:
            print("\n" + "="*60)
            print("Testing query...")
            answer = rag.query("What is machine learning?")
        else:
            print("\n⚠️ No documents indexed yet.")
            print("Run: python main.py --mode index --docs data/documents")
            
    except Exception as e:
        print(f"Error: {e}")
