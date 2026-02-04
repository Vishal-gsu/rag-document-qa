"""
RAG Engine Module
Orchestrates the complete RAG pipeline: retrieval + generation.
"""
from typing import List, Dict, Optional
from openai import OpenAI
from config import Config
from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from vector_store import VectorStore


class RAGEngine:
    """
    Retrieval Augmented Generation Engine.
    Combines document retrieval with LLM generation.
    """
    
    def __init__(self, 
                 db_path: str = None,
                 collection_name: str = None,
                 chat_model: str = None):
        """
        Initialize RAG engine.
        
        Args:
            db_path: Path to vector database
            collection_name: Name of the collection
            chat_model: OpenAI chat model to use
        """
        # Validate configuration
        Config.validate()
        
        # Initialize components
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore(
            db_path=db_path or Config.ENDEE_DB_PATH,
            collection_name=collection_name or Config.COLLECTION_NAME
        )
        self.document_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # LLM client
        self.chat_model = chat_model or Config.CHAT_MODEL
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
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
        # Extract vectors and metadata for Endee
        vectors = [doc['embedding'] for doc in chunks_with_embeddings]
        metadata = [{'text': doc['text'], 'filename': doc['metadata'].get('filename', 'Unknown')} 
                   for doc in chunks_with_embeddings]
        
        # Generate IDs starting from current vector count
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
              similarity_threshold: float = 0.0) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            return_context: Whether to return retrieved context
            similarity_threshold: Minimum similarity score (0.0 to 1.0) for results
            
        Returns:
            Generated answer (and context if return_context=True)
        """
        if not question or not question.strip():
            return "Please provide a valid question."
        
        top_k = top_k or Config.TOP_K_RESULTS
        
        print(f"\n{'='*60}")
        print(f"🤔 Question: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Embed the question
        print("Step 1: Embedding question...")
        question_embedding = self.embedding_engine.embed_text(question)
        
        # Step 2: Retrieve relevant chunks
        print(f"Step 2: Retrieving top-{top_k} relevant chunks...")
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
        print(f"\nStep 3: Generating answer with {self.chat_model}...")
        answer = self._generate_answer(question, context)
        
        print(f"\n{'='*60}")
        print("💡 Answer:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
        if return_context:
            return {
                'answer': answer,
                'context': results
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
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        # Construct prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information from the context above
2. If the context doesn't contain enough information to answer, say so
3. Be concise and accurate
4. Cite which source(s) you used (e.g., "According to Source 1...")

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based on given context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
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
