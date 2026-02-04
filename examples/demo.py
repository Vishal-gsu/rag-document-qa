"""
Demo Script - RAG System with Endee
This script demonstrates the complete RAG workflow.
"""
from rag_engine import RAGEngine
from pathlib import Path
import time


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_indexing():
    """Demonstrate document indexing."""
    print_section("PART 1: INDEXING DOCUMENTS")
    
    print("This demonstrates how documents are processed and indexed:\n")
    print("1. Documents are loaded from the data/documents folder")
    print("2. Text is split into chunks (500 chars with 50 char overlap)")
    print("3. Each chunk is converted to a 1536-dimensional vector")
    print("4. Vectors are stored in Endee Vector Database\n")
    
    input("Press Enter to start indexing...")
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Index documents
    docs_path = Path("data/documents")
    if docs_path.exists():
        rag.index_documents(str(docs_path))
    else:
        print("⚠️ No documents folder found. Please create data/documents/")
        print("   and add some .txt, .md, .pdf, or .docx files.\n")
    
    return rag


def demo_retrieval(rag: RAGEngine):
    """Demonstrate semantic retrieval."""
    print_section("PART 2: SEMANTIC RETRIEVAL")
    
    print("This demonstrates how the system retrieves relevant information:\n")
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How do vector databases work?",
        "What is RAG?"
    ]
    
    print("Example queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    print("\nLet's try the first query...\n")
    input("Press Enter to continue...")
    
    # Query the system
    query = queries[0]
    print(f"\n🔍 Query: '{query}'\n")
    print("Process:")
    print("  1. Convert query to embedding vector")
    print("  2. Search vector database for similar chunks")
    print("  3. Retrieve top-3 most relevant pieces\n")
    
    # This will show the full retrieval process
    rag.query(query, top_k=3)


def demo_generation(rag: RAGEngine):
    """Demonstrate answer generation."""
    print_section("PART 3: ANSWER GENERATION")
    
    print("This demonstrates how the system generates answers:\n")
    print("1. Retrieved context is added to the prompt")
    print("2. LLM generates an answer based on the context")
    print("3. Answer is grounded in your documents, not general knowledge\n")
    
    input("Press Enter to see another example...")
    
    # Query with different question
    query = "What are the types of machine learning?"
    rag.query(query, top_k=3)


def demo_comparison():
    """Show difference between RAG and standard LLM."""
    print_section("PART 4: RAG vs. STANDARD LLM")
    
    print("Key Differences:\n")
    print("Standard LLM:")
    print("  ❌ Limited to training data")
    print("  ❌ May hallucinate information")
    print("  ❌ No access to your proprietary documents")
    print("  ❌ Can't cite sources\n")
    
    print("RAG System:")
    print("  ✅ Uses your custom knowledge base")
    print("  ✅ Grounded in retrieved documents")
    print("  ✅ Can cite sources")
    print("  ✅ Reduces hallucination")
    print("  ✅ Up-to-date with your latest documents\n")


def demo_interactive(rag: RAGEngine):
    """Interactive Q&A session."""
    print_section("PART 5: INTERACTIVE DEMO")
    
    print("Now you can ask your own questions!\n")
    print("Suggested questions:")
    print("  - What is supervised learning?")
    print("  - Explain transformers in NLP")
    print("  - What are vector embeddings?")
    print("  - How does semantic search work?\n")
    
    print("Type 'done' to finish the demo.\n")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['done', 'exit', 'quit']:
                break
            
            rag.query(question, top_k=3)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


def show_internals():
    """Explain system internals."""
    print_section("UNDERSTANDING THE INTERNALS")
    
    print("🔧 System Components:\n")
    
    print("1. Document Processor (document_processor.py)")
    print("   - Loads .txt, .md, .pdf, .docx files")
    print("   - Splits text into overlapping chunks")
    print("   - Preserves context across chunks\n")
    
    print("2. Embedding Engine (embedding_engine.py)")
    print("   - Uses OpenAI's text-embedding-3-small model")
    print("   - Converts text → 1536-dimensional vectors")
    print("   - Batch processing for efficiency\n")
    
    print("3. Vector Store (vector_store.py)")
    print("   - Endee database interface")
    print("   - Stores vectors with metadata")
    print("   - Cosine similarity search\n")
    
    print("4. RAG Engine (rag_engine.py)")
    print("   - Orchestrates the entire pipeline")
    print("   - Retrieval + Generation")
    print("   - Context building and prompt engineering\n")
    
    print("💾 Data Flow:\n")
    print("  Document → Chunks → Embeddings → Vector DB")
    print("  Query → Embedding → Search → Context → LLM → Answer\n")


def main():
    """Run the complete demo."""
    print("\n" + "🎓"*35)
    print("\n         RAG SYSTEM DEMONSTRATION")
    print("     Using Endee Vector Database")
    print("\n" + "🎓"*35)
    
    print("\n\nThis demo will walk you through:")
    print("  1. Document indexing")
    print("  2. Semantic retrieval")
    print("  3. Answer generation")
    print("  4. RAG vs standard LLM")
    print("  5. Interactive Q&A")
    print("  6. System internals\n")
    
    input("Press Enter to start the demo...")
    
    try:
        # Part 1: Indexing
        rag = demo_indexing()
        time.sleep(1)
        
        # Part 2: Retrieval
        demo_retrieval(rag)
        time.sleep(1)
        
        # Part 3: Generation
        demo_generation(rag)
        time.sleep(1)
        
        # Part 4: Comparison
        demo_comparison()
        time.sleep(1)
        
        # Part 5: Interactive
        demo_interactive(rag)
        
        # Part 6: Internals
        show_internals()
        
        # Conclusion
        print_section("DEMO COMPLETE")
        print("🎉 You now understand how RAG systems work!\n")
        print("Next steps:")
        print("  1. Add your own documents to data/documents/")
        print("  2. Re-index: python main.py --mode index --docs data/documents")
        print("  3. Query: python main.py --mode query --question 'your question'")
        print("  4. Interactive: python main.py --mode interactive\n")
        print("📚 Read README.md for detailed documentation")
        print("💡 Check docs/THEORY.md for theoretical concepts\n")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        print("\nMake sure:")
        print("  1. You've set OPENAI_API_KEY in .env file")
        print("  2. You've installed dependencies: pip install -r requirements.txt")
        print("  3. You have documents in data/documents/\n")


if __name__ == "__main__":
    main()
