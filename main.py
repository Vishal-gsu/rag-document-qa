"""
Main Application Entry Point
Command-line interface for the RAG system.
"""
import argparse
import sys
from pathlib import Path
from rag_engine import RAGEngine
from config import Config


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System with Endee Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python main.py --mode index --docs data/documents
  
  # Query the system
  python main.py --mode query --question "What is machine learning?"
  
  # Interactive mode
  python main.py --mode interactive
  
  # Clear the database
  python main.py --mode clear
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['index', 'query', 'interactive', 'clear', 'stats'],
        required=True,
        help='Operation mode'
    )
    
    parser.add_argument(
        '--docs',
        type=str,
        help='Directory containing documents to index (for index mode)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Question to ask (for query mode)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=Config.TOP_K_RESULTS,
        help=f'Number of chunks to retrieve (default: {Config.TOP_K_RESULTS})'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG engine
        print("\n🚀 Initializing RAG System...")
        rag = RAGEngine()
        
        # Execute based on mode
        if args.mode == 'index':
            if not args.docs:
                print("✗ Error: --docs argument required for index mode")
                sys.exit(1)
            
            docs_path = Path(args.docs)
            if not docs_path.exists():
                print(f"✗ Error: Directory not found: {args.docs}")
                sys.exit(1)
            
            rag.index_documents(args.docs)
        
        elif args.mode == 'query':
            if not args.question:
                print("✗ Error: --question argument required for query mode")
                sys.exit(1)
            
            rag.query(args.question, top_k=args.top_k)
        
        elif args.mode == 'interactive':
            rag.interactive_mode()
        
        elif args.mode == 'clear':
            confirm = input("⚠️ Are you sure you want to clear the database? (yes/no): ")
            if confirm.lower() == 'yes':
                rag.vector_store.clear_collection()
                print("✓ Database cleared")
            else:
                print("Cancelled")
        
        elif args.mode == 'stats':
            stats = rag.vector_store.get_stats()
            print("\n📊 Collection Statistics:")
            print("=" * 50)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print("=" * 50)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # ASCII Art Banner
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║        RAG System with Endee Vector Database         ║
    ║                                                       ║
    ║     Retrieval Augmented Generation for Documents     ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)
    
    main()
