"""
Interview Demonstration Script for RAG System
==============================================

This script demonstrates all 5 phases of the enhanced RAG system:
- Phase 1: Multi-collection infrastructure
- Phase 2: Specialized parsers
- Phase 3: Conversational memory
- Phase 4: User personas  
- Phase 5: Query routing intelligence

Run this to showcase the system capabilities in your interview.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag_engine import RAGEngine
from config import Config
import time


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_query():
    """Demo: Basic RAG query without any enhancements."""
    print_section("DEMO 1: Basic RAG Query")
    
    rag = RAGEngine(
        enable_multi_collection=False,
        enable_conversation=False,
        enable_personas=False,
        enable_query_routing=False
    )
    
    question = "What is machine learning?"
    print(f"Question: {question}\n")
    
    start = time.time()
    result = rag.query(question, top_k=3, return_context=True)
    elapsed = time.time() - start
    
    print(f"Answer: {result['answer']}\n")
    print(f"Sources: {len(result['context'])} documents")
    print(f"Time: {elapsed:.2f}s")
    print("\n✓ Basic RAG functionality working!")


def demo_multi_collection():
    """Demo: Multi-collection document organization."""
    print_section("DEMO 2: Multi-Collection Infrastructure (Phase 1)")
    
    rag = RAGEngine(enable_multi_collection=True)
    
    # Show collections
    collections = list(rag.vector_store.collections.keys())
    print(f"Available collections: {collections}\n")
    
    # Query specific collection
    question = "What is machine learning?"
    print(f"Question: {question}")
    print(f"Searching only: textbooks collection\n")
    
    result = rag.query(question, collections=['textbooks'], top_k=3, return_context=True)
    
    print(f"Answer: {result['answer'][:200]}...\n")
    print(f"Sources from textbooks: {len(result['context'])}")
    print("\n✓ Multi-collection retrieval working!")


def demo_conversational_memory():
    """Demo: Conversational memory and query rewriting."""
    print_section("DEMO 3: Conversational Memory (Phase 3)")
    
    rag = RAGEngine(enable_conversation=True)
    
    session_id = "demo_session"
    
    # First question
    q1 = "What is machine learning?"
    print(f"Q1: {q1}")
    result1 = rag.query(q1, top_k=3, return_context=True, session_id=session_id, enable_rewrite=True)
    print(f"A1: {result1['answer'][:150]}...\n")
    
    # Follow-up question (requires context)
    q2 = "What about deep learning?"
    print(f"Q2: {q2}")
    result2 = rag.query(q2, top_k=3, return_context=True, session_id=session_id, enable_rewrite=True)
    print(f"A2: {result2['answer'][:150]}...\n")
    
    # Show rewriting
    if result2.get('metadata', {}).get('rewritten_query'):
        print(f"🔄 Query was rewritten:")
        print(f"   Original: {q2}")
        print(f"   Rewritten: {result2['metadata']['rewritten_query']}")
    
    # Show conversation history
    if rag.conversation_manager:
        history = rag.conversation_manager.get_history(session_id)
        print(f"\n✓ Conversation has {len(history)} turns in memory!")


def demo_user_personas():
    """Demo: User persona adaptation."""
    print_section("DEMO 4: User Personas (Phase 4)")
    
    rag = RAGEngine(enable_personas=True)
    
    question = "What is machine learning?"
    
    # Beginner persona
    print("🌱 BEGINNER PERSONA (detailed explanations)")
    result_beginner = rag.query(question, persona='beginner', return_context=True)
    print(f"Answer length: {len(result_beginner['answer'])} chars")
    print(f"Sources: {len(result_beginner['context'])} (fewer for simplicity)")
    print(f"Preview: {result_beginner['answer'][:120]}...\n")
    
    # Expert persona
    print("🎯 EXPERT PERSONA (concise, technical)")
    result_expert = rag.query(question, persona='expert', return_context=True)
    print(f"Answer length: {len(result_expert['answer'])} chars")
    print(f"Sources: {len(result_expert['context'])} (more for depth)")
    print(f"Preview: {result_expert['answer'][:120]}...\n")
    
    print("✓ Personalization adapts retrieval and generation!")


def demo_query_routing():
    """Demo: Intelligent query routing."""
    print_section("DEMO 5: Query Routing Intelligence (Phase 5)")
    
    rag = RAGEngine(
        enable_multi_collection=True,
        enable_query_routing=True,
        use_llm_routing=False  # Use fast heuristic routing
    )
    
    # Test different query types
    queries = [
        ("Find research papers about BERT", "Expected: research_papers"),
        ("What Python skills do candidates have?", "Expected: resumes"),
        ("Explain machine learning chapter 3", "Expected: textbooks"),
        ("What is in the documents?", "Expected: general_docs (fallback)")
    ]
    
    for query, expected in queries:
        print(f"\nQuery: '{query}'")
        print(f"   {expected}")
        
        # Get routing decision
        if rag.query_router:
            routed_collections = rag.query_router.route_query(query, max_collections=2)
            confidence = rag.query_router.get_intent_confidence(query)
            explanation = rag.query_router.explain_routing(query)
            
            print(f"   Routed to: {routed_collections}")
            print(f"   Confidence: {max(confidence.values()):.0%}")
            print(f"   Explanation: {explanation}")
    
    print("\n✓ Query routing intelligently selects relevant collections!")


def demo_full_system():
    """Demo: All features combined."""
    print_section("DEMO 6: Full System Integration (All Phases)")
    
    rag = RAGEngine(
        enable_multi_collection=True,
        enable_conversation=True,
        enable_personas=True,
        enable_query_routing=True,
        use_llm_routing=False
    )
    
    session_id = "full_demo"
    
    # Complex conversation with all features
    questions = [
        ("What is machine learning?", "intermediate"),
        ("Show me research papers about it", "researcher"),
        ("Explain it like I'm a beginner", "beginner"),
    ]
    
    for i, (question, persona) in enumerate(questions, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Question: {question}")
        print(f"Persona: {persona}")
        
        result = rag.query(
            question=question,
            session_id=session_id,
            enable_rewrite=True,
            persona=persona,
            auto_route=True,
            return_context=True
        )
        
        print(f"Answer: {result['answer'][:150]}...")
        
        # Show metadata
        metadata = result.get('metadata', {})
        if metadata.get('rewritten_query'):
            print(f"  🔄 Rewritten: {metadata['rewritten_query']}")
        if metadata.get('collections_searched'):
            print(f"  🗂️ Collections: {', '.join(metadata['collections_searched'])}")
        if metadata.get('routing_confidence'):
            print(f"  🎯 Confidence: {metadata['routing_confidence']:.0%}")
    
    # Final stats
    if rag.conversation_manager:
        stats = rag.conversation_manager.get_session_stats(session_id)
        print(f"\n📊 Session Stats:")
        print(f"   Total turns: {stats.get('turn_count', 0)}")
        print(f"   Rewrite rate: {stats.get('rewrite_rate', 0):.0%}")
    
    print("\n✓ Full system demonstration complete!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  RAG SYSTEM - INTERVIEW DEMONSTRATION")
    print("  Multi-Collection • Conversational • Personalized • Intelligent")
    print("=" * 70)
    
    try:
        # Run each demo
        demos = [
            ("Basic RAG", demo_basic_query),
            ("Multi-Collection", demo_multi_collection),
            ("Conversational Memory", demo_conversational_memory),
            ("User Personas", demo_user_personas),
            ("Query Routing", demo_query_routing),
            ("Full Integration", demo_full_system)
        ]
        
        print("\nAvailable Demonstrations:")
        for i, (name, _) in enumerate(demos, 1):
            print(f"  {i}. {name}")
        
        print("\n" + "-" * 70)
        choice = input("\nSelect demo (1-6, or 'all' for all demos): ").strip().lower()
        
        if choice == 'all':
            for name, demo_func in demos:
                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ Error in {name}: {e}")
                input("\nPress Enter to continue to next demo...")
        elif choice.isdigit() and 1 <= int(choice) <= 6:
            demos[int(choice) - 1][1]()
        else:
            print("Invalid choice. Running full integration demo...")
            demo_full_system()
        
        print("\n" + "=" * 70)
        print("  DEMONSTRATION COMPLETE!")
        print("=" * 70)
        
        print("\n📝 Key Talking Points for Interview:")
        print("1. Multi-collection infrastructure for organized document management")
        print("2. Specialized parsers for different document types")
        print("3. Conversational memory with intelligent query rewriting")
        print("4. User personas that adapt retrieval & generation")
        print("5. Query routing for automatic collection selection")
        print("6. All features work together seamlessly")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
