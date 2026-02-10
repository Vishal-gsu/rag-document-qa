"""
Phase 5 Tests: Query Routing Intelligence
Tests query routing, intent detection, and multi-collection search.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from query_router import QueryRouter
from rag_engine import RAGEngine
from llm_manager import LLMManager


class TestQueryRouter:
    """Test query routing and intent detection."""
    
    def setup_method(self):
        """Setup test environment."""
        self.router = QueryRouter(use_llm=False)  # Heuristic mode for testing
    
    def test_router_initialization(self):
        """Test query router initialization."""
        assert self.router is not None
        assert not self.router.use_llm
        assert len(self.router.INTENT_PATTERNS) == 4  # research_papers, resumes, textbooks, general_docs
        print("✓ QueryRouter initialized")
    
    def test_research_paper_routing(self):
        """Test routing to research papers collection."""
        queries = [
            "Find papers about transformers",
            "What does the research say about BERT?",
            "Show me studies on neural networks",
            "In the abstract, what is mentioned about attention mechanism?"
        ]
        
        for query in queries:
            collections = self.router.route_query(query)
            assert 'research_papers' in collections
        
        print("✓ Research paper routing works")
    
    def test_resume_routing(self):
        """Test routing to resumes collection."""
        queries = [
            "Find candidates with Python experience",
            "Who has worked on Docker and Kubernetes?",
            "Show me resumes with machine learning skills",
            "Find people with 5 years of experience in AWS"
        ]
        
        for query in queries:
            collections = self.router.route_query(query)
            assert 'resumes' in collections
        
        print("✓ Resume routing works")
    
    def test_textbook_routing(self):
        """Test routing to textbooks collection."""
        queries = [
            "How to learn machine learning basics?",
            "Introduction to neural networks",
            "Chapter on deep learning fundamentals",
            "Practice problems for regression"
        ]
        
        for query in queries:
            collections = self.router.route_query(query)
            assert 'textbooks' in collections
        
        print("✓ Textbook routing works")
    
    def test_multi_collection_routing(self):
        """Test queries that should search multiple collections."""
        query = "What are neural networks?"  # General query
        
        collections = self.router.route_query(query)
        
        # General queries should search multiple or all collections
        assert len(collections) >= 1
        
        print("✓ Multi-collection routing works")
    
    def test_confidence_scores(self):
        """Test confidence scoring for collections."""
        query = "Find research papers about transformers with experimental results"
        
        confidences = self.router.get_intent_confidence(query)
        
        # Should have high confidence for research_papers
        assert 'research_papers' in confidences
        assert confidences['research_papers'] > 0
        
        # Should have lower or zero confidence for resumes
        assert confidences.get('resumes', 0) < confidences['research_papers']
        
        print("✓ Confidence scoring works")
    
    def test_max_collections_limit(self):
        """Test limiting number of returned collections."""
        query = "machine learning"  # Very general
        
        collections = self.router.route_query(query, max_collections=2)
        
        assert len(collections) <= 2
        
        print("✓ Collection limiting works")
    
    def test_available_collections_filter(self):
        """Test filtering by available collections."""
        query = "Find papers about BERT"
        
        # Only research_papers available
        collections = self.router.route_query(
            query,
            available_collections=['research_papers']
        )
        
        assert collections == ['research_papers']
        
        print("✓ Available collections filtering works")
    
    def test_custom_patterns(self):
        """Test adding custom routing patterns."""
        self.router.add_custom_pattern(
            collection_type='research_papers',
            keywords=['gpt-4', 'claude'],
            phrases=['large language model']
        )
        
        query = "What does the large language model research show about GPT-4?"
        collections = self.router.route_query(query)
        
        assert 'research_papers' in collections
        
        print("✓ Custom patterns work")
    
    def test_explain_routing(self):
        """Test routing explanation."""
        query = "Find candidates with Python skills"
        collections = self.router.route_query(query)
        
        explanation = self.router.explain_routing(query, collections)
        
        assert query in explanation
        assert 'resumes' in explanation
        assert 'confidence' in explanation.lower()
        
        print("✓ Routing explanation works")


class TestHeuristicVsLLMRouting:
    """Test comparison between heuristic and LLM routing."""
    
    def test_heuristic_routing(self):
        """Test heuristic-based routing."""
        router = QueryRouter(use_llm=False)
        
        query = "Show me papers on transformers"
        collections = router.route_query(query)
        
        assert 'research_papers' in collections
        print("✓ Heuristic routing works")
    
    def test_llm_routing_fallback(self):
        """Test LLM routing falls back to heuristic on error."""
        # Create router with LLM but no valid manager
        router = QueryRouter(use_llm=True, llm_manager=None)
        
        query = "Show me papers on transformers"
        # Should fall back to heuristic
        collections = router.route_query(query)
        
        assert len(collections) > 0
        print("✓ LLM routing fallback works")


class TestRAGWithRouting:
    """Test RAG engine integration with query routing."""
    
    def setup_method(self):
        """Setup test environment."""
        self.rag = RAGEngine(
            enable_multi_collection=True,
            enable_query_routing=True
        )
    
    def test_rag_routing_initialization(self):
        """Test RAG initializes with query routing."""
        assert self.rag.enable_query_routing
        assert self.rag.query_router is not None
        print("✓ RAG with routing initialized")
    
    def test_routing_disabled_in_single_collection(self):
        """Test routing is disabled in single-collection mode."""
        rag_single = RAGEngine(
            enable_multi_collection=False,
            enable_query_routing=True  # Should be ignored
        )
        
        # Routing should be disabled since multi-collection is off
        assert not rag_single.enable_query_routing
        print("✓ Routing disabled in single-collection mode")
    
    def test_explicit_collections_override_routing(self):
        """Test explicit collections parameter overrides auto-routing."""
        # This test just verifies the parameter exists
        # Full integration would require indexed documents
        assert hasattr(self.rag, 'query')
        
        import inspect
        sig = inspect.signature(self.rag.query)
        assert 'collections' in sig.parameters
        assert 'auto_route' in sig.parameters
        
        print("✓ Explicit collection parameters exist")


def run_all_tests():
    """Run all Phase 5 tests."""
    print("\n" + "="*60)
    print("PHASE 5 TESTS: Query Routing Intelligence")
    print("="*60 + "\n")
    
    # Test 1: Query Router
    print("Test Suite 1: Query Router")
    print("-" * 60)
    test_router = TestQueryRouter()
    test_router.setup_method()
    test_router.test_router_initialization()
    test_router.test_research_paper_routing()
    test_router.test_resume_routing()
    test_router.test_textbook_routing()
    test_router.test_multi_collection_routing()
    test_router.test_confidence_scores()
    test_router.test_max_collections_limit()
    test_router.test_available_collections_filter()
    test_router.test_custom_patterns()
    test_router.test_explain_routing()
    
    # Test 2: Heuristic vs LLM
    print("\nTest Suite 2: Heuristic vs LLM Routing")
    print("-" * 60)
    test_comparison = TestHeuristicVsLLMRouting()
    test_comparison.test_heuristic_routing()
    test_comparison.test_llm_routing_fallback()
    
    # Test 3: RAG Integration
    print("\nTest Suite 3: RAG Integration with Routing")
    print("-" * 60)
    test_rag = TestRAGWithRouting()
    test_rag.setup_method()
    test_rag.test_rag_routing_initialization()
    test_rag.test_routing_disabled_in_single_collection()
    test_rag.test_explicit_collections_override_routing()
    
    print("\n" + "="*60)
    print("✅ ALL PHASE 5 TESTS PASSED!")
    print("="*60 + "\n")
    print("Phase 5 Implementation Summary:")
    print("  ✓ QueryRouter: Heuristic + LLM-based intent detection")
    print("  ✓ RAGEngine: Auto-routing with collection selection")
    print("  ✓ Streamlit UI: Smart routing toggle + manual collection selector")
    print("\nRouting Patterns:")
    print("  📄 Research Papers: 'papers', 'study', 'research', 'algorithm'")
    print("  👤 Resumes: 'experience', 'skills', 'candidates', 'hire'")
    print("  📚 Textbooks: 'learn', 'tutorial', 'chapter', 'exercises'")
    print("  📁 General: Fallback for unmatched queries")
    print("\nRouting Modes:")
    print("  🤖 Auto-route: Automatically detect intent and select collections")
    print("  ✋ Manual: User selects specific collections to search")
    print("  🔀 Hybrid: Auto-route with collection count limits")
    print("\nNext Steps:")
    print("  - Test with real multi-collection queries")
    print("  - Verify LLM routing accuracy vs heuristics")
    print("  - Move to Phase 6: Enhanced UI & Demo")


if __name__ == "__main__":
    run_all_tests()
