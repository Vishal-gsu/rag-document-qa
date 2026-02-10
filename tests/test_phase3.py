"""
Phase 3 Tests: Conversational Memory System
Tests conversation tracking, query rewriting, and multi-turn dialogue.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from conversation_manager import ConversationManager
from query_rewriter import QueryRewriter
from llm_manager import LLMManager
from rag_engine import RAGEngine
from config import Config
import shutil


class TestConversationManager:
    """Test conversation storage and session management."""
    
    def setup_method(self):
        """Setup test environment."""
        # Use test storage path
        self.test_storage = "./data/test_conversations"
        self.manager = ConversationManager(storage_path=self.test_storage)
    
    def teardown_method(self):
        """Cleanup test data."""
        test_path = Path(self.test_storage)
        if test_path.exists():
            shutil.rmtree(test_path)
    
    def test_create_session(self):
        """Test session creation."""
        session_id = self.manager.create_session(user_id="test_user")
        
        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in self.manager.sessions
        
        session = self.manager.sessions[session_id]
        assert session['user_id'] == "test_user"
        assert len(session['turns']) == 0
        print("✓ Session creation works")
    
    def test_add_turn(self):
        """Test adding conversation turns."""
        session_id = self.manager.create_session()
        
        # Add first turn
        self.manager.add_turn(
            session_id=session_id,
            question="What is machine learning?",
            answer="Machine learning is a subset of AI...",
            retrieved_context=[{'text': 'ML is...', 'source': 'doc1.md'}]
        )
        
        history = self.manager.get_history(session_id)
        assert len(history) == 1
        assert history[0]['question'] == "What is machine learning?"
        assert history[0]['turn_number'] == 1
        
        # Add second turn
        self.manager.add_turn(
            session_id=session_id,
            question="What about deep learning?",
            answer="Deep learning uses neural networks..."
        )
        
        history = self.manager.get_history(session_id)
        assert len(history) == 2
        assert history[1]['turn_number'] == 2
        print("✓ Adding conversation turns works")
    
    def test_get_recent_context(self):
        """Test getting recent conversation context."""
        session_id = self.manager.create_session()
        
        # Add multiple turns
        for i in range(5):
            self.manager.add_turn(
                session_id=session_id,
                question=f"Question {i+1}",
                answer=f"Answer {i+1}"
            )
        
        # Get last 3 turns
        context = self.manager.get_recent_context(session_id, last_n=3)
        
        assert "Question 3" in context
        assert "Question 4" in context
        assert "Question 5" in context
        assert "Question 1" not in context
        print("✓ Getting recent context works")
    
    def test_save_and_load_session(self):
        """Test persisting sessions to disk."""
        session_id = self.manager.create_session()
        self.manager.add_turn(
            session_id=session_id,
            question="Test question",
            answer="Test answer"
        )
        
        # Save session
        success = self.manager.save_session(session_id)
        assert success
        
        # Create new manager and load
        new_manager = ConversationManager(storage_path=self.test_storage)
        loaded = new_manager._load_session(session_id)
        
        assert loaded
        assert session_id in new_manager.sessions
        
        history = new_manager.get_history(session_id)
        assert len(history) == 1
        assert history[0]['question'] == "Test question"
        print("✓ Session persistence works")
    
    def test_session_stats(self):
        """Test session statistics."""
        session_id = self.manager.create_session()
        
        self.manager.add_turn(
            session_id=session_id,
            question="Short question?",
            answer="Very detailed and comprehensive answer that goes on for quite a while."
        )
        
        stats = self.manager.get_session_stats(session_id)
        
        assert stats['turn_count'] == 1
        assert stats['avg_question_length'] > 0
        assert stats['avg_answer_length'] > stats['avg_question_length']
        print("✓ Session statistics work")


class TestQueryRewriter:
    """Test query rewriting with conversation context."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create mock LLM manager for testing
        self.llm_manager = None  # Will use actual LLM in real tests
        self.rewriter = QueryRewriter(llm_manager=self.llm_manager)
    
    def test_needs_rewriting_pronouns(self):
        """Test detection of pronoun-based queries."""
        assert self.rewriter._needs_rewriting("What about it?")
        assert self.rewriter._needs_rewriting("Tell me more about this")
        assert self.rewriter._needs_rewriting("How does that work?")
        print("✓ Pronoun detection works")
    
    def test_needs_rewriting_context_phrases(self):
        """Test detection of context-dependent phrases."""
        assert self.rewriter._needs_rewriting("What about deep learning?")
        assert self.rewriter._needs_rewriting("Tell me more")
        assert self.rewriter._needs_rewriting("Also, what are RNNs?")
        print("✓ Context phrase detection works")
    
    def test_needs_rewriting_short_queries(self):
        """Test detection of very short queries."""
        assert self.rewriter._needs_rewriting("And CNNs?")
        assert self.rewriter._needs_rewriting("Why?")
        print("✓ Short query detection works")
    
    def test_standalone_queries_skip_rewriting(self):
        """Test that standalone queries are not flagged for rewriting."""
        assert not self.rewriter._needs_rewriting("What is machine learning?")
        assert not self.rewriter._needs_rewriting("Explain neural networks in detail")
        print("✓ Standalone query detection works")
    
    def test_build_context_prompt(self):
        """Test prompt building for LLM rewriting."""
        history = [
            {'question': 'What is machine learning?', 'answer': 'ML is a subset of AI...'},
            {'question': 'How does it work?', 'answer': 'ML works by training models on data...'}
        ]
        
        prompt = self.rewriter._build_context_prompt(
            current_query="What about deep learning?",
            history=history
        )
        
        assert "machine learning" in prompt.lower()
        assert "What about deep learning?" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt
        print("✓ Context prompt building works")
    
    def test_rewrite_without_context(self):
        """Test rewriting when no context is available."""
        result = self.rewriter.rewrite_with_context(
            current_query="What is ML?",
            conversation_history=[],
            enable_rewrite=True
        )
        
        # Should return original query unchanged
        assert result == "What is ML?"
        print("✓ No-context rewriting works")


class TestConversationalRAG:
    """Test end-to-end conversational RAG system."""
    
    def setup_method(self):
        """Setup test environment."""
        # Clean up any existing test data
        test_conv_path = Path("./data/test_rag_conversations")
        if test_conv_path.exists():
            shutil.rmtree(test_conv_path)
        
        # Initialize RAG engine with conversation enabled
        self.rag = RAGEngine(enable_conversation=True)
        
        # Override conversation storage path for testing
        if self.rag.conversation_manager:
            self.rag.conversation_manager.storage_path = test_conv_path
            test_conv_path.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Cleanup test data."""
        test_path = Path("./data/test_rag_conversations")
        if test_path.exists():
            shutil.rmtree(test_path)
    
    def test_rag_initialization_with_conversation(self):
        """Test RAG engine initializes with conversation components."""
        assert self.rag.enable_conversation
        assert self.rag.conversation_manager is not None
        assert self.rag.query_rewriter is not None
        print("✓ RAG with conversation initialized")
    
    def test_conversation_disabled_mode(self):
        """Test RAG engine can run without conversation."""
        rag_no_conv = RAGEngine(enable_conversation=False)
        
        assert not rag_no_conv.enable_conversation
        assert rag_no_conv.conversation_manager is None
        assert rag_no_conv.query_rewriter is None
        print("✓ RAG without conversation works")
    
    def test_session_creation_and_tracking(self):
        """Test conversation sessions are tracked properly."""
        if not self.rag.conversation_manager:
            pytest.skip("Conversation manager not available")
        
        session_id = self.rag.conversation_manager.create_session(user_id="test_user")
        
        # Add a turn manually
        self.rag.conversation_manager.add_turn(
            session_id=session_id,
            question="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence..."
        )
        
        # Verify history
        history = self.rag.conversation_manager.get_history(session_id)
        assert len(history) == 1
        assert history[0]['question'] == "What is machine learning?"
        print("✓ Session tracking works")


def run_all_tests():
    """Run all Phase 3 tests."""
    print("\n" + "="*60)
    print("PHASE 3 TESTS: Conversational Memory System")
    print("="*60 + "\n")
    
    # Test 1: ConversationManager
    print("Test Suite 1: Conversation Manager")
    print("-" * 60)
    test_conv = TestConversationManager()
    test_conv.setup_method()
    try:
        test_conv.test_create_session()
        test_conv.test_add_turn()
        test_conv.test_get_recent_context()
        test_conv.test_save_and_load_session()
        test_conv.test_session_stats()
    finally:
        test_conv.teardown_method()
    
    # Test 2: QueryRewriter
    print("\nTest Suite 2: Query Rewriter")
    print("-" * 60)
    test_rewriter = TestQueryRewriter()
    test_rewriter.setup_method()
    test_rewriter.test_needs_rewriting_pronouns()
    test_rewriter.test_needs_rewriting_context_phrases()
    test_rewriter.test_needs_rewriting_short_queries()
    test_rewriter.test_standalone_queries_skip_rewriting()
    test_rewriter.test_build_context_prompt()
    test_rewriter.test_rewrite_without_context()
    
    # Test 3: Conversational RAG Integration
    print("\nTest Suite 3: Conversational RAG Integration")
    print("-" * 60)
    test_rag = TestConversationalRAG()
    test_rag.setup_method()
    try:
        test_rag.test_rag_initialization_with_conversation()
        test_rag.test_conversation_disabled_mode()
        test_rag.test_session_creation_and_tracking()
    finally:
        test_rag.teardown_method()
    
    print("\n" + "="*60)
    print("✅ ALL PHASE 3 TESTS PASSED!")
    print("="*60 + "\n")
    print("Phase 3 Implementation Summary:")
    print("  ✓ ConversationManager: Session tracking with pickle persistence")
    print("  ✓ QueryRewriter: LLM-based query expansion with context")
    print("  ✓ RAGEngine: Integrated conversation + query rewriting")
    print("  ✓ Streamlit App: UI for conversation history and session management")
    print("\nNext Steps:")
    print("  - Test with real documents and multi-turn conversations")
    print("  - Verify LLM-based query rewriting with actual API calls")
    print("  - Move to Phase 4: User Persona System")


if __name__ == "__main__":
    run_all_tests()
