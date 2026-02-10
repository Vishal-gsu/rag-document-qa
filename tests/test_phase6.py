"""
Phase 6 Tests: Enhanced UI & Demo Features
===========================================

Tests for:
- Enhanced conversation visualization
- Routing explanation display
- Query performance tracking
- Export/import functionality
- Demo script validation
"""

import unittest
import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import RAGEngine
from config import Config


class TestEnhancedConversationVisualization(unittest.TestCase):
    """Test enhanced conversation features."""
    
    def setUp(self):
        """Initialize RAG with conversation enabled."""
        self.rag = RAGEngine(enable_conversation=True)
        self.session_id = "test_viz_session"
    
    def test_conversation_metadata_tracking(self):
        """Test that conversation tracks comprehensive metadata."""
        # Ask multiple questions
        questions = [
            "What is machine learning?",
            "What about deep learning?",
            "Show me examples"
        ]
        
        for q in questions:
            self.rag.query(
                question=q,
                session_id=self.session_id,
                enable_rewrite=True,
                return_context=True
            )
        
        # Get history
        history = self.rag.conversation_manager.get_history(self.session_id)
        
        # Verify metadata exists
        self.assertEqual(len(history), 3)
        
        for turn in history:
            self.assertIn('question', turn)
            self.assertIn('answer', turn)
            self.assertIn('timestamp', turn)
            self.assertIn('metadata', turn)
    
    def test_conversation_stats(self):
        """Test conversation statistics calculation."""
        # Create conversation
        for i in range(5):
            self.rag.query(
                f"Question {i+1} about machine learning",
                session_id=self.session_id,
                enable_rewrite=True
            )
        
        # Get stats
        stats = self.rag.conversation_manager.get_session_stats(self.session_id)
        
        self.assertEqual(stats['turn_count'], 5)
        self.assertIn('rewrite_rate', stats)
        self.assertIsInstance(stats['rewrite_rate'], float)
    
    def test_rewrite_tracking(self):
        """Test that query rewrites are properly tracked."""
        # First question (no rewrite expected)
        self.rag.query(
            "What is machine learning?",
            session_id=self.session_id,
            enable_rewrite=True
        )
        
        # Follow-up (should trigger rewrite)
        result = self.rag.query(
            "What about it?",
            session_id=self.session_id,
            enable_rewrite=True,
            return_context=True
        )
        
        # Get history
        history = self.rag.conversation_manager.get_history(self.session_id)
        
        # Second turn should have rewrite metadata
        second_turn = history[1]
        self.assertIn('metadata', second_turn)
        # Rewrite might happen depending on LLM
        if second_turn['metadata'].get('rewritten_query'):
            self.assertNotEqual(
                second_turn['metadata']['rewritten_query'],
                "What about it?"
            )


class TestRoutingExplanation(unittest.TestCase):
    """Test routing explanation and transparency features."""
    
    def setUp(self):
        """Initialize RAG with routing enabled."""
        self.rag = RAGEngine(
            enable_multi_collection=True,
            enable_query_routing=True,
            use_llm_routing=False  # Use heuristic for predictability
        )
    
    def test_routing_explanation_generation(self):
        """Test that routing explanations are generated."""
        query = "Find research papers about BERT"
        
        explanation = self.rag.query_router.explain_routing(query)
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        self.assertIn('research', explanation.lower())
    
    def test_routing_confidence_scores(self):
        """Test confidence score calculation for all collections."""
        query = "What Python skills do candidates have?"
        
        confidence = self.rag.query_router.get_intent_confidence(query)
        
        # Should return scores for all collections
        self.assertIsInstance(confidence, dict)
        self.assertGreater(len(confidence), 0)
        
        # Resumes should have highest confidence
        if 'resumes' in confidence:
            self.assertGreater(confidence['resumes'], 0.5)
        
        # All scores should be between 0 and 1
        for score in confidence.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_routing_metadata_in_results(self):
        """Test that routing metadata is included in query results."""
        result = self.rag.query(
            "Find papers about transformers",
            auto_route=True,
            return_context=True
        )
        
        # Should have metadata
        self.assertIn('metadata', result)
        metadata = result['metadata']
        
        # Should include routing information when auto-routing
        if self.rag.enable_query_routing:
            # Collections searched should be present
            self.assertIn('collections_searched', metadata)


class TestPerformanceTracking(unittest.TestCase):
    """Test query performance metrics."""
    
    def setUp(self):
        """Initialize RAG engine."""
        self.rag = RAGEngine()
    
    def test_query_timing(self):
        """Test that queries complete in reasonable time."""
        start = time.time()
        
        result = self.rag.query(
            "What is machine learning?",
            return_context=True
        )
        
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds (reasonable for LLM call)
        self.assertLess(elapsed, 5.0)
        
        # Should return valid result
        self.assertIn('answer', result)
        self.assertIn('context', result)
    
    def test_retrieval_quality_metrics(self):
        """Test that retrieval quality can be measured."""
        result = self.rag.query(
            "What is machine learning?",
            return_context=True
        )
        
        # Get context
        context = result['context']
        
        if context:
            # Calculate average similarity
            avg_similarity = sum(r['score'] for r in context) / len(context)
            
            # Average should be reasonable
            self.assertGreater(avg_similarity, 0.3)
            self.assertLess(avg_similarity, 1.0)
            
            # Should have multiple sources
            self.assertGreater(len(context), 0)


class TestExportImportFunctionality(unittest.TestCase):
    """Test data export and import capabilities."""
    
    def setUp(self):
        """Initialize RAG with conversation."""
        self.rag = RAGEngine(enable_conversation=True)
        self.session_id = "test_export_session"
    
    def test_export_single_result(self):
        """Test exporting a single query result."""
        result = self.rag.query(
            "What is machine learning?",
            return_context=True,
            session_id=self.session_id
        )
        
        # Create export data structure
        export_data = {
            "question": "What is machine learning?",
            "answer": result['answer'],
            "session_id": self.session_id,
            "sources": [
                {
                    "file": r['metadata'].get('filename', 'Unknown'),
                    "similarity": r['score'],
                    "text": r['text']
                }
                for r in result['context']
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Should be JSON serializable
        json_str = json.dumps(export_data, indent=2)
        self.assertIsInstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed['question'], "What is machine learning?")
        self.assertIn('answer', parsed)
        self.assertIn('sources', parsed)
    
    def test_export_full_conversation(self):
        """Test exporting entire conversation history."""
        # Create conversation
        questions = [
            "What is machine learning?",
            "What about deep learning?",
            "Show me examples"
        ]
        
        for q in questions:
            self.rag.query(
                q,
                session_id=self.session_id,
                enable_rewrite=True
            )
        
        # Get full history
        history = self.rag.conversation_manager.get_history(self.session_id)
        
        # Create export structure
        conversation_export = {
            "session_id": self.session_id,
            "total_turns": len(history),
            "conversation": history,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Should be JSON serializable
        json_str = json.dumps(conversation_export, indent=2)
        self.assertIsInstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        self.assertEqual(parsed['session_id'], self.session_id)
        self.assertEqual(parsed['total_turns'], 3)
        self.assertEqual(len(parsed['conversation']), 3)


class TestDemoScriptValidation(unittest.TestCase):
    """Test that demo script components work correctly."""
    
    def test_basic_rag_initialization(self):
        """Test basic RAG initialization for demo."""
        rag = RAGEngine(
            enable_multi_collection=False,
            enable_conversation=False,
            enable_personas=False,
            enable_query_routing=False
        )
        
        self.assertIsNotNone(rag)
        self.assertFalse(rag.enable_multi_collection)
        self.assertFalse(rag.enable_conversation)
        self.assertFalse(rag.enable_personas)
        self.assertFalse(rag.enable_query_routing)
    
    def test_multi_collection_demo(self):
        """Test multi-collection demo scenario."""
        rag = RAGEngine(enable_multi_collection=True)
        
        # Should have collections
        collections = list(rag.vector_store.collections.keys())
        self.assertGreater(len(collections), 0)
        
        # Should be able to query specific collection
        result = rag.query(
            "What is machine learning?",
            collections=['textbooks'],
            return_context=True
        )
        
        self.assertIn('answer', result)
    
    def test_conversation_demo(self):
        """Test conversational demo scenario."""
        rag = RAGEngine(enable_conversation=True)
        session_id = "demo_conv"
        
        # First question
        r1 = rag.query(
            "What is machine learning?",
            session_id=session_id,
            enable_rewrite=True
        )
        
        # Follow-up
        r2 = rag.query(
            "What about deep learning?",
            session_id=session_id,
            enable_rewrite=True
        )
        
        # Should have history
        history = rag.conversation_manager.get_history(session_id)
        self.assertEqual(len(history), 2)
    
    def test_persona_demo(self):
        """Test persona demo scenario."""
        rag = RAGEngine(enable_personas=True)
        
        question = "What is machine learning?"
        
        # Beginner
        r_beginner = rag.query(question, persona='beginner', return_context=True)
        
        # Expert
        r_expert = rag.query(question, persona='expert', return_context=True)
        
        # Both should work
        self.assertIn('answer', r_beginner)
        self.assertIn('answer', r_expert)
        
        # Different personas might retrieve different amounts
        # (not guaranteed, but personas have different top_k)
    
    def test_routing_demo(self):
        """Test routing demo scenario."""
        rag = RAGEngine(
            enable_multi_collection=True,
            enable_query_routing=True,
            use_llm_routing=False
        )
        
        # Test routing
        query = "Find research papers about BERT"
        
        routed = rag.query_router.route_query(query, max_collections=2)
        
        self.assertIsInstance(routed, list)
        self.assertGreater(len(routed), 0)
    
    def test_full_system_integration(self):
        """Test full system with all features enabled."""
        rag = RAGEngine(
            enable_multi_collection=True,
            enable_conversation=True,
            enable_personas=True,
            enable_query_routing=True,
            use_llm_routing=False
        )
        
        session_id = "full_integration"
        
        # Query with all features
        result = rag.query(
            question="What is machine learning?",
            session_id=session_id,
            enable_rewrite=True,
            persona="intermediate",
            auto_route=True,
            return_context=True
        )
        
        # Should return complete result
        self.assertIn('answer', result)
        self.assertIn('context', result)
        self.assertIn('metadata', result)
        
        # Should have conversation history
        history = rag.conversation_manager.get_history(session_id)
        self.assertEqual(len(history), 1)


class TestUIEnhancements(unittest.TestCase):
    """Test UI enhancement features (logic only, not Streamlit rendering)."""
    
    def test_metadata_enrichment(self):
        """Test that query results include UI-relevant metadata."""
        rag = RAGEngine(
            enable_personas=True,
            enable_query_routing=True,
            enable_multi_collection=True
        )
        
        result = rag.query(
            "What is machine learning?",
            persona="expert",
            auto_route=True,
            return_context=True
        )
        
        # Should include metadata
        self.assertIn('metadata', result)
        metadata = result['metadata']
        
        # Should include persona info
        self.assertIn('persona', metadata)
        self.assertEqual(metadata['persona'], 'expert')
    
    def test_source_metadata_completeness(self):
        """Test that sources include complete metadata for UI display."""
        rag = RAGEngine(enable_multi_collection=True)
        
        result = rag.query(
            "What is machine learning?",
            return_context=True
        )
        
        # Check context metadata
        for source in result['context']:
            self.assertIn('metadata', source)
            self.assertIn('score', source)
            self.assertIn('text', source)
            
            # Metadata should have filename
            self.assertIn('filename', source['metadata'])


if __name__ == '__main__':
    print("=" * 70)
    print("  Phase 6 Tests: Enhanced UI & Demo Features")
    print("=" * 70)
    print()
    
    # Run tests
    unittest.main(verbosity=2)
