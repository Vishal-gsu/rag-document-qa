"""
Phase 4 Tests: User Persona System
Tests persona profiles, adaptive behavior, and RAG integration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from user_persona import UserPersona, PersonaProfile
from rag_engine import RAGEngine
from config import Config


class TestPersonaProfiles:
    """Test persona profile management."""
    
    def setup_method(self):
        """Setup test environment."""
        self.persona_manager = UserPersona(default_persona='intermediate')
    
    def test_persona_initialization(self):
        """Test persona manager initialization."""
        assert self.persona_manager.current_persona == 'intermediate'
        assert len(UserPersona.PERSONAS) == 4  # beginner, intermediate, expert, researcher
        print("✓ Persona manager initialized")
    
    def test_get_profile(self):
        """Test getting persona profiles."""
        beginner = self.persona_manager.get_profile('beginner')
        expert = self.persona_manager.get_profile('expert')
        
        assert beginner.name == 'Beginner'
        assert expert.name == 'Expert'
        
        # Verify different configurations
        assert beginner.top_k < expert.top_k  # Beginner gets fewer sources
        assert beginner.temperature > expert.temperature  # Beginner gets more creative
        assert beginner.max_tokens > expert.max_tokens  # Beginner gets longer explanations
        
        print("✓ Profile retrieval works")
    
    def test_all_persona_profiles(self):
        """Test all predefined personas have valid configurations."""
        for persona_name in ['beginner', 'intermediate', 'expert', 'researcher']:
            profile = self.persona_manager.get_profile(persona_name)
            
            assert profile.name is not None
            assert profile.description is not None
            assert profile.top_k > 0
            assert 0.0 <= profile.temperature <= 1.0
            assert profile.max_tokens > 0
            assert len(profile.system_prompt) > 0
            assert profile.retrieval_strategy in ['broad', 'balanced', 'precise', 'comprehensive']
        
        print("✓ All persona profiles valid")
    
    def test_set_persona(self):
        """Test switching personas."""
        success = self.persona_manager.set_persona('expert')
        assert success
        assert self.persona_manager.current_persona == 'expert'
        
        # Try invalid persona
        success = self.persona_manager.set_persona('invalid_persona')
        assert not success
        assert self.persona_manager.current_persona == 'expert'  # Unchanged
        
        print("✓ Persona switching works")
    
    def test_list_personas(self):
        """Test listing available personas."""
        personas = self.persona_manager.list_personas()
        
        assert 'beginner' in personas
        assert 'intermediate' in personas
        assert 'expert' in personas
        assert 'researcher' in personas
        
        # Check descriptions exist
        for name, desc in personas.items():
            assert len(desc) > 0
        
        print("✓ Persona listing works")
    
    def test_retrieval_params(self):
        """Test getting retrieval parameters."""
        beginner_params = self.persona_manager.get_retrieval_params('beginner')
        expert_params = self.persona_manager.get_retrieval_params('expert')
        
        assert 'top_k' in beginner_params
        assert 'retrieval_strategy' in beginner_params
        
        # Expert should retrieve more
        assert expert_params['top_k'] > beginner_params['top_k']
        
        print("✓ Retrieval parameters work")
    
    def test_generation_params(self):
        """Test getting generation parameters."""
        params = self.persona_manager.get_generation_params('intermediate')
        
        assert 'temperature' in params
        assert 'max_tokens' in params
        assert 'system_prompt' in params
        
        assert len(params['system_prompt']) > 0
        
        print("✓ Generation parameters work")
    
    def test_custom_persona(self):
        """Test adding custom personas."""
        self.persona_manager.add_custom_persona(
            name='custom_test',
            description='Test persona',
            top_k=6,
            temperature=0.4,
            max_tokens=450,
            system_prompt='Custom prompt',
            retrieval_strategy='balanced'
        )
        
        profile = self.persona_manager.get_profile('custom_test')
        assert profile.name == 'custom_test'
        assert profile.top_k == 6
        assert profile.system_prompt == 'Custom prompt'
        
        print("✓ Custom persona creation works")


class TestAdaptivePersona:
    """Test adaptive persona behavior."""
    
    def setup_method(self):
        """Setup test environment."""
        self.persona_manager = UserPersona()
    
    def test_adapt_to_simple_query(self):
        """Test adaptation to simple beginner-style query."""
        query = "What is machine learning?"
        
        adapted = self.persona_manager.adapt_to_query_complexity(query, 'intermediate')
        
        # Should reduce top_k for simple queries
        base_profile = self.persona_manager.get_profile('intermediate')
        assert adapted.top_k <= base_profile.top_k
        
        print("✓ Simple query adaptation works")
    
    def test_adapt_to_complex_query(self):
        """Test adaptation to complex technical query."""
        query = "Explain the implementation details of the transformer architecture's multi-head attention mechanism and its computational complexity"
        
        adapted = self.persona_manager.adapt_to_query_complexity(query, 'intermediate')
        
        # Should increase top_k for complex queries
        base_profile = self.persona_manager.get_profile('intermediate')
        assert adapted.top_k >= base_profile.top_k
        
        print("✓ Complex query adaptation works")
    
    def test_adapt_preserves_other_params(self):
        """Test that adaptation only changes top_k, not other params."""
        query = "What are neural networks?"
        base_profile = self.persona_manager.get_profile('expert')
        
        adapted = self.persona_manager.adapt_to_query_complexity(query, 'expert')
        
        # Other params should match
        assert adapted.temperature == base_profile.temperature
        assert adapted.max_tokens == base_profile.max_tokens
        assert adapted.system_prompt == base_profile.system_prompt
        
        print("✓ Adaptation preserves other parameters")


class TestRAGWithPersonas:
    """Test RAG engine integration with personas."""
    
    def setup_method(self):
        """Setup test environment."""
        self.rag = RAGEngine(enable_personas=True, default_persona='intermediate')
    
    def test_rag_persona_initialization(self):
        """Test RAG initializes with persona system."""
        assert self.rag.enable_personas
        assert self.rag.persona_manager is not None
        assert self.rag.persona_manager.current_persona == 'intermediate'
        
        print("✓ RAG with personas initialized")
    
    def test_rag_persona_disabled(self):
        """Test RAG can run without persona system."""
        rag_no_persona = RAGEngine(enable_personas=False)
        
        assert not rag_no_persona.enable_personas
        assert rag_no_persona.persona_manager is None
        
        print("✓ RAG without personas works")
    
    def test_persona_affects_retrieval(self):
        """Test that different personas use different top_k values."""
        # Get profiles
        beginner_profile = self.rag.persona_manager.get_profile('beginner')
        expert_profile = self.rag.persona_manager.get_profile('expert')
        
        # Verify different settings
        assert beginner_profile.top_k != expert_profile.top_k
        assert beginner_profile.temperature != expert_profile.temperature
        
        print("✓ Personas have different retrieval settings")
    
    def test_generation_params_differ(self):
        """Test generation parameters differ by persona."""
        beginner = self.rag.persona_manager.get_generation_params('beginner')
        expert = self.rag.persona_manager.get_generation_params('expert')
        
        # System prompts should be different
        assert beginner['system_prompt'] != expert['system_prompt']
        
        # beginner should be more creative
        assert beginner['temperature'] > expert['temperature']
        
        print("✓ Generation parameters differ by persona")


def run_all_tests():
    """Run all Phase 4 tests."""
    print("\n" + "="*60)
    print("PHASE 4 TESTS: User Persona System")
    print("="*60 + "\n")
    
    # Test 1: Persona Profiles
    print("Test Suite 1: Persona Profile Management")
    print("-" * 60)
    test_profiles = TestPersonaProfiles()
    test_profiles.setup_method()
    test_profiles.test_persona_initialization()
    test_profiles.test_get_profile()
    test_profiles.test_all_persona_profiles()
    test_profiles.test_set_persona()
    test_profiles.test_list_personas()
    test_profiles.test_retrieval_params()
    test_profiles.test_generation_params()
    test_profiles.test_custom_persona()
    
    # Test 2: Adaptive Persona
    print("\nTest Suite 2: Adaptive Persona Behavior")
    print("-" * 60)
    test_adaptive = TestAdaptivePersona()
    test_adaptive.setup_method()
    test_adaptive.test_adapt_to_simple_query()
    test_adaptive.test_adapt_to_complex_query()
    test_adaptive.test_adapt_preserves_other_params()
    
    # Test 3: RAG Integration
    print("\nTest Suite 3: RAG Integration with Personas")
    print("-" * 60)
    test_rag = TestRAGWithPersonas()
    test_rag.setup_method()
    test_rag.test_rag_persona_initialization()
    test_rag.test_rag_persona_disabled()
    test_rag.test_persona_affects_retrieval()
    test_rag.test_generation_params_differ()
    
    print("\n" + "="*60)
    print("✅ ALL PHASE 4 TESTS PASSED!")
    print("="*60 + "\n")
    print("Phase 4 Implementation Summary:")
    print("  ✓ UserPersona: 4 predefined personas + custom persona support")
    print("  ✓ Adaptive behavior: Query complexity analysis")
    print("  ✓ RAGEngine: Persona-aware retrieval and generation")
    print("  ✓ Streamlit UI: Persona selector with auto-adapt toggle")
    print("\nPersona Profiles:")
    print("  🌱 Beginner: top_k=3, temp=0.7, detailed explanations")
    print("  📚 Intermediate: top_k=5, temp=0.5, balanced approach")
    print("  🎯 Expert: top_k=7, temp=0.3, technical depth")
    print("  🔬 Researcher: top_k=10, temp=0.2, scholarly focus")
    print("\nNext Steps:")
    print("  - Test with real queries and different personas")
    print("  - Verify LLM responses match persona expectations")
    print("  - Move to Phase 5: Query Routing Intelligence")


if __name__ == "__main__":
    run_all_tests()
