"""
Phase 1 Verification Test Script
Tests multi-collection infrastructure and document classification.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    from document_classifier import DocumentClassifier
    from vector_store import VectorStore
    from config import Config
    from rag_engine import RAGEngine
    print("✓ All imports successful\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_document_classifier():
    """Test document classification with sample texts."""
    print("="*70)
    print("TEST 1: Document Classifier")
    print("="*70)
    
    classifier = DocumentClassifier()
    
    # Test research paper
    paper_text = """
    Abstract: This paper presents a novel approach to machine learning.
    1. Introduction
    Recent work by Smith et al. (2020) has shown promising results.
    2. Methodology
    We conducted experiments on the MNIST dataset.
    3. Results
    Figure 1 shows the accuracy improvements.
    4. Conclusion
    Our method achieves state-of-the-art performance.
    References:
    [1] Smith, J. et al. (2020). Deep Learning Methods. IEEE.
    """
    
    result = classifier.classify(content=paper_text)
    print(f"\nResearch Paper Classification: {result}")
    assert result == 'research_paper', f"Expected 'research_paper', got '{result}'"
    print("✓ Research paper correctly identified")
    
    # Test resume
    resume_text = """
    John Doe
    Email: john@example.com | Phone: 123-456-7890
    
    Education
    B.S. Computer Science, MIT, 2020-2024
    
    Skills
    - Python, Java, C++
    - Machine Learning, Deep Learning
    - TensorFlow, PyTorch
    
    Experience
    Software Engineer, Google, 2024-Present
    - Developed ML models for search ranking
    """
    
    result = classifier.classify(content=resume_text)
    print(f"\nResume Classification: {result}")
    assert result == 'resume', f"Expected 'resume', got '{result}'"
    print("✓ Resume correctly identified")
    
    # Test textbook
    textbook_text = """
    Chapter 7: Neural Networks
    
    Learning Objectives:
    - Understand backpropagation algorithm
    - Implement gradient descent
    
    In this chapter, we will explore artificial neural networks.
    
    Key Terms:
    - Activation function
    - Weight initialization
    
    Exercises:
    7.1 Implement a simple perceptron
    7.2 Train a network on XOR problem
    
    Summary:
    Neural networks are powerful function approximators.
    """
    
    result = classifier.classify(content=textbook_text)
    print(f"\nTextbook Classification: {result}")
    assert result == 'textbook', f"Expected 'textbook', got '{result}'"
    print("✓ Textbook correctly identified")
    
    # Test generic
    generic_text = """
    This is a simple note about meeting tomorrow.
    We should discuss the project timeline.
    """
    
    result = classifier.classify(content=generic_text)
    print(f"\nGeneric Document Classification: {result}")
    assert result == 'generic', f"Expected 'generic', got '{result}'"
    print("✓ Generic document correctly identified")
    
    print("\n✓ All classifier tests passed!\n")


def test_multi_collection_vector_store():
    """Test multi-collection vector store operations."""
    print("="*70)
    print("TEST 2: Multi-Collection Vector Store")
    print("="*70)
    
    # Create test store with multi-collection enabled
    store = VectorStore(
        db_path="./data/test_vectordb",
        enable_multi_collection=True
    )
    
    print("\n1. Testing collection creation...")
    
    # Create multiple collections
    store.get_or_create_collection("research_papers", dimension=1024)
    store.get_or_create_collection("resumes", dimension=1024)
    store.get_or_create_collection("textbooks", dimension=1024)
    
    print("✓ Created 3 collections")
    
    # List collections
    collections = store.list_collections()
    print(f"\n2. Collections created: {len(collections)}")
    for coll in collections:
        print(f"  - {coll['name']}: {coll['vector_count']} vectors, {coll['dimension']}D")
    
    assert len(collections) >= 3, "Should have at least 3 collections"
    print("✓ Collection listing works")
    
    # Test adding vectors to specific collection
    print("\n3. Testing vector addition to specific collection...")
    import numpy as np
    
    test_vectors = np.random.rand(5, 1024).tolist()
    test_metadata = [
        {'text': f'Research paper chunk {i}', 'filename': 'test_paper.pdf'}
        for i in range(5)
    ]
    
    store.add_vectors_to_collection(
        collection_name="research_papers",
        vectors=test_vectors,
        metadata=test_metadata
    )
    
    stats = store.get_collection_stats("research_papers")
    print(f"✓ Added {stats['total_vectors']} vectors to research_papers collection")
    
    # Test multi-collection search
    print("\n4. Testing multi-collection search...")
    query_vector = np.random.rand(1024).tolist()
    
    results = store.search_multi_collection(
        query_vector=query_vector,
        collections=["research_papers"],
        top_k=3
    )
    
    print(f"✓ Search returned {len(results)} results")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: score={result['score']:.4f}, collection={result['collection']}")
    
    print("\n✓ All multi-collection tests passed!\n")
    
    # Cleanup
    print("Cleaning up test data...")
    import shutil
    test_path = Path("./data/test_vectordb")
    if test_path.exists():
        shutil.rmtree(test_path)
    print("✓ Cleanup complete\n")


def test_config_updates():
    """Test configuration updates."""
    print("="*70)
    print("TEST 3: Configuration")
    print("="*70)
    
    print(f"\nMulti-collection enabled: {Config.ENABLE_MULTI_COLLECTION}")
    print(f"Auto-classification enabled: {Config.ENABLE_AUTO_CLASSIFICATION}")
    print(f"Chunk size: {Config.CHUNK_SIZE}")
    print(f"Chunk overlap: {Config.CHUNK_OVERLAP}")
    print(f"Top-K results: {Config.TOP_K_RESULTS}")
    
    print("\nCollection name mapping:")
    for doc_type, coll_name in Config.COLLECTION_NAMES.items():
        print(f"  {doc_type} → {coll_name}")
    
    # Test get_collection_name
    assert Config.get_collection_name('research_paper') in ['research_papers', 'document_embeddings']
    assert Config.get_collection_name('resume') in ['resumes', 'document_embeddings']
    assert Config.get_collection_name('generic') in ['general_docs', 'document_embeddings']
    
    print("\n✓ Configuration tests passed!\n")


def test_rag_engine_initialization():
    """Test RAG engine with multi-collection support."""
    print("="*70)
    print("TEST 4: RAG Engine Initialization")
    print("="*70)
    
    print("\n1. Testing legacy mode (single collection)...")
    rag_legacy = RAGEngine(enable_multi_collection=False)
    assert not rag_legacy.enable_multi_collection
    assert not rag_legacy.vector_store.enable_multi_collection
    print("✓ Legacy mode initialized correctly")
    
    print("\n2. Testing multi-collection mode...")
    rag_multi = RAGEngine(enable_multi_collection=True)
    assert rag_multi.enable_multi_collection
    assert rag_multi.vector_store.enable_multi_collection
    assert rag_multi.classifier is not None
    print("✓ Multi-collection mode initialized correctly")
    
    print("\n✓ RAG engine tests passed!\n")


def run_all_tests():
    """Run all Phase 1 tests."""
    print("\n" + "="*70)
    print("PHASE 1 VERIFICATION TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        ("Document Classifier", test_document_classifier),
        ("Multi-Collection Vector Store", test_multi_collection_vector_store),
        ("Configuration Updates", test_config_updates),
        ("RAG Engine Initialization", test_rag_engine_initialization),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ Test '{test_name}' failed: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test '{test_name}' error: {e}\n")
            failed += 1
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(tests)}")
    else:
        print("\n🎉 ALL TESTS PASSED! Phase 1 implementation is complete.\n")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
