"""
Phase 2 Verification Test Script
Tests specialized document parsers for different document types.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    from parsers import BaseParser, GenericParser, ResearchPaperParser, ResumeParser, TextbookParser
    from document_processor import DocumentProcessor
    from document_classifier import DocumentClassifier
    print("✓ All imports successful\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_research_paper_parser():
    """Test research paper parser with section detection."""
    print("="*70)
    print("TEST 1: Research Paper Parser")
    print("="*70)
    
    paper_text = """
A Novel Approach to Machine Learning

Abstract
This paper presents a novel approach to machine learning using transformers.

1. Introduction
Machine learning has revolutionized many fields. Recent work by Smith et al. (2020) 
has shown promising results in natural language processing.

2. Related Work
Previous studies have explored various architectures for sequence modeling.

3. Methodology
We propose a new architecture based on self-attention mechanisms. Our approach
uses the following components:
- Multi-head attention
- Layer normalization
- Residual connections

4. Results
Figure 1 shows our experimental results. On the MNIST dataset, we achieved 
98.5% accuracy, outperforming previous methods by 2%.

5. Discussion
The results demonstrate the effectiveness of our approach.

6. Conclusion
We have presented a novel method for machine learning that achieves state-of-the-art
performance on benchmark datasets.

References
[1] Smith, J. et al. (2020). Deep Learning Methods. IEEE Transactions.
    """
    
    parser = ResearchPaperParser(chunk_size=500, chunk_overlap=50)
    metadata = {'filename': 'paper.pdf', 'source': 'test.pdf', 'type': '.pdf', 'doc_type': 'research_paper'}
    
    chunks = parser.parse(paper_text, metadata)
    
    print(f"\nParsed {len(chunks)} chunks")
    
    # Check section detection
    sections_found = set()
    for chunk in chunks:
        section = chunk['metadata'].get('section', 'unknown')
        sections_found.add(section)
    
    print(f"Sections detected: {sorted(sections_found)}")
    
    # Verify key sections present
    expected_sections = {'abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references'}
    found_expected = expected_sections.intersection(sections_found)
    print(f"Expected sections found: {found_expected}")
    
    assert len(found_expected) >= 4, f"Should detect at least 4 sections, found {len(found_expected)}"
    
    # Check metadata enrichment
    sample_chunk = chunks[0]
    print(f"\nSample chunk metadata:")
    for key, value in sample_chunk['metadata'].items():
        if key != 'text':
            print(f"  {key}: {value}")
    
    assert 'section' in sample_chunk['metadata'], "Chunks should have 'section' metadata"
    print("\n✓ Research paper parser tests passed!")


def test_resume_parser():
    """Test resume parser with skill extraction."""
    print("\n" + "="*70)
    print("TEST 2: Resume Parser")
    print("="*70)
    
    resume_text = """
John Doe
Email: john.doe@example.com | Phone: 555-123-4567

Professional Summary
Experienced software engineer with expertise in machine learning and cloud computing.

Education
B.S. Computer Science
Massachusetts Institute of Technology, 2020-2024

Experience
Senior Software Engineer
Google, 2024-Present
- Developed machine learning models using Python and TensorFlow
- Built RESTful APIs with FastAPI
- Deployed services on AWS and Kubernetes

Skills
Programming Languages: Python, Java, JavaScript, C++
Frameworks: TensorFlow, PyTorch, React, Django
Cloud: AWS, Azure, Docker, Kubernetes
AI/ML: Machine Learning, Deep Learning, NLP, RAG

Certifications
- AWS Certified Solutions Architect
- Google Cloud Professional ML Engineer
    """
    
    parser = ResumeParser(chunk_size=500, chunk_overlap=50)
    metadata = {'filename': 'resume.pdf', 'source': 'test.pdf', 'type': '.pdf', 'doc_type': 'resume'}
    
    chunks = parser.parse(resume_text, metadata)
    
    print(f"\nParsed {len(chunks)} chunks")
    
    # Check section detection
    sections_found = set()
    for chunk in chunks:
        section = chunk['metadata'].get('section', 'unknown')
        sections_found.add(section)
    
    print(f"Sections detected: {sorted(sections_found)}")
    
    # Check skill extraction
    sample_chunk = chunks[0]
    skills = sample_chunk['metadata'].get('skills', [])
    print(f"\nSkills extracted: {len(skills)} skills")
    print(f"Sample skills: {skills[:10]}")
    
    # Verify expected skills detected
    expected_skills = {'python', 'machine learning', 'tensorflow', 'aws', 'docker'}
    found_skills = set(skill.lower() for skill in skills)
    found_expected = expected_skills.intersection(found_skills)
    
    print(f"Expected skills found: {found_expected}")
    assert len(found_expected) >= 3, f"Should detect at least 3 expected skills, found {len(found_expected)}"
    
    # Check contact info extraction
    print(f"\nContact info extracted:")
    if 'email' in sample_chunk['metadata']:
        print(f"  Email: {sample_chunk['metadata']['email']}")
    if 'phone' in sample_chunk['metadata']:
        print(f"  Phone: {sample_chunk['metadata']['phone']}")
    if 'name' in sample_chunk['metadata']:
        print(f"  Name: {sample_chunk['metadata']['name']}")
    
    print("\n✓ Resume parser tests passed!")


def test_textbook_parser():
    """Test textbook parser with chapter detection."""
    print("\n" + "="*70)
    print("TEST 3: Textbook Parser")
    print("="*70)
    
    textbook_text = """
Chapter 7: Neural Networks

Learning Objectives:
- Understand the architecture of neural networks
- Learn the backpropagation algorithm
- Implement gradient descent optimization

7.1 Introduction to Neural Networks

Artificial neural networks are computational models inspired by biological neurons.
They consist of interconnected layers of processing units.

7.2 Architecture

A typical neural network consists of:
- Input layer
- Hidden layers
- Output layer

7.3 Training Process

The training process involves forward propagation and backpropagation.

Summary
In this chapter, we covered the fundamentals of neural networks including
architecture, training, and applications.

Exercises
7.1 Implement a simple perceptron in Python
7.2 Train a network to solve the XOR problem
7.3 Experiment with different activation functions

Key Terms
- Activation function
- Backpropagation
- Gradient descent
- Weight initialization
    """
    
    parser = TextbookParser(chunk_size=500, chunk_overlap=50)
    metadata = {'filename': 'textbook.pdf', 'source': 'test.pdf', 'type': '.pdf', 'doc_type': 'textbook'}
    
    chunks = parser.parse(textbook_text, metadata)
    
    print(f"\nParsed {len(chunks)} chunks")
    
    # Check chapter detection
    chapters_found = set()
    sections_found = set()
    for chunk in chunks:
        chapter = chunk['metadata'].get('chapter')
        section = chunk['metadata'].get('section')
        if chapter:
            chapters_found.add(chapter)
        if section:
            sections_found.add(section)
    
    print(f"Chapters detected: {sorted(chapters_found)}")
    print(f"Sections detected: {sorted(sections_found)}")
    
    # Check special sections
    section_types = set()
    for chunk in chunks:
        section_type = chunk['metadata'].get('section_type')
        if section_type:
            section_types.add(section_type)
    
    print(f"Section types detected: {sorted(section_types)}")
    
    # Verify special sections
    expected_types = {'learning_objectives', 'summary', 'exercises', 'key_terms'}
    found_expected = expected_types.intersection(section_types)
    print(f"Special sections found: {found_expected}")
    
    assert len(found_expected) >= 2, f"Should detect at least 2 special sections, found {len(found_expected)}"
    
    # Check sample chunk
    sample_chunk = chunks[0]
    print(f"\nSample chunk metadata:")
    for key, value in sample_chunk['metadata'].items():
        if key != 'text':
            print(f"  {key}: {value}")
    
    print("\n✓ Textbook parser tests passed!")


def test_generic_parser():
    """Test generic parser fallback."""
    print("\n" + "="*70)
    print("TEST 4: Generic Parser")
    print("="*70)
    
    generic_text = """
This is a simple document without any specific structure.
It could be meeting notes, a blog post, or any general text.

The parser should handle this gracefully by creating simple chunks
without trying to detect any document-specific structure.
    """
    
    parser = GenericParser(chunk_size=200, chunk_overlap=50)
    metadata = {'filename': 'notes.txt', 'source': 'test.txt', 'type': '.txt', 'doc_type': 'generic'}
    
    chunks = parser.parse(generic_text, metadata)
    
    print(f"\nParsed {len(chunks)} chunks")
    
    assert len(chunks) > 0, "Should create at least one chunk"
    
    # Verify basic metadata
    sample_chunk = chunks[0]
    assert 'text' in sample_chunk, "Chunk should have text"
    assert 'metadata' in sample_chunk, "Chunk should have metadata"
    assert sample_chunk['metadata']['chunk_id'] == 0, "First chunk should have chunk_id 0"
    
    print(f"Chunk 0 length: {len(sample_chunk['text'])} chars")
    print("\n✓ Generic parser tests passed!")


def test_integrated_document_processor():
    """Test DocumentProcessor with classifier and parser integration."""
    print("\n" + "="*70)
    print("TEST 5: Integrated Document Processor")
    print("="*70)
    
    classifier = DocumentClassifier()
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=50,
        classifier=classifier,
        enable_specialized_parsing=True
    )
    
    # Create test documents
    paper_text = """
Abstract: Novel machine learning approach.
Introduction: Recent work shows...
Methodology: We propose...
Results: Achieved 98% accuracy.
Conclusion: Effective method demonstrated.
References: [1] Smith et al. 2020.
    """
    
    resume_text = """
John Doe
Email: john@example.com

Education
B.S. Computer Science, MIT

Skills
Python, TensorFlow, AWS

Experience
Engineer at Google
    """
    
    documents = [
        {
            'content': paper_text,
            'metadata': {
                'filename': 'paper.pdf',
                'source': 'test1.pdf',
                'type': '.pdf',
                'doc_type': 'research_paper'
            }
        },
        {
            'content': resume_text,
            'metadata': {
                'filename': 'resume.pdf',
                'source': 'test2.pdf',
                'type': '.pdf',
                'doc_type': 'resume'
            }
        }
    ]
    
    chunks = processor.chunk_documents(documents)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Verify different parsers were used
    doc_types = set()
    for chunk in chunks:
        doc_type = chunk['metadata'].get('doc_type')
        doc_types.add(doc_type)
    
    print(f"Document types processed: {doc_types}")
    assert 'research_paper' in doc_types, "Should have research paper chunks"
    assert 'resume' in doc_types, "Should have resume chunks"
    
    # Check for parser-specific metadata
    paper_chunks = [c for c in chunks if c['metadata'].get('doc_type') == 'research_paper']
    resume_chunks = [c for c in chunks if c['metadata'].get('doc_type') == 'resume']
    
    print(f"Paper chunks: {len(paper_chunks)}")
    print(f"Resume chunks: {len(resume_chunks)}")
    
    if paper_chunks:
        print(f"Paper chunk has section: {'section' in paper_chunks[0]['metadata']}")
    if resume_chunks:
        print(f"Resume chunk has skills: {'skills' in resume_chunks[0]['metadata']}")
    
    print("\n✓ Integrated processor tests passed!")


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n" + "="*70)
    print("PHASE 2 VERIFICATION TEST SUITE")
    print("Specialized Document Parsers")
    print("="*70 + "\n")
    
    tests = [
        ("Research Paper Parser", test_research_paper_parser),
        ("Resume Parser", test_resume_parser),
        ("Textbook Parser", test_textbook_parser),
        ("Generic Parser", test_generic_parser),
        ("Integrated Document Processor", test_integrated_document_processor),
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(tests)}")
    else:
        print("\n🎉 ALL TESTS PASSED! Phase 2 implementation is complete.\n")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
