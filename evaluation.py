"""
Evaluation Module for RAG System
Provides metrics, benchmarking, and visualization capabilities.
"""
import time
import json
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime


class RAGEvaluator:
    """Evaluate RAG system performance with metrics and visualizations."""
    
    def __init__(self, rag_engine):
        """
        Initialize evaluator.
        
        Args:
            rag_engine: RAGEngine instance to evaluate
        """
        self.rag_engine = rag_engine
        self.results = []
    
    def create_test_set(self) -> List[Dict]:
        """
        Create evaluation test set with questions and expected answers.
        
        Returns:
            List of test cases with questions and ground truth
        """
        test_cases = [
            {
                "question": "What is machine learning?",
                "expected_keywords": ["learning", "data", "algorithm", "patterns"],
                "category": "definition"
            },
            {
                "question": "What are the types of machine learning?",
                "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
                "category": "classification"
            },
            {
                "question": "Explain neural networks",
                "expected_keywords": ["neurons", "layers", "weights", "network"],
                "category": "explanation"
            },
            {
                "question": "What is supervised learning?",
                "expected_keywords": ["labeled", "training", "input", "output"],
                "category": "definition"
            },
            {
                "question": "How do vector embeddings work?",
                "expected_keywords": ["vector", "semantic", "similarity", "space"],
                "category": "explanation"
            },
            {
                "question": "What is the purpose of vector databases?",
                "expected_keywords": ["similarity", "search", "embeddings", "efficient"],
                "category": "purpose"
            },
            {
                "question": "What is RAG?",
                "expected_keywords": ["retrieval", "augmented", "generation", "context"],
                "category": "definition"
            },
            {
                "question": "How does semantic search differ from keyword search?",
                "expected_keywords": ["meaning", "semantic", "context", "similarity"],
                "category": "comparison"
            },
            {
                "question": "What are transformers in NLP?",
                "expected_keywords": ["attention", "architecture", "self-attention", "model"],
                "category": "explanation"
            },
            {
                "question": "What is the difference between overfitting and underfitting?",
                "expected_keywords": ["overfitting", "underfitting", "generalization", "training"],
                "category": "comparison"
            }
        ]
        
        return test_cases
    
    def evaluate_retrieval_quality(self, question: str, expected_keywords: List[str]) -> Dict:
        """
        Evaluate retrieval quality.
        
        Args:
            question: Query question
            expected_keywords: Keywords expected in retrieved context
            
        Returns:
            Retrieval metrics
        """
        # Get query embedding
        query_embedding = self.rag_engine.embedding_engine.embed_text(question)
        
        # Retrieve top-K results
        results = self.rag_engine.vector_store.search(
            query_vector=query_embedding,
            top_k=5
        )
        
        # Calculate metrics
        retrieved_text = " ".join([r['text'].lower() for r in results])
        
        # Keyword coverage
        keywords_found = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text)
        keyword_coverage = keywords_found / len(expected_keywords) if expected_keywords else 0
        
        # Average similarity score
        avg_similarity = np.mean([r['score'] for r in results]) if results else 0
        
        # Top-1 similarity
        top1_similarity = results[0]['score'] if results else 0
        
        return {
            'keyword_coverage': keyword_coverage,
            'avg_similarity_score': avg_similarity,
            'top1_similarity_score': top1_similarity,
            'num_results': len(results),
            'retrieved_texts': [r['text'][:100] + "..." for r in results[:3]]
        }
    
    def evaluate_answer_quality(self, question: str, answer: str, expected_keywords: List[str]) -> Dict:
        """
        Evaluate generated answer quality.
        
        Args:
            question: Query question
            answer: Generated answer
            expected_keywords: Keywords expected in answer
            
        Returns:
            Answer quality metrics
        """
        answer_lower = answer.lower()
        
        # Keyword presence in answer
        keywords_in_answer = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_score = keywords_in_answer / len(expected_keywords) if expected_keywords else 0
        
        # Answer length (reasonable range)
        answer_length = len(answer.split())
        length_score = 1.0 if 20 <= answer_length <= 200 else 0.5
        
        # Contains source attribution
        has_attribution = any(marker in answer_lower for marker in ['source', 'according to', 'from'])
        
        return {
            'keyword_score': keyword_score,
            'answer_length': answer_length,
            'length_score': length_score,
            'has_attribution': has_attribution,
            'answer_preview': answer[:200] + "..." if len(answer) > 200 else answer
        }
    
    def measure_latency(self, question: str) -> Dict:
        """
        Measure query latency breakdown.
        
        Args:
            question: Query question
            
        Returns:
            Latency metrics
        """
        timings = {}
        
        # Embedding time
        start = time.time()
        query_embedding = self.rag_engine.embedding_engine.embed_text(question)
        timings['embedding_ms'] = (time.time() - start) * 1000
        
        # Retrieval time
        start = time.time()
        results = self.rag_engine.vector_store.search(query_embedding, top_k=3)
        timings['retrieval_ms'] = (time.time() - start) * 1000
        
        # Generation time
        start = time.time()
        context = self.rag_engine._build_context(results)
        answer = self.rag_engine._generate_answer(question, context)
        timings['generation_ms'] = (time.time() - start) * 1000
        
        # Total time
        timings['total_ms'] = sum(timings.values())
        
        return timings
    
    def run_evaluation(self, save_results: bool = True) -> Dict:
        """
        Run complete evaluation on test set.
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Aggregated evaluation results
        """
        print("\n" + "="*70)
        print("🔬 RUNNING RAG SYSTEM EVALUATION")
        print("="*70 + "\n")
        
        test_cases = self.create_test_set()
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_case['question'][:50]}...")
            
            question = test_case['question']
            expected_keywords = test_case['expected_keywords']
            
            # Measure latency
            latency = self.measure_latency(question)
            
            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval_quality(question, expected_keywords)
            
            # Get answer
            answer = self.rag_engine.query(question, top_k=3)
            
            # Evaluate answer
            answer_metrics = self.evaluate_answer_quality(question, answer, expected_keywords)
            
            # Combine results
            result = {
                'question': question,
                'category': test_case['category'],
                'latency': latency,
                'retrieval': retrieval_metrics,
                'answer': answer_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            print(f"  ✓ Completed\n")
        
        # Aggregate metrics
        aggregated = self._aggregate_results(results)
        
        # Save results
        if save_results:
            self._save_results(results, aggregated)
        
        # Print summary
        self._print_summary(aggregated)
        
        self.results = results
        return aggregated
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate individual results into summary statistics."""
        
        # Latency statistics
        total_times = [r['latency']['total_ms'] for r in results]
        embedding_times = [r['latency']['embedding_ms'] for r in results]
        retrieval_times = [r['latency']['retrieval_ms'] for r in results]
        generation_times = [r['latency']['generation_ms'] for r in results]
        
        # Retrieval statistics
        keyword_coverages = [r['retrieval']['keyword_coverage'] for r in results]
        avg_similarities = [r['retrieval']['avg_similarity_score'] for r in results]
        top1_similarities = [r['retrieval']['top1_similarity_score'] for r in results]
        
        # Answer statistics
        keyword_scores = [r['answer']['keyword_score'] for r in results]
        answer_lengths = [r['answer']['answer_length'] for r in results]
        attribution_count = sum(1 for r in results if r['answer']['has_attribution'])
        
        return {
            'latency': {
                'avg_total_ms': np.mean(total_times),
                'p50_total_ms': np.percentile(total_times, 50),
                'p95_total_ms': np.percentile(total_times, 95),
                'avg_embedding_ms': np.mean(embedding_times),
                'avg_retrieval_ms': np.mean(retrieval_times),
                'avg_generation_ms': np.mean(generation_times),
            },
            'retrieval': {
                'avg_keyword_coverage': np.mean(keyword_coverages),
                'avg_similarity_score': np.mean(avg_similarities),
                'avg_top1_similarity': np.mean(top1_similarities),
                'keyword_coverage_std': np.std(keyword_coverages),
            },
            'answer': {
                'avg_keyword_score': np.mean(keyword_scores),
                'avg_answer_length': np.mean(answer_lengths),
                'attribution_rate': attribution_count / len(results),
                'keyword_score_std': np.std(keyword_scores),
            },
            'overall': {
                'total_tests': len(results),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _save_results(self, results: List[Dict], aggregated: Dict):
        """Save results to JSON file."""
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(output_dir / f"detailed_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save aggregated results
        with open(output_dir / f"summary_{timestamp}.json", 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"\n💾 Results saved to evaluation_results/")
    
    def _print_summary(self, aggregated: Dict):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70 + "\n")
        
        print("⏱️  LATENCY METRICS:")
        print(f"  Average Total:      {aggregated['latency']['avg_total_ms']:.0f} ms")
        print(f"  P50 (Median):       {aggregated['latency']['p50_total_ms']:.0f} ms")
        print(f"  P95:                {aggregated['latency']['p95_total_ms']:.0f} ms")
        print(f"  Breakdown:")
        print(f"    - Embedding:      {aggregated['latency']['avg_embedding_ms']:.0f} ms")
        print(f"    - Retrieval:      {aggregated['latency']['avg_retrieval_ms']:.0f} ms")
        print(f"    - Generation:     {aggregated['latency']['avg_generation_ms']:.0f} ms")
        
        print(f"\n🔍 RETRIEVAL QUALITY:")
        print(f"  Keyword Coverage:   {aggregated['retrieval']['avg_keyword_coverage']:.1%}")
        print(f"  Avg Similarity:     {aggregated['retrieval']['avg_similarity_score']:.3f}")
        print(f"  Top-1 Similarity:   {aggregated['retrieval']['avg_top1_similarity']:.3f}")
        
        print(f"\n💬 ANSWER QUALITY:")
        print(f"  Keyword Score:      {aggregated['answer']['avg_keyword_score']:.1%}")
        print(f"  Attribution Rate:   {aggregated['answer']['attribution_rate']:.1%}")
        print(f"  Avg Answer Length:  {aggregated['answer']['avg_answer_length']:.0f} words")
        
        print(f"\n📈 OVERALL:")
        print(f"  Total Tests:        {aggregated['overall']['total_tests']}")
        print(f"  Timestamp:          {aggregated['overall']['timestamp']}")
        
        print("\n" + "="*70 + "\n")


# Example usage
if __name__ == "__main__":
    from rag_engine import RAGEngine
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Check if documents are indexed
    stats = rag.vector_store.get_stats()
    if stats['total_vectors'] == 0:
        print("⚠️  No documents indexed. Please run:")
        print("   python main.py --mode index --docs data/documents")
    else:
        # Run evaluation
        evaluator = RAGEvaluator(rag)
        results = evaluator.run_evaluation(save_results=True)
        
        print("✓ Evaluation complete!")
        print("  Check evaluation_results/ for detailed results")
