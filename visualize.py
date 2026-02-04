"""
Visualization Module for RAG Evaluation Results
Creates charts and graphs for performance analysis.
"""
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class RAGVisualizer:
    """Create visualizations for RAG evaluation results."""
    
    def __init__(self, results_file: str = None):
        """
        Initialize visualizer.
        
        Args:
            results_file: Path to detailed results JSON file
        """
        self.results_file = results_file
        self.results = None
        self.aggregated = None
        
        if results_file:
            self.load_results(results_file)
    
    def load_results(self, results_file: str):
        """Load results from JSON file."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        print(f"✓ Loaded {len(self.results)} test results")
    
    def load_latest_results(self):
        """Load the most recent results file."""
        results_dir = Path("evaluation_results")
        if not results_dir.exists():
            print("⚠️  No evaluation results found. Run evaluation first.")
            return False
        
        # Find latest detailed results file
        files = list(results_dir.glob("detailed_results_*.json"))
        if not files:
            print("⚠️  No results files found.")
            return False
        
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        self.load_results(str(latest_file))
        
        # Load corresponding summary
        timestamp = latest_file.stem.replace("detailed_results_", "")
        summary_file = results_dir / f"summary_{timestamp}.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                self.aggregated = json.load(f)
        
        return True
    
    def create_latency_breakdown_chart(self, save_path: str = None):
        """Create pie chart showing latency breakdown."""
        if not self.aggregated:
            print("⚠️  No aggregated results available")
            return
        
        latency = self.aggregated['latency']
        
        # Data
        labels = ['Embedding', 'Retrieval', 'Generation']
        sizes = [
            latency['avg_embedding_ms'],
            latency['avg_retrieval_ms'],
            latency['avg_generation_ms']
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.05, 0.05, 0.05)
        
        # Create figure
        plt.figure(figsize=(10, 7))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Query Latency Breakdown\n' + 
                  f'Total: {sum(sizes):.0f} ms', 
                  fontsize=14, fontweight='bold')
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved chart to {save_path}")
        else:
            plt.savefig('evaluation_results/latency_breakdown.png', dpi=300, bbox_inches='tight')
            print("✓ Saved to evaluation_results/latency_breakdown.png")
        
        plt.close()
    
    def create_similarity_distribution_chart(self, save_path: str = None):
        """Create histogram of similarity scores."""
        if not self.results:
            print("⚠️  No results available")
            return
        
        # Extract similarity scores
        top1_scores = [r['retrieval']['top1_similarity_score'] for r in self.results]
        avg_scores = [r['retrieval']['avg_similarity_score'] for r in self.results]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Top-1 similarity
        ax1.hist(top1_scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(top1_scores), color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean: {np.mean(top1_scores):.3f}')
        ax1.set_xlabel('Similarity Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Top-1 Similarity Score Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Average similarity
        ax2.hist(avg_scores, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(avg_scores), color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {np.mean(avg_scores):.3f}')
        ax2.set_xlabel('Similarity Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Average Similarity Score Distribution', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved chart to {save_path}")
        else:
            plt.savefig('evaluation_results/similarity_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved to evaluation_results/similarity_distribution.png")
        
        plt.close()
    
    def create_metrics_comparison_chart(self, save_path: str = None):
        """Create bar chart comparing different metrics."""
        if not self.aggregated:
            print("⚠️  No aggregated results available")
            return
        
        # Data
        metrics = ['Keyword\nCoverage', 'Avg\nSimilarity', 'Keyword\nScore', 'Attribution\nRate']
        values = [
            self.aggregated['retrieval']['avg_keyword_coverage'],
            self.aggregated['retrieval']['avg_similarity_score'],
            self.aggregated['answer']['avg_keyword_score'],
            self.aggregated['answer']['attribution_rate']
        ]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}\n({value*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('RAG System Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved chart to {save_path}")
        else:
            plt.savefig('evaluation_results/metrics_comparison.png', dpi=300, bbox_inches='tight')
            print("✓ Saved to evaluation_results/metrics_comparison.png")
        
        plt.close()
    
    def create_category_performance_chart(self, save_path: str = None):
        """Create chart showing performance by question category."""
        if not self.results:
            print("⚠️  No results available")
            return
        
        # Group by category
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        # Calculate average scores per category
        cat_names = []
        keyword_scores = []
        similarity_scores = []
        
        for cat, results in categories.items():
            cat_names.append(cat.title())
            keyword_scores.append(np.mean([r['answer']['keyword_score'] for r in results]))
            similarity_scores.append(np.mean([r['retrieval']['avg_similarity_score'] for r in results]))
        
        # Create figure
        x = np.arange(len(cat_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, keyword_scores, width, label='Answer Quality', 
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, similarity_scores, width, label='Retrieval Quality',
                       color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Question Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Question Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved chart to {save_path}")
        else:
            plt.savefig('evaluation_results/category_performance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved to evaluation_results/category_performance.png")
        
        plt.close()
    
    def create_latency_percentiles_chart(self, save_path: str = None):
        """Create chart showing latency percentiles."""
        if not self.results:
            print("⚠️  No results available")
            return
        
        # Extract latencies
        total_latencies = [r['latency']['total_ms'] for r in self.results]
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        values = [np.percentile(total_latencies, p) for p in percentiles]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bars = plt.bar([f'P{p}' for p in percentiles], values, 
                       color='#9b59b6', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f} ms',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        plt.title('Query Latency Percentiles', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved chart to {save_path}")
        else:
            plt.savefig('evaluation_results/latency_percentiles.png', dpi=300, bbox_inches='tight')
            print("✓ Saved to evaluation_results/latency_percentiles.png")
        
        plt.close()
    
    def create_all_charts(self):
        """Generate all visualization charts."""
        print("\n📊 Generating all visualization charts...\n")
        
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        self.create_latency_breakdown_chart()
        self.create_similarity_distribution_chart()
        self.create_metrics_comparison_chart()
        self.create_category_performance_chart()
        self.create_latency_percentiles_chart()
        
        print("\n✓ All charts generated in evaluation_results/")
        print("\nGenerated files:")
        print("  - latency_breakdown.png")
        print("  - similarity_distribution.png")
        print("  - metrics_comparison.png")
        print("  - category_performance.png")
        print("  - latency_percentiles.png")


# Example usage
if __name__ == "__main__":
    visualizer = RAGVisualizer()
    
    if visualizer.load_latest_results():
        visualizer.create_all_charts()
    else:
        print("\n⚠️  No evaluation results found.")
        print("Run evaluation first: python evaluation.py")
