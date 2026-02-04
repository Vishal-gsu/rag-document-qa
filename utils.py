"""
Utility Functions
Helper functions for the RAG system.
"""
import re
from typing import List
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count (rough approximation).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4


def format_source_citation(metadata: dict) -> str:
    """
    Format metadata into a source citation.
    
    Args:
        metadata: Document metadata
        
    Returns:
        Formatted citation string
    """
    filename = metadata.get('filename', 'Unknown')
    chunk_id = metadata.get('chunk_id', 0)
    
    return f"[{filename}, chunk {chunk_id}]"


def get_file_extension(filepath: str) -> str:
    """
    Get file extension from filepath.
    
    Args:
        filepath: Path to file
        
    Returns:
        File extension (e.g., '.txt')
    """
    return Path(filepath).suffix.lower()


def ensure_directory(directory: str):
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_results_table(results: List[dict]) -> str:
    """
    Format search results as a table.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No results found"
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Rank':<6} {'Score':<8} {'Source':<20} {'Preview':<44}")
    lines.append("=" * 80)
    
    for i, result in enumerate(results):
        rank = i + 1
        score = f"{result['score']:.4f}"
        source = truncate_text(result['metadata'].get('filename', 'Unknown'), 18)
        preview = truncate_text(result['text'], 42)
        
        lines.append(f"{rank:<6} {score:<8} {source:<20} {preview:<44}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Test clean_text
    dirty_text = "  This   is    messy    text!!!   "
    clean = clean_text(dirty_text)
    print(f"Clean: '{clean}'")
    
    # Test truncate
    long_text = "This is a very long text that needs to be truncated for display purposes"
    short = truncate_text(long_text, 30)
    print(f"Truncated: '{short}'")
    
    # Test token estimate
    text = "Machine learning is fascinating and powerful"
    tokens = count_tokens_estimate(text)
    print(f"Estimated tokens: {tokens}")
