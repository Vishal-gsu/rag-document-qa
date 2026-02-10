"""
Embedding Engine Module
Handles text-to-vector conversion using local BGE model (state-of-the-art, free).
"""
from typing import List
import numpy as np


class EmbeddingEngine:
    """Generate embeddings using local sentence-transformers model (BGE)."""
    
    def __init__(self, model: str = None, use_local: bool = True):
        """
        Initialize embedding engine with local BGE model.
        
        Args:
            model: Model name (default: BAAI/bge-large-en-v1.5)
            use_local: Always True (kept for backward compatibility)
        """
        # Always use local BGE embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.model_name = model or "BAAI/bge-large-en-v1.5"
            print(f"Loading local embedding model: {self.model_name}...")
            self.local_model = SentenceTransformer(self.model_name)
            self.dimension = self.local_model.get_sentence_embedding_dimension()
            print(f"✓ Local embeddings ready (dimension: {self.dimension})")
        except ImportError:
            raise ValueError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using local BGE model."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            embedding = self.local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Error generating embedding: {e}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        try:
            embeddings = self.local_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Error generating batch embeddings: {e}")
    
    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """Embed a list of document chunks."""
        texts = [doc['text'] for doc in documents]
        
        print(f"\n🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed_batch(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        print(f"✓ All embeddings generated ({self.dimension}D vectors)\n")
        return documents
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
