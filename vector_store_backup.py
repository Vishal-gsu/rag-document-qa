"""
Vector Store Module
Interface for Endee Vector Database operations.
"""
import os
from typing import List, Dict, Optional
import pickle
from pathlib import Path


class VectorStore:
    """
    Endee Vector Database interface for storing and retrieving embeddings.
    
    Note: This is a simplified implementation. Replace with actual Endee SDK
    when available. Current implementation uses a file-based approach for
    demonstration purposes.
    """
    
    def __init__(self, db_path: str = "./data/vectordb", collection_name: str = "documents"):
        """
        Initialize vector store.
        
        Args:
            db_path: Path to database directory
            collection_name: Name of the collection
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.collection_path = self.db_path / f"{collection_name}.pkl"
        
        # Create directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load collection
        self.vectors = []
        self.metadata = []
        self._load_collection()
    
    def _load_collection(self):
        """Load existing collection from disk."""
        if self.collection_path.exists():
            try:
                with open(self.collection_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectors = data.get('vectors', [])
                    self.metadata = data.get('metadata', [])
                print(f"📂 Loaded collection '{self.collection_name}' with {len(self.vectors)} vectors")
            except Exception as e:
                print(f"⚠️ Error loading collection: {e}. Starting fresh.")
                self.vectors = []
                self.metadata = []
        else:
            print(f"📂 Created new collection '{self.collection_name}'")
    
    def _save_collection(self):
        """Save collection to disk."""
        try:
            data = {
                'vectors': self.vectors,
                'metadata': self.metadata
            }
            with open(self.collection_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved {len(self.vectors)} vectors to collection")
        except Exception as e:
            print(f"✗ Error saving collection: {e}")
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict]):
        """
        Add vectors to the collection.
        
        Args:
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries (one per vector)
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)
        self._save_collection()
        
        print(f"✓ Added {len(vectors)} vectors to collection")
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents with embeddings to the collection.
        
        Args:
            documents: List of dicts with 'embedding', 'text', and 'metadata'
        """
        vectors = [doc['embedding'] for doc in documents]
        metadata = [
            {
                'text': doc['text'],
                **doc['metadata']
            }
            for doc in documents
        ]
        
        self.add_vectors(vectors, metadata)
    
    def search(self, query_vector: List[float], top_k: int = 3, 
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {'type': '.pdf'})
            
        Returns:
            List of result dictionaries with 'score', 'text', and 'metadata'
        """
        if not self.vectors:
            print("⚠️ No vectors in collection")
            return []
        
        # Calculate similarities
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Apply metadata filter if specified
            if filter_metadata:
                match = all(
                    self.metadata[i].get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue
            
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append({
                'index': i,
                'score': similarity,
                'text': self.metadata[i].get('text', ''),
                'metadata': self.metadata[i]
            })
        
        # Sort by similarity (descending) and get top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        results = similarities[:top_k]
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def clear_collection(self):
        """Clear all vectors from the collection."""
        self.vectors = []
        self.metadata = []
        self._save_collection()
        print(f"🗑️ Cleared collection '{self.collection_name}'")
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        return {
            'collection_name': self.collection_name,
            'total_vectors': len(self.vectors),
            'dimension': len(self.vectors[0]) if self.vectors else 0,
            'db_path': str(self.db_path)
        }


# Example usage
if __name__ == "__main__":
    # Initialize vector store
    store = VectorStore(db_path="./test_vectordb", collection_name="test_collection")
    
    # Create sample vectors
    sample_vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.9, 0.1, 0.1, 0.1]
    ]
    
    sample_metadata = [
        {'text': 'Machine learning is great', 'source': 'doc1.txt'},
        {'text': 'Deep learning uses neural networks', 'source': 'doc2.txt'},
        {'text': 'Completely different topic', 'source': 'doc3.txt'}
    ]
    
    # Add vectors
    store.add_vectors(sample_vectors, sample_metadata)
    
    # Search
    query_vector = [0.15, 0.25, 0.35, 0.45]
    results = store.search(query_vector, top_k=2)
    
    print("\n🔍 Search Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f} | Text: {result['text']}")
    
    # Stats
    print("\n📊 Collection Stats:")
    print(store.get_stats())
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_vectordb")
