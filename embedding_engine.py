"""
Embedding Engine Module
Handles text-to-vector conversion using OpenAI or local models.
"""
from typing import List, Union
import numpy as np
from openai import OpenAI
from config import Config


class EmbeddingEngine:
    """Generate embeddings using OpenAI or local models."""
    
    def __init__(self, model: str = None, api_key: str = None, use_local: bool = False):
        """
        Initialize embedding engine.
        
        Args:
            model: Model name (OpenAI or sentence-transformers model)
            api_key: OpenAI API key (if using OpenAI)
            use_local: Use local sentence-transformers model instead of OpenAI
        """
        self.use_local = use_local
        
        if use_local:
            # Use local sentence-transformers model
            try:
                from sentence_transformers import SentenceTransformer
                self.model_name = model or "BAAI/bge-large-en-v1.5"  # Best FREE model, 1024 dimensions
                print(f"Loading local embedding model: {self.model_name}...")
                self.local_model = SentenceTransformer(self.model_name)
                self.dimension = self.local_model.get_sentence_embedding_dimension()
                print(f"✓ Local embeddings ready (dimension: {self.dimension})")
            except ImportError:
                raise ValueError("sentence-transformers not installed. Run: pip install sentence-transformers")
        else:
            # Use OpenAI API
            self.model = model or Config.EMBEDDING_MODEL
            self.api_key = api_key or Config.OPENAI_API_KEY
            
            if not self.api_key:
                raise ValueError("OpenAI API key is required for cloud embeddings")
            
            self.client = OpenAI(api_key=self.api_key)
            self.dimension = self._get_embedding_dimension()
    
    def _get_embedding_dimension(self) -> int:
        """
        Get embedding dimension for the model.
        
        Returns:
            Embedding dimension size
        """
        dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536,
        }
        return dimensions.get(self.model, 1536)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if self.use_local:
            # Use local sentence-transformers
            try:
                embedding = self.local_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                raise Exception(f"Error generating local embedding: {e}")
        else:
            # Use OpenAI API
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                raise Exception(f"Error generating embedding: {e}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to embed per API call
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self.use_local:
            # Use local sentence-transformers (batch processing)
            try:
                print(f"Generating embeddings for {len(texts)} texts using local model...")
                embeddings = self.local_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                return embeddings.tolist()
            except Exception as e:
                raise Exception(f"Error generating local embeddings: {e}")
        else:
            # Use OpenAI API
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    
                    # Extract embeddings in correct order
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    print(f"✓ Embedded batch {i//batch_size + 1} ({len(batch)} texts)")
                    
                except Exception as e:
                    print(f"✗ Error in batch {i//batch_size + 1}: {e}")
                    # Add zero vectors as fallback
                    all_embeddings.extend([[0.0] * self.dimension] * len(batch))
            
            return all_embeddings
    
    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """
        Embed a list of document chunks.
        
        Args:
            documents: List of dicts with 'text' and 'metadata'
            
        Returns:
            Documents with added 'embedding' field
        """
        texts = [doc['text'] for doc in documents]
        
        print(f"\n🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed_batch(texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        print(f"✓ All embeddings generated ({self.dimension}D vectors)\n")
        return documents
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0 to 1)
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# Example usage
if __name__ == "__main__":
    # This will fail without API key - for demonstration only
    try:
        engine = EmbeddingEngine()
        
        # Embed single text
        text = "Machine learning is fascinating"
        embedding = engine.embed_text(text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Embed multiple texts
        texts = [
            "Deep learning uses neural networks",
            "Natural language processing is a subfield of AI",
            "Computer vision enables image recognition"
        ]
        embeddings = engine.embed_batch(texts)
        print(f"\nGenerated {len(embeddings)} embeddings")
        
        # Calculate similarity
        similarity = engine.cosine_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between first two: {similarity:.4f}")
        
    except Exception as e:
        print(f"Note: {e}")
        print("Set OPENAI_API_KEY to run this example")
