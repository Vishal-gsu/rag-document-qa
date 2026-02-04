"""
Vector Store Module using Endee Vector Database
Handles vector storage and similarity search using real Endee HNSW algorithm.
"""
from typing import List, Dict, Optional
import numpy as np
from endee import Endee
import pickle
from pathlib import Path


class VectorStore:
    """Vector store using Endee database with HNSW for efficient similarity search."""
    
    def __init__(self, 
                 db_path: str = "./data/vectordb",
                 collection_name: str = "document_embeddings"):
        """
        Initialize Endee vector store.
        
        Args:
            db_path: Path for metadata storage
            collection_name: Name of the Endee index
        """
        self.collection_name = collection_name
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.db_path / f"{collection_name}_metadata.pkl"
        
        # Initialize Endee client
        self.client = Endee()
        self.index = None
        self.dimension = None
        
        # Metadata storage (Endee handles vectors, we handle metadata)
        self.metadata_store = {}
        self.vector_count = 0
        
        # Load existing data if available
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get('metadata', {})
                    self.dimension = data.get('dimension', None)
                    self.vector_count = len(self.metadata_store)
                
                # Try to get existing Endee index
                if self.dimension:
                    try:
                        self.index = self.client.get_index(self.collection_name)
                        print(f"📂 Loaded collection '{self.collection_name}' with {self.vector_count} vectors")
                    except:
                        print(f"📂 Metadata found. Index will be created on first add.")
            except Exception as e:
                print(f"⚠️ Error loading metadata: {e}")
        else:
            print(f"📂 Created new collection '{self.collection_name}'")
    
    def _ensure_index(self, dimension: int):
        """Ensure Endee index exists with correct dimension."""
        if self.index is not None and self.dimension == dimension:
            return  # Index already exists with correct dimension
        
        # Update dimension
        self.dimension = dimension
        
        # Try to get existing index
        try:
            self.index = self.client.get_index(self.collection_name)
            # Check if dimension matches (we'll assume it does if we get here)
            return
        except:
            pass
        
        # Create new index with HNSW algorithm
        try:
            self.client.create_index(
                name=self.collection_name,
                dimension=dimension,
                space_type="cosine",  # Cosine similarity
                M=16,  # HNSW: connections per layer
                ef_con=200  # HNSW: construction quality
            )
            # Now get the index object
            self.index = self.client.get_index(self.collection_name)
            print(f"✓ Created Endee HNSW index ({dimension}D, cosine similarity)")
        except Exception as e:
            print(f"⚠️ Error creating index: {e}")
            raise
    
    def add_vectors(self, 
                   vectors: List[List[float]], 
                   metadata: List[Dict],
                   ids: Optional[List[str]] = None) -> None:
        """
        Add vectors to Endee index.
        
        Args:
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: Optional IDs (auto-generated if None)
        """
        if not vectors:
            return
        
        # Ensure index exists
        dimension = len(vectors[0])
        self._ensure_index(dimension)
        
        # Convert to numpy
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Generate IDs if needed
        if ids is None:
            ids = [f"vec_{self.vector_count + i}" for i in range(len(vectors))]
        
        # Store metadata
        for vec_id, meta in zip(ids, metadata):
            self.metadata_store[vec_id] = meta
        
        # Add to Endee in batches (max 1000 per batch)
        try:
            batch_size = 1000
            total_added = 0
            
            for i in range(0, len(vectors_np), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_vectors = vectors_np[i:i + batch_size]
                
                # Prepare data in Endee format
                data = [
                    {"id": vec_id, "vector": vec.tolist()}
                    for vec_id, vec in zip(batch_ids, batch_vectors)
                ]
                
                self.index.upsert(data)
                total_added += len(data)
                print(f"  → Batch {i//batch_size + 1}: Added {len(data)} vectors")
            
            self.vector_count += total_added
            self._save_metadata()
            print(f"✓ Added {total_added} vectors to Endee ({len(vectors_np)//batch_size + 1} batches)")
        except Exception as e:
            print(f"✗ Error adding to Endee: {e}")
            raise
    
    def search(self, 
              query_vector: List[float], 
              top_k: int = 5) -> List[Dict]:
        """
        Search for similar vectors using Endee HNSW.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            
        Returns:
            List of results with score, text, metadata
        """
        if self.index is None:
            return []
        
        try:
            # Search with Endee using query method
            results = self.index.query(
                vector=query_vector,
                top_k=top_k
            )
            
            # Format results
            formatted = []
            for result in results:
                vec_id = result.get('id')
                similarity = result.get('similarity', 0.0)
                
                # Debug: Print similarity scores
                print(f"  → Result ID: {vec_id}, Similarity: {similarity:.4f}")
                
                if vec_id in self.metadata_store:
                    meta = self.metadata_store[vec_id]
                    formatted.append({
                        'id': vec_id,
                        'score': float(similarity),
                        'text': meta.get('text', ''),
                        'metadata': meta
                    })
            
            return formatted
            
        except Exception as e:
            print(f"✗ Endee search error: {e}")
            return []
    
    def clear_collection(self) -> None:
        """Clear the collection."""
        try:
            if self.index is not None:
                self.client.delete_index(self.collection_name)
                self.index = None
            
            self.metadata_store = {}
            self.vector_count = 0
            self.dimension = None
            
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            print(f"✓ Cleared collection '{self.collection_name}'")
        except Exception as e:
            print(f"⚠️ Error clearing: {e}")
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            'collection_name': self.collection_name,
            'total_vectors': self.vector_count,
            'dimension': self.dimension or 0,
            'backend': 'Endee (HNSW)'
        }
    
    def get_indexed_files(self) -> set:
        """Get set of already indexed filenames."""
        indexed_files = set()
        for meta in self.metadata_store.values():
            if 'filename' in meta:
                indexed_files.add(meta['filename'])
        return indexed_files
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            data = {
                'metadata': self.metadata_store,
                'dimension': self.dimension,
                'vector_count': self.vector_count
            }
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")
