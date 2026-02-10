"""
Vector Store Module using Endee Vector Database
Handles vector storage and similarity search using real Endee HNSW algorithm.
Supports multi-collection architecture for document type segregation.
"""
from typing import List, Dict, Optional
import numpy as np
from endee import Endee
import pickle
from pathlib import Path
from datetime import datetime


class VectorStore:
    """
    Vector store using Endee database with HNSW for efficient similarity search.
    Supports multiple collections for different document types.
    """
    
    def __init__(self, 
                 db_path: str = "./data/vectordb",
                 collection_name: str = "document_embeddings",
                 enable_multi_collection: bool = False):
        """
        Initialize Endee vector store.
        
        Args:
            db_path: Path for metadata storage
            collection_name: Name of the default/primary Endee index
            enable_multi_collection: Enable multi-collection mode (new architecture)
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Endee client (shared across all collections)
        self.client = Endee()
        
        # Multi-collection mode
        self.enable_multi_collection = enable_multi_collection
        
        if enable_multi_collection:
            # New architecture: manage multiple collections
            self.collections = {}  # {name: {'index': index_obj, 'dimension': int}}
            self.metadata_stores = {}  # {name: {vec_id: metadata}}
            self.default_collection = collection_name
            
            # Load all existing collections
            self._load_all_collections()
        else:
            # Legacy single-collection mode (backward compatible)
            self.collection_name = collection_name
            self.metadata_file = self.db_path / f"{collection_name}_metadata.pkl"
            self.index = None
            self.dimension = None
            self.metadata_store = {}
            self.vector_count = 0
            self._load_metadata()
    
    
    def _load_all_collections(self):
        """Load all existing collections from disk (multi-collection mode)."""
        # Find all metadata files
        for meta_file in self.db_path.glob("*_metadata.pkl"):
            collection_name = meta_file.stem.replace("_metadata", "")
            self._load_collection(collection_name)
        
        if not self.collections:
            print(f"No existing collections found. Multi-collection mode enabled.")
        else:
            print(f"Loaded {len(self.collections)} existing collections")
    
    def _load_collection(self, collection_name: str):
        """Load a specific collection (multi-collection mode)."""
        metadata_file = self.db_path / f"{collection_name}_metadata.pkl"
        
        if not metadata_file.exists():
            return
        
        try:
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                metadata_store = data.get('metadata', {})
                dimension = data.get('dimension', None)
            
            # Try to get Endee index
            index = None
            if dimension:
                try:
                    index = self.client.get_index(collection_name)
                except:
                    pass
            
            # Store collection info
            self.collections[collection_name] = {
                'index': index,
                'dimension': dimension,
                'vector_count': len(metadata_store)
            }
            self.metadata_stores[collection_name] = metadata_store
            
            print(f"  ✓ Loaded collection '{collection_name}' ({len(metadata_store)} vectors)")
            
        except Exception as e:
            print(f"  ⚠️ Error loading collection '{collection_name}': {e}")
    
    def _load_metadata(self):
        """Load metadata from disk (legacy single-collection mode)."""
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
                        print(f"Loaded collection '{self.collection_name}' with {self.vector_count} vectors")
                    except:
                        print(f"Metadata found. Index will be created on first add.")
            except Exception as e:
                print(f"⚠️ Error loading metadata: {e}")
        else:
            print(f"Created new collection '{self.collection_name}'")
    
    def get_or_create_collection(self, name: str, dimension: int) -> Dict:
        """
        Get or create a collection (multi-collection mode).
        
        Args:
            name: Collection name
            dimension: Vector dimension
            
        Returns:
            Collection info dictionary
        """
        if not self.enable_multi_collection:
            raise RuntimeError("Multi-collection mode not enabled. Set enable_multi_collection=True")
        
        # Check if collection exists
        if name in self.collections:
            coll = self.collections[name]
            if coll['dimension'] != dimension:
                raise ValueError(
                    f"Collection '{name}' exists with dimension {coll['dimension']}, "
                    f"but requested dimension {dimension}"
                )
            return coll
        
        # Create new collection
        try:
            self.client.create_index(
                name=name,
                dimension=dimension,
                space_type="cosine",
                M=16,
                ef_con=200
            )
            index = self.client.get_index(name)
            
            self.collections[name] = {
                'index': index,
                'dimension': dimension,
                'vector_count': 0
            }
            self.metadata_stores[name] = {}
            
            print(f"✓ Created collection '{name}' ({dimension}D)")
            return self.collections[name]
            
        except Exception as e:
            print(f"⚠️ Error creating collection '{name}': {e}")
            raise
    
    def add_vectors_to_collection(self,
                                  collection_name: str,
                                  vectors: List[List[float]],
                                  metadata: List[Dict],
                                  ids: Optional[List[str]] = None) -> None:
        """
        Add vectors to a specific collection (multi-collection mode).
        
        Args:
            collection_name: Target collection name
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: Optional IDs (auto-generated if None)
        """
        if not self.enable_multi_collection:
            raise RuntimeError("Multi-collection mode not enabled")
        
        if not vectors:
            return
        
        # Ensure collection exists
        dimension = len(vectors[0])
        self.get_or_create_collection(collection_name, dimension)
        
        coll = self.collections[collection_name]
        index = coll['index']
        
        # Convert to numpy
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Generate IDs if needed
        if ids is None:
            base_count = coll['vector_count']
            ids = [f"{collection_name}_vec_{base_count + i}" for i in range(len(vectors))]
        
        # Store metadata
        for vec_id, meta in zip(ids, metadata):
            self.metadata_stores[collection_name][vec_id] = meta
        
        # Add to Endee in batches
        try:
            batch_size = 1000
            total_added = 0
            
            for i in range(0, len(vectors_np), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_vectors = vectors_np[i:i + batch_size]
                
                data = [
                    {"id": vec_id, "vector": vec.tolist()}
                    for vec_id, vec in zip(batch_ids, batch_vectors)
                ]
                
                index.upsert(data)
                total_added += len(data)
            
            # Update counts
            coll['vector_count'] += total_added
            self._save_collection_metadata(collection_name)
            
            print(f"✓ Added {total_added} vectors to collection '{collection_name}'")
            
        except Exception as e:
            print(f"✗ Error adding to collection '{collection_name}': {e}")
            raise
    
    def search_collection(self,
                         collection_name: str,
                         query_vector: List[float],
                         top_k: int = 5) -> List[Dict]:
        """
        Search a specific collection (multi-collection mode).
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            top_k: Number of results
            
        Returns:
            List of results with score, text, metadata
        """
        if not self.enable_multi_collection:
            raise RuntimeError("Multi-collection mode not enabled")
        
        if collection_name not in self.collections:
            return []
        
        coll = self.collections[collection_name]
        index = coll['index']
        
        if index is None:
            return []
        
        try:
            results = index.query(vector=query_vector, top_k=top_k)
            
            formatted = []
            for result in results:
                vec_id = result.get('id')
                similarity = result.get('similarity', 0.0)
                
                if vec_id in self.metadata_stores[collection_name]:
                    meta = self.metadata_stores[collection_name][vec_id]
                    formatted.append({
                        'id': vec_id,
                        'score': float(similarity),
                        'text': meta.get('text', ''),
                        'metadata': meta,
                        'collection': collection_name
                    })
            
            return formatted
            
        except Exception as e:
            print(f"✗ Search error in collection '{collection_name}': {e}")
            return []
    
    def search_multi_collection(self,
                               query_vector: List[float],
                               collections: List[str] = None,
                               top_k: int = 5) -> List[Dict]:
        """
        Search across multiple collections and merge results.
        
        Args:
            query_vector: Query embedding
            collections: List of collection names (None = all collections)
            top_k: Total number of results to return
            
        Returns:
            Merged and re-ranked results from all collections
        """
        if not self.enable_multi_collection:
            raise RuntimeError("Multi-collection mode not enabled")
        
        # Default to all collections
        if collections is None:
            collections = list(self.collections.keys())
        
        # Search each collection
        all_results = []
        for coll_name in collections:
            if coll_name in self.collections:
                # Get more results per collection, then re-rank
                results = self.search_collection(coll_name, query_vector, top_k * 2)
                all_results.extend(results)
        
        # Sort by similarity score and return top-k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def list_collections(self) -> List[Dict]:
        """Get list of all collections with statistics."""
        if not self.enable_multi_collection:
            return [{
                'name': self.collection_name,
                'vector_count': self.vector_count,
                'dimension': self.dimension or 0
            }]
        
        return [
            {
                'name': name,
                'vector_count': info['vector_count'],
                'dimension': info['dimension'] or 0
            }
            for name, info in self.collections.items()
        ]
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics for a specific collection."""
        if not self.enable_multi_collection:
            if collection_name == self.collection_name:
                return self.get_stats()
            return {}
        
        if collection_name not in self.collections:
            return {}
        
        coll = self.collections[collection_name]
        return {
            'collection_name': collection_name,
            'total_vectors': coll['vector_count'],
            'dimension': coll['dimension'] or 0,
            'backend': 'Endee (HNSW)'
        }
    
    def _save_collection_metadata(self, collection_name: str):
        """Save metadata for a specific collection."""
        if not self.enable_multi_collection:
            return
        
        if collection_name not in self.collections:
            return
        
        metadata_file = self.db_path / f"{collection_name}_metadata.pkl"
        coll = self.collections[collection_name]
        
        try:
            data = {
                'metadata': self.metadata_stores[collection_name],
                'dimension': coll['dimension'],
                'vector_count': coll['vector_count']
            }
            with open(metadata_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Error saving metadata for '{collection_name}': {e}")
    
    def _ensure_index(self, dimension: int):
        """Ensure Endee index exists with correct dimension (legacy mode)."""
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
            if self.enable_multi_collection:
                # Multi-collection mode: clear all collections
                for col_name in list(self.collections.keys()):
                    try:
                        if self.collections[col_name]['index'] is not None:
                            self.client.delete_index(col_name)
                    except Exception as e:
                        print(f"⚠️ Error deleting index '{col_name}': {e}")
                
                self.collections = {}
                self.metadata_stores = {}
                
                # Delete all metadata files
                for meta_file in self.db_path.glob("*_metadata.pkl"):
                    meta_file.unlink()
                
                print(f"✓ Cleared all collections")
            else:
                # Legacy single-collection mode
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
        if self.enable_multi_collection:
            # Multi-collection mode: aggregate stats
            total_vectors = sum(col['vector_count'] for col in self.collections.values())
            dimensions = [col['dimension'] for col in self.collections.values() if col['dimension']]
            dimension = dimensions[0] if dimensions else 0
            return {
                'collection_name': f"{len(self.collections)} collections",
                'total_vectors': total_vectors,
                'dimension': dimension,
                'backend': 'Endee (HNSW)'
            }
        else:
            # Legacy single-collection mode
            return {
                'collection_name': self.collection_name,
                'total_vectors': self.vector_count,
                'dimension': self.dimension or 0,
                'backend': 'Endee (HNSW)'
            }
    
    
    def get_indexed_files(self, collection_name: str = None) -> set:
        """
        Get set of already indexed filenames.
        
        Args:
            collection_name: Specific collection (None = default collection or all in multi-mode)
            
        Returns:
            Set of indexed filenames
        """
        indexed_files = set()
        
        if self.enable_multi_collection:
            if collection_name:
                # Specific collection
                if collection_name in self.metadata_stores:
                    for meta in self.metadata_stores[collection_name].values():
                        if 'filename' in meta:
                            indexed_files.add(meta['filename'])
            else:
                # All collections
                for coll_metadata in self.metadata_stores.values():
                    for meta in coll_metadata.values():
                        if 'filename' in meta:
                            indexed_files.add(meta['filename'])
        else:
            # Legacy mode
            for meta in self.metadata_store.values():
                if 'filename' in meta:
                    indexed_files.add(meta['filename'])
        
        return indexed_files
    
    def _save_metadata(self):
        """Save metadata to disk (legacy single-collection mode)."""
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
