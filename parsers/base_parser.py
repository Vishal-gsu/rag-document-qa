"""
Base Parser Interface
Abstract class defining the parser contract.
"""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize parser.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def parse(self, content: str, metadata: Dict) -> List[Dict]:
        """
        Parse document content into chunks with enhanced metadata.
        
        Args:
            content: Raw document text
            metadata: Document-level metadata (filename, source, etc.)
            
        Returns:
            List of chunk dictionaries with text and metadata
            Each chunk: {
                'text': str,
                'metadata': {
                    'filename': str,
                    'source': str,
                    'type': str,
                    'doc_type': str,
                    'chunk_id': int,
                    'total_chunks': int,
                    ... (parser-specific fields)
                }
            }
        """
        pass
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap (helper method).
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings within last 100 chars of chunk
                chunk_end_search = text[max(start, end - 100):end]
                sentence_endings = ['. ', '.\n', '? ', '?\n', '! ', '!\n']
                
                best_break = -1
                for ending in sentence_endings:
                    pos = chunk_end_search.rfind(ending)
                    if pos > best_break:
                        best_break = pos
                
                if best_break != -1:
                    # Adjust end to sentence boundary
                    end = max(start, end - 100) + best_break + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _create_chunk_metadata(self, base_metadata: Dict, chunk_id: int, 
                              total_chunks: int, **extra_fields) -> Dict:
        """
        Create metadata dictionary for a chunk.
        
        Args:
            base_metadata: Document-level metadata
            chunk_id: Chunk index (0-based)
            total_chunks: Total number of chunks
            **extra_fields: Additional parser-specific fields
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            'filename': base_metadata.get('filename', 'Unknown'),
            'source': base_metadata.get('source', ''),
            'type': base_metadata.get('type', ''),
            'doc_type': base_metadata.get('doc_type', 'generic'),
            'chunk_id': chunk_id,
            'total_chunks': total_chunks
        }
        
        # Add parser-specific fields
        metadata.update(extra_fields)
        
        return metadata
