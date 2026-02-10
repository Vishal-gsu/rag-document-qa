"""
Generic Parser
Default parser for unclassified documents using simple chunking.
"""
from typing import List, Dict
from .base_parser import BaseParser


class GenericParser(BaseParser):
    """
    Generic document parser.
    Uses simple text chunking without document-specific features.
    Fallback for unclassified documents.
    """
    
    def parse(self, content: str, metadata: Dict) -> List[Dict]:
        """
        Parse generic document into chunks.
        
        Args:
            content: Raw document text
            metadata: Document-level metadata
            
        Returns:
            List of chunks with basic metadata
        """
        if not content or not content.strip():
            return []
        
        # Simple text splitting
        chunks = self._split_text(content)
        
        # Create chunk dictionaries
        result = []
        total_chunks = len(chunks)
        
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = self._create_chunk_metadata(
                base_metadata=metadata,
                chunk_id=i,
                total_chunks=total_chunks
            )
            
            result.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return result
