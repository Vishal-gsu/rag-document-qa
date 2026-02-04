"""
Document Processor Module
Handles loading and chunking of various document formats.
"""
import os
from typing import List, Dict
from pathlib import Path


class DocumentProcessor:
    """Process and chunk documents for RAG system."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory: str) -> List[Dict[str, str]]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of document dictionaries with 'content' and 'metadata'
        """
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Supported file extensions
        supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
        
        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    content = self._load_file(file_path)
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': str(file_path),
                            'filename': file_path.name,
                            'type': file_path.suffix
                        }
                    })
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"✗ Error loading {file_path.name}: {e}")
        
        print(f"\n📄 Total documents loaded: {len(documents)}")
        return documents
    
    def _load_file(self, file_path: Path) -> str:
        """
        Load content from a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        extension = file_path.suffix.lower()
        
        if extension in {'.txt', '.md'}:
            return self._load_text_file(file_path)
        elif extension == '.pdf':
            return self._load_pdf_file(file_path)
        elif extension == '.docx':
            return self._load_docx_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_pdf_file(self, file_path: Path) -> str:
        """Load PDF file."""
        try:
            import PyPDF2
            text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except ImportError:
            raise ImportError("PyPDF2 required for PDF support. Install: pip install PyPDF2")
    
    def _load_docx_file(self, file_path: Path) -> str:
        """Load DOCX file."""
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx required for DOCX support. Install: pip install python-docx")
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Split documents into chunks for embedding.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        all_chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata']
            
            chunks = self._split_text(content)
            
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                })
        
        print(f"📦 Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Strategy: Split by sentences/paragraphs while respecting chunk_size
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            
            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings (., !, ?, \n)
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_newline = text.rfind('\n', start, end)
                
                # Use the latest sentence boundary
                boundary = max(last_period, last_question, last_exclamation, last_newline)
                
                if boundary > start:
                    end = boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start forward with overlap
            start = end - self.chunk_overlap
        
        return chunks


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Example text
    sample_text = """
    Machine learning is a subset of artificial intelligence. 
    It focuses on building systems that learn from data.
    Deep learning uses neural networks with multiple layers.
    """ * 10
    
    chunks = processor._split_text(sample_text)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}: {chunk[:100]}...")
