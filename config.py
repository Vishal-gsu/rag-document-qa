"""
Configuration management for the RAG application.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG application."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    
    # Vector Database Configuration
    ENDEE_DB_PATH = os.getenv("ENDEE_DB_PATH", "./data/vectordb")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_embeddings")
    
    # Multi-Collection Configuration
    ENABLE_MULTI_COLLECTION = os.getenv("ENABLE_MULTI_COLLECTION", "false").lower() == "true"
    
    # Collection naming strategy
    COLLECTION_NAMES = {
        'research_paper': 'research_papers',
        'resume': 'resumes',
        'textbook': 'textbooks',
        'generic': 'general_docs'
    }
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # Increased from 500 to 1000
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # Increased from 50 to 200
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))  # Increased from 3 to 5
    
    # Document Classification
    ENABLE_AUTO_CLASSIFICATION = os.getenv("ENABLE_AUTO_CLASSIFICATION", "true").lower() == "true"
    CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "2.0"))
    
    # Specialized Parsing
    ENABLE_SPECIALIZED_PARSING = os.getenv("ENABLE_SPECIALIZED_PARSING", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in .env file"
            )
        return True
    
    @classmethod
    def get_collection_name(cls, doc_type: str) -> str:
        """
        Get collection name for a document type.
        
        Args:
            doc_type: Document type from classifier
            
        Returns:
            Collection name
        """
        if cls.ENABLE_MULTI_COLLECTION:
            return cls.COLLECTION_NAMES.get(doc_type, 'general_docs')
        else:
            # Legacy mode: single collection for all
            return cls.COLLECTION_NAME
