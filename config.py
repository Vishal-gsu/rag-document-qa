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
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # Increased from 500 to 1000
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # Increased from 50 to 200
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))  # Increased from 3 to 5
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in .env file"
            )
        return True
