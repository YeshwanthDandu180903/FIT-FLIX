"""Configuration settings for FIT-FLIX RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for FIT-FLIX RAG system."""
    
    def __init__(self):
        """Initialize configuration settings."""
        # Base directory (project root) - go up two levels from src/config.py
        self.base_dir = Path(__file__).parent.parent.parent
        
        # Directory structure
        self.data_dir = self.base_dir / "data"
        self.knowledge_base_dir = self.data_dir / "knowledge_base"
        self.processed_dir = self.data_dir / "processed"
        self.vector_db_dir = self.base_dir / "vector_db"
        self.logs_dir = self.base_dir / "logs"
        
        # Ensure directories exist
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB settings
        self.chroma_db_path = self.vector_db_dir / "fitflix_chroma_db_gemini"
        self.collection_name = "fitflix_documents"
        
        # API Keys
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Model settings
        self.llm_model = "gemini-1.5-flash"
        self.embedding_model = "all-MiniLM-L6-v2"
        
        # Text processing settings
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Retrieval settings
        self.retrieval_top_k = 5
        self.similarity_threshold = 0.7
        
        # Generation settings
        self.max_tokens = 1000
        self.temperature = 0.7