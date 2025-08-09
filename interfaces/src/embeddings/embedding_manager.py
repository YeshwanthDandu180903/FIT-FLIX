"""Embedding management for FIT-FLIX RAG system."""

import logging
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages embeddings for the RAG system."""
    
    def __init__(self, config):
        """Initialize embedding manager.
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.initialize()
    
    def initialize(self):
        """Initialize the embedding model."""
        try:
            self.model = SentenceTransformer(self.config.embedding_model)
            self.logger.info(f"Initialized embedding model: {self.config.embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def get_embedding_function(self):
        """Get ChromaDB compatible embedding function.
        
        Returns:
            ChromaDB embedding function
        """
        class SentenceTransformerEmbeddings:
            def __init__(self, model, model_name):
                self.model = model
                self.name = model_name  # Add the name attribute
            
            def __call__(self, input_texts: List[str]) -> List[List[float]]:
                """Generate embeddings for input texts.
                
                Args:
                    input_texts: List of text strings
                    
                Returns:
                    List of embedding vectors
                """
                if isinstance(input_texts, str):
                    input_texts = [input_texts]
                return self.model.encode(input_texts).tolist()
        
        return SentenceTransformerEmbeddings(self.model, self.config.embedding_model)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"status": "not_initialized"}
        
        return {
            "model_name": self.config.embedding_model,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "status": "ready"
        }