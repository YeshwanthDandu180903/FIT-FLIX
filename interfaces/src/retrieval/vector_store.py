"""Vector store management for FIT-FLIX RAG System."""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
from ..config import Config
from ..embeddings.embedding_manager import EmbeddingManager


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, config: Optional[Config] = None, collection_name: str = "fitflix_documents"):
        """Initialize the vector store.
        
        Args:
            config: Configuration object
            collection_name: Name of the ChromaDB collection
        """
        self.config = config or Config()
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager(self.config)
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the ChromaDB client and collection."""
        try:
            # Create vector db directory if it doesn't exist
            db_path = Path(self.config.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                # Create new collection with custom embedding function
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            
            # Load embedding model
            self.embedding_manager.load_model()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries for each document
            ids: List of document IDs (generated if not provided)
        """
        if self.collection is None:
            self.initialize()
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Generate default metadata if not provided
            if metadatas is None:
                metadatas = [{"source": "unknown"} for _ in documents]
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_manager.encode_documents(documents)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def search(self, query: str, n_results: int = 5, 
               where: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
        """Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter conditions
            
        Returns:
            Search results dictionary
        """
        if self.collection is None:
            self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.encode_query(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search documents: {str(e)}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        if self.collection is None:
            self.initialize()
        
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents from vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {str(e)}")
            raise
    
    def update_documents(self, ids: List[str], documents: List[str],
                        metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Update existing documents in the vector store.
        
        Args:
            ids: List of document IDs to update
            documents: List of updated document texts
            metadatas: List of updated metadata dictionaries
        """
        if self.collection is None:
            self.initialize()
        
        try:
            # Generate new embeddings
            embeddings = self.embedding_manager.encode_documents(documents)
            
            # Update collection
            update_data = {
                "ids": ids,
                "embeddings": embeddings.tolist(),
                "documents": documents
            }
            
            if metadatas is not None:
                update_data["metadatas"] = metadatas
            
            self.collection.update(**update_data)
            self.logger.info(f"Updated {len(ids)} documents in vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to update documents: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Collection information dictionary
        """
        if self.collection is None:
            return {"initialized": False}
        
        try:
            count = self.collection.count()
            return {
                "initialized": True,
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_manager.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            return {"initialized": True, "error": str(e)}
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all documents)."""
        if self.collection is None:
            self.initialize()
        
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Reset collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {str(e)}")
            raise
