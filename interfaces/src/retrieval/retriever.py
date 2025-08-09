"""Document retrieval functionality for FIT-FLIX RAG system."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging

from ..embeddings.embedding_manager import EmbeddingManager


class DocumentRetriever:
    """Handles document retrieval from vector database."""
    
    def __init__(self, config):
        """Initialize document retriever.
        
        Args:
            config: Configuration object containing database settings
        """
        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize the ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.config.chroma_db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Try to get existing collection first
            try:
                self.collection = self.client.get_or_create_collection(name=self.config.collection_name)
                self.logger.info(f"Loaded existing collection '{self.config.collection_name}'")
            except ValueError:
                # Collection doesn't exist, create it without embedding function
                # We'll handle embeddings manually
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"description": "FIT-FLIX fitness knowledge base"}
                )
                self.logger.info(f"Created new collection '{self.config.collection_name}'")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize retriever: {str(e)}")
            raise
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: List of metadata dictionaries
            ids: Optional list of document IDs
        """
        if not self.collection:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        
        try:
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Generate embeddings manually
            embeddings = self.embedding_manager.embed_texts(documents)
            
            # Ensure we don't exceed collection limits
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
            
            self.logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.collection:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        
        try:
            # Check if collection is empty
            count = self.collection.count()
            if count == 0:
                self.logger.warning("Collection is empty. No documents to retrieve.")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_text(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count)
            )
            
            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    documents.append(doc)
            
            self.logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {str(e)}")
            return []
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary containing collection statistics
        """
        if not self.collection:
            return {'document_count': 0, 'status': 'not_initialized'}
        
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': self.config.collection_name,
                'status': 'ready'
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {'document_count': 0, 'status': 'error', 'error': str(e)}
    
    def delete_collection(self):
        """Delete the collection (useful for testing/reset)."""
        if self.client:
            try:
                self.client.delete_collection(self.config.collection_name)
                self.logger.info(f"Deleted collection '{self.config.collection_name}'")
            except Exception as e:
                self.logger.warning(f"Could not delete collection: {str(e)}")
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        try:
            self.delete_collection()
            self.initialize()
            self.logger.info("Collection reset successfully")
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {str(e)}")
            raise