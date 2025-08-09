"""Text splitting utilities for FIT-FLIX RAG System."""

import logging
from typing import List, Dict, Any, Optional
import re
from ..config import Config


class TextSplitter:
    """Handles text splitting and chunking for vector storage."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the text splitter.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunked document dictionaries
        """
        chunked_documents = []
        
        for doc in documents:
            try:
                chunks = self.split_text(doc['content'])
                
                for i, chunk in enumerate(chunks):
                    chunked_doc = {
                        'content': chunk,
                        'metadata': {
                            **doc['metadata'],
                            'chunk_id': i,
                            'total_chunks': len(chunks),
                            'original_length': len(doc['content'])
                        }
                    }
                    chunked_documents.append(chunked_doc)
                    
            except Exception as e:
                self.logger.error(f"Failed to split document {doc.get('metadata', {}).get('source', 'unknown')}: {str(e)}")
                continue
        
        self.logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
        return chunked_documents
    
    def split_text(self, text: str, 
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk (uses config default if None)
            chunk_overlap: Overlap between chunks (uses config default if None)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        # Try to split by sentences first
        sentences = self._split_by_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + sentence
                else:
                    # Single sentence is too long, split by words
                    sentence_chunks = self._split_long_sentence(sentence, chunk_size, chunk_overlap)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
            else:
                current_chunk += sentence
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be enhanced with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s + ' ' for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a long sentence by words.
        
        Args:
            sentence: Sentence to split
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of sentence chunks
        """
        words = sentence.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            # Estimate current chunk length
            current_length = sum(len(w) + 1 for w in current_chunk)
            
            if current_length + len(word) > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_words = self._get_overlap_words(current_chunk, chunk_overlap)
                    current_chunk = overlap_words + [word]
                else:
                    # Single word is too long, just add it
                    chunks.append(word)
                    current_chunk = []
            else:
                current_chunk.append(word)
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk.
        
        Args:
            text: Source text
            overlap_size: Size of overlap
            
        Returns:
            Overlap text
        """
        if len(text) <= overlap_size:
            return text
        
        # Try to find a good breaking point (word boundary)
        overlap_text = text[-overlap_size:]
        space_idx = overlap_text.find(' ')
        
        if space_idx > 0:
            return overlap_text[space_idx + 1:]
        
        return overlap_text
    
    def _get_overlap_words(self, words: List[str], overlap_size: int) -> List[str]:
        """Get overlap words from the end of a word list.
        
        Args:
            words: List of words
            overlap_size: Size of overlap in characters
            
        Returns:
            List of overlap words
        """
        if not words:
            return []
        
        overlap_words = []
        current_length = 0
        
        # Start from the end and work backwards
        for word in reversed(words):
            if current_length + len(word) + 1 > overlap_size:
                break
            overlap_words.insert(0, word)
            current_length += len(word) + 1
        
        return overlap_words
    
    def split_by_sections(self, text: str) -> List[str]:
        """Split text by sections (headers, double newlines).
        
        Args:
            text: Text to split
            
        Returns:
            List of text sections
        """
        # Split by double newlines first
        sections = re.split(r'\n\s*\n', text)
        
        # Filter out very short sections
        sections = [s.strip() for s in sections if len(s.strip()) > 50]
        
        return sections
    
    def adaptive_split(self, text: str, max_chunk_size: Optional[int] = None) -> List[str]:
        """Adaptively split text based on structure.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of adaptively split chunks
        """
        max_chunk_size = max_chunk_size or self.chunk_size
        
        # First try splitting by sections
        sections = self.split_by_sections(text)
        
        chunks = []
        for section in sections:
            if len(section) <= max_chunk_size:
                chunks.append(section)
            else:
                # Section is too large, split further
                section_chunks = self.split_text(section, max_chunk_size)
                chunks.extend(section_chunks)
        
        return chunks
    
    def get_splitting_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about document splitting.
        
        Args:
            documents: List of chunked documents
            
        Returns:
            Splitting statistics
        """
        if not documents:
            return {"total_chunks": 0}
        
        chunk_lengths = [len(doc['content']) for doc in documents]
        sources = set()
        
        for doc in documents:
            source = doc.get('metadata', {}).get('source', 'unknown')
            sources.add(source)
        
        return {
            "total_chunks": len(documents),
            "unique_sources": len(sources),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }
    
    def merge_small_chunks(self, chunks: List[str], min_size: int = 100) -> List[str]:
        """Merge chunks that are smaller than minimum size.
        
        Args:
            chunks: List of text chunks
            min_size: Minimum chunk size
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < self.chunk_size:
                current_chunk += " " + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk.strip())
                current_chunk = chunk
        
        # Add final chunk
        if current_chunk:
            merged_chunks.append(current_chunk.strip())
        
        return merged_chunks
