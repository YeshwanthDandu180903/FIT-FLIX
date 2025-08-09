"""Document loading utilities for FIT-FLIX RAG System."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re
from ..config import Config


class DocumentLoader:
    """Handles loading and preprocessing of documents."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the document loader.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
    
    def load_markdown_files(self, directory: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load markdown files from a directory.
        
        Args:
            directory: Directory containing markdown files
            
        Returns:
            List of document dictionaries
        """
        directory = Path(directory)
        documents = []
        
        try:
            for md_file in directory.glob("*.md"):
                content = self._load_markdown_file(md_file)
                if content:
                    doc = {
                        "content": content,
                        "metadata": {
                            "source": md_file.name,
                            "file_path": str(md_file),
                            "file_type": "markdown",
                            "category": self._infer_category(md_file.name)
                        }
                    }
                    documents.append(doc)
            
            self.logger.info(f"Loaded {len(documents)} markdown documents from {directory}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load markdown files: {str(e)}")
            raise
    
    def _load_markdown_file(self, file_path: Path) -> Optional[str]:
        """Load content from a single markdown file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            File content or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic markdown processing
            content = self._clean_markdown(content)
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {str(e)}")
            return None
    
    def _clean_markdown(self, content: str) -> str:
        """Clean and preprocess markdown content.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        
        # Remove markdown headers (keep content)
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        
        # Remove markdown links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove markdown emphasis markers
        content = re.sub(r'\*\*([^\*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^\*]+)\*', r'\1', content)
        
        # Remove markdown code blocks
        content = re.sub(r'```[^`]*```', '', content)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        return content
    
    def _infer_category(self, filename: str) -> str:
        """Infer document category from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Inferred category
        """
        filename_lower = filename.lower()
        
        if 'class' in filename_lower:
            return 'classes'
        elif 'trainer' in filename_lower:
            return 'trainers'
        elif 'nutrition' in filename_lower:
            return 'nutrition'
        elif 'membership' in filename_lower:
            return 'membership'
        elif 'facilit' in filename_lower:
            return 'facilities'
        elif 'faq' in filename_lower:
            return 'faq'
        elif 'community' in filename_lower:
            return 'community'
        elif 'contact' in filename_lower:
            return 'contact'
        elif 'about' in filename_lower:
            return 'about'
        else:
            return 'general'
    
    def load_text_files(self, directory: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load text files from a directory.
        
        Args:
            directory: Directory containing text files
            
        Returns:
            List of document dictionaries
        """
        directory = Path(directory)
        documents = []
        
        try:
            for txt_file in directory.glob("*.txt"):
                content = self._load_text_file(txt_file)
                if content:
                    doc = {
                        "content": content,
                        "metadata": {
                            "source": txt_file.name,
                            "file_path": str(txt_file),
                            "file_type": "text",
                            "category": self._infer_category(txt_file.name)
                        }
                    }
                    documents.append(doc)
            
            self.logger.info(f"Loaded {len(documents)} text documents from {directory}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load text files: {str(e)}")
            raise
    
    def _load_text_file(self, file_path: Path) -> Optional[str]:
        """Load content from a single text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic text cleaning
            content = content.strip()
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {str(e)}")
            return None
    
    def load_all_documents(self, directory: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Load all supported document types from a directory.
        
        Args:
            directory: Directory to load from (uses knowledge_base_dir if None)
            
        Returns:
            List of all document dictionaries
        """
        if directory is None:
            directory = self.config.knowledge_base_dir
        
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory}")
            return []
        
        documents = []
        
        # Load markdown files
        try:
            md_docs = self.load_markdown_files(directory)
            documents.extend(md_docs)
        except Exception as e:
            self.logger.warning(f"Failed to load markdown files: {str(e)}")
        
        # Load text files
        try:
            txt_docs = self.load_text_files(directory)
            documents.extend(txt_docs)
        except Exception as e:
            self.logger.warning(f"Failed to load text files: {str(e)}")
        
        self.logger.info(f"Loaded total {len(documents)} documents from {directory}")
        return documents
    
    def validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of valid documents
        """
        valid_documents = []
        
        for doc in documents:
            if self._is_valid_document(doc):
                valid_documents.append(doc)
            else:
                self.logger.warning(f"Invalid document filtered out: {doc.get('metadata', {}).get('source', 'unknown')}")
        
        self.logger.info(f"Validated {len(valid_documents)}/{len(documents)} documents")
        return valid_documents
    
    def _is_valid_document(self, document: Dict[str, Any]) -> bool:
        """Check if a document is valid.
        
        Args:
            document: Document dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if 'content' not in document or 'metadata' not in document:
            return False
        
        # Check content length
        content = document['content']
        if not content or len(content.strip()) < 10:
            return False
        
        # Check metadata
        metadata = document['metadata']
        if not isinstance(metadata, dict) or 'source' not in metadata:
            return False
        
        return True
    
    def get_document_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the loaded documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {"total_documents": 0}
        
        # Count by category
        categories = {}
        file_types = {}
        total_chars = 0
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            category = metadata.get('category', 'unknown')
            file_type = metadata.get('file_type', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_chars += len(doc.get('content', ''))
        
        return {
            "total_documents": len(documents),
            "categories": categories,
            "file_types": file_types,
            "total_characters": total_chars,
            "average_document_length": total_chars / len(documents) if documents else 0
        }
