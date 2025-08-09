"""LLM management for FIT-FLIX RAG System."""

import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from ..config import Config


class LLMManager:
    """Manages Large Language Model interactions."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the LLM manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.model_name = self.config.llm_model
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> None:
        """Initialize the LLM model."""
        try:
            if "gemini" in self.model_name.lower():
                self._initialize_gemini()
            elif "gpt" in self.model_name.lower():
                self._initialize_openai()
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            self.logger.info(f"LLM model initialized: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_gemini(self) -> None:
        """Initialize Google Gemini model."""
        if not self.config.google_api_key:
            raise ValueError("Google API key not provided")
        
        genai.configure(api_key=self.config.google_api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def _initialize_openai(self) -> None:
        """Initialize OpenAI model."""
        # This would initialize OpenAI client
        # For now, it's a placeholder
        raise NotImplementedError("OpenAI integration not implemented yet")
    
    def generate_response(self, query: str, 
                         context_documents: List[Dict[str, Any]],
                         system_prompt: Optional[str] = None) -> str:
        """Generate a response using RAG.
        
        Args:
            query: User query
            context_documents: Retrieved documents for context
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        if self.model is None:
            self.initialize()
        
        try:
            # Build context from retrieved documents
            context = self._build_context(context_documents)
            
            # Create prompt
            prompt = self._create_rag_prompt(query, context, system_prompt)
            
            # Generate response
            if "gemini" in self.model_name.lower():
                response = self._generate_gemini_response(prompt)
            else:
                raise ValueError(f"Unsupported model for generation: {self.model_name}")
            
            self.logger.info(f"Generated response for query: {query[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {str(e)}")
            raise
    
    def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini model."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return response.text
            
        except Exception as e:
            self.logger.error(f"Gemini generation failed: {str(e)}")
            raise
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            source = doc.get('metadata', {}).get('source', 'Unknown')
            similarity = doc.get('similarity', 0)
            
            context_parts.append(
                f"Document {i} (Relevance: {similarity:.2f}, Source: {source}):\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_rag_prompt(self, query: str, context: str, 
                          system_prompt: Optional[str] = None) -> str:
        """Create RAG prompt combining query and context.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            
        Returns:
            Complete RAG prompt
        """
        default_system_prompt = """You are FIT-FLIX, an AI assistant specialized in fitness, wellness, and health.
You provide helpful, accurate, and encouraging responses based on the provided context.

Guidelines:
- Use the provided context to answer questions
- If the context doesn't contain relevant information, say so clearly
- Provide practical, actionable advice when appropriate
- Be encouraging and supportive
- Focus on safety and proper form for exercises
- Recommend consulting professionals when necessary"""

        system = system_prompt or default_system_prompt
        
        prompt = f"""{system}

Context Information:
{context}

User Question: {query}

Response:"""
        
        return prompt
    
    def generate_summary(self, text: str, max_length: Optional[int] = None) -> str:
        """Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        if self.model is None:
            self.initialize()
        
        max_len = max_length or self.max_tokens // 2
        
        prompt = f"""Please provide a concise summary of the following text in no more than {max_len} words:

{text}

Summary:"""
        
        try:
            if "gemini" in self.model_name.lower():
                response = self.model.generate_content(prompt)
                return response.text
            else:
                raise ValueError(f"Unsupported model for summarization: {self.model_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            raise
    
    def evaluate_response_quality(self, query: str, response: str, 
                                 context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality of a generated response.
        
        Args:
            query: Original query
            response: Generated response
            context_documents: Context used for generation
            
        Returns:
            Evaluation metrics
        """
        # This is a placeholder for response evaluation
        # In a full implementation, this would use various metrics
        return {
            "length": len(response),
            "context_relevance": len(context_documents) > 0,
            "has_sources": any("source" in doc.get("metadata", {}) for doc in context_documents)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "initialized": self.model is not None
        }
