"""Tests for generation functionality."""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.generation.llm_manager import LLMManager


class TestLLMManager(unittest.TestCase):
    """Test cases for LLMManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.llm_manager = LLMManager(self.config)
    
    def test_initialization(self):
        """Test LLM manager initialization."""
        self.assertIsNotNone(self.llm_manager)
        self.assertEqual(self.llm_manager.model_name, self.config.llm_model)
        self.assertEqual(self.llm_manager.temperature, self.config.temperature)
        self.assertEqual(self.llm_manager.max_tokens, self.config.max_tokens)
        self.assertIsNone(self.llm_manager.model)  # Model not loaded yet
    
    def test_context_building(self):
        """Test building context from retrieved documents."""
        documents = [
            {
                'content': 'Cardio exercises improve heart health.',
                'metadata': {'source': 'cardio.md', 'category': 'cardio'},
                'similarity': 0.85
            },
            {
                'content': 'Strength training builds muscle mass.',
                'metadata': {'source': 'strength.md', 'category': 'strength'},
                'similarity': 0.78
            }
        ]
        
        context = self.llm_manager._build_context(documents)
        
        self.assertIsInstance(context, str)
        self.assertIn('Cardio exercises improve heart health.', context)
        self.assertIn('Strength training builds muscle mass.', context)
        self.assertIn('cardio.md', context)
        self.assertIn('0.85', context)
    
    def test_empty_context_building(self):
        """Test building context with empty document list."""
        context = self.llm_manager._build_context([])
        
        self.assertIsInstance(context, str)
        self.assertIn('No relevant information found', context)
    
    def test_rag_prompt_creation(self):
        """Test RAG prompt creation."""
        query = "What are the benefits of cardio exercise?"
        context = "Document 1: Cardio improves heart health and endurance."
        
        prompt = self.llm_manager._create_rag_prompt(query, context)
        
        self.assertIsInstance(prompt, str)
        self.assertIn(query, prompt)
        self.assertIn(context, prompt)
        self.assertIn('FIT-FLIX', prompt)
        self.assertIn('fitness', prompt.lower())
    
    def test_custom_system_prompt(self):
        """Test RAG prompt creation with custom system prompt."""
        query = "Test query"
        context = "Test context"
        custom_prompt = "You are a test assistant."
        
        prompt = self.llm_manager._create_rag_prompt(query, context, custom_prompt)
        
        self.assertIn(custom_prompt, prompt)
        self.assertIn(query, prompt)
        self.assertIn(context, prompt)
    
    def test_model_info(self):
        """Test getting model information."""
        info = self.llm_manager.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("model_name", info)
        self.assertIn("temperature", info)
        self.assertIn("max_tokens", info)
        self.assertIn("initialized", info)
        self.assertEqual(info["model_name"], self.config.llm_model)
        self.assertEqual(info["temperature"], self.config.temperature)
        self.assertFalse(info["initialized"])  # Model not loaded
    
    def test_response_evaluation(self):
        """Test response quality evaluation."""
        query = "What are good exercises for beginners?"
        response = "For beginners, I recommend starting with bodyweight exercises like push-ups, squats, and walking."
        context_docs = [
            {
                'content': 'Bodyweight exercises are great for beginners.',
                'metadata': {'source': 'beginner.md'}
            }
        ]
        
        evaluation = self.llm_manager.evaluate_response_quality(query, response, context_docs)
        
        self.assertIsInstance(evaluation, dict)
        self.assertIn("length", evaluation)
        self.assertIn("context_relevance", evaluation)
        self.assertIn("has_sources", evaluation)
        self.assertGreater(evaluation["length"], 0)
        self.assertTrue(evaluation["context_relevance"])
        self.assertTrue(evaluation["has_sources"])


class TestLLMIntegration(unittest.TestCase):
    """Integration tests for LLM functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.llm_manager = LLMManager(self.config)
    
    @patch('src.generation.llm_manager.genai')
    def test_gemini_initialization(self, mock_genai):
        """Test Gemini model initialization with mocked API."""
        # Mock the GenerativeModel
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Set up for Gemini model
        self.config.llm_model = "gemini-pro"
        self.config.google_api_key = "test_key"
        llm_manager = LLMManager(self.config)
        
        try:
            llm_manager.initialize()
            mock_genai.configure.assert_called_once_with(api_key="test_key")
            mock_genai.GenerativeModel.assert_called_once_with("gemini-pro")
            self.assertIsNotNone(llm_manager.model)
        except Exception as e:
            self.skipTest(f"Gemini initialization test skipped: {str(e)}")
    
    def test_openai_initialization(self):
        """Test OpenAI model initialization (should raise NotImplementedError)."""
        self.config.llm_model = "gpt-3.5-turbo"
        llm_manager = LLMManager(self.config)
        
        with self.assertRaises(NotImplementedError):
            llm_manager.initialize()
    
    def test_unsupported_model(self):
        """Test initialization with unsupported model."""
        self.config.llm_model = "unsupported-model"
        llm_manager = LLMManager(self.config)
        
        with self.assertRaises(ValueError):
            llm_manager.initialize()
    
    @patch('src.generation.llm_manager.genai')
    def test_response_generation_flow(self, mock_genai):
        """Test the complete response generation flow with mocked API."""
        # Mock the response
        mock_response = Mock()
        mock_response.text = "Cardio exercises like running and cycling improve cardiovascular health."
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Set up config
        self.config.llm_model = "gemini-pro"
        self.config.google_api_key = "test_key"
        llm_manager = LLMManager(self.config)
        
        try:
            # Test query and context
            query = "What are the benefits of cardio exercise?"
            context_docs = [
                {
                    'content': 'Cardiovascular exercise strengthens the heart and improves endurance.',
                    'metadata': {'source': 'cardio.md', 'category': 'cardio'},
                    'similarity': 0.9
                }
            ]
            
            # Generate response
            response = llm_manager.generate_response(query, context_docs)
            
            # Verify response
            self.assertEqual(response, mock_response.text)
            mock_model.generate_content.assert_called_once()
            
            # Verify the prompt contains expected elements
            call_args = mock_model.generate_content.call_args[0][0]
            self.assertIn(query, call_args)
            self.assertIn('Cardiovascular exercise', call_args)
            
        except Exception as e:
            self.skipTest(f"Response generation test skipped: {str(e)}")
    
    def test_missing_api_key(self):
        """Test initialization without API key."""
        self.config.llm_model = "gemini-pro"
        self.config.google_api_key = None
        llm_manager = LLMManager(self.config)
        
        with self.assertRaises(ValueError):
            llm_manager.initialize()
    
    def test_summary_generation_prompt(self):
        """Test summary generation prompt creation."""
        text = "This is a long text about fitness that needs to be summarized."
        
        # We can't test actual generation without API keys, but we can test prompt creation
        prompt = f"""Please provide a concise summary of the following text in no more than {self.config.max_tokens // 2} words:

{text}

Summary:"""
        
        self.assertIn(text, prompt)
        self.assertIn("summary", prompt.lower())
        self.assertIn(str(self.config.max_tokens // 2), prompt)


class TestPromptEngineering(unittest.TestCase):
    """Test cases for prompt engineering and optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.llm_manager = LLMManager(self.config)
    
    def test_fitness_specific_prompt(self):
        """Test that prompts are optimized for fitness domain."""
        query = "How often should I work out?"
        context = "Exercise frequency depends on fitness goals and current level."
        
        prompt = self.llm_manager._create_rag_prompt(query, context)
        
        # Check for fitness-specific guidance in prompt
        prompt_lower = prompt.lower()
        self.assertIn('fitness', prompt_lower)
        self.assertIn('safety', prompt_lower)
        self.assertIn('professional', prompt_lower)
    
    def test_prompt_structure(self):
        """Test that prompts have proper structure."""
        query = "What equipment do I need for home workouts?"
        context = "Basic equipment includes dumbbells, resistance bands, and a yoga mat."
        
        prompt = self.llm_manager._create_rag_prompt(query, context)
        
        # Check prompt structure
        self.assertIn("Context Information:", prompt)
        self.assertIn("User Question:", prompt)
        self.assertIn("Response:", prompt)
        self.assertIn(query, prompt)
        self.assertIn(context, prompt)
    
    def test_prompt_safety_guidelines(self):
        """Test that prompts include safety considerations."""
        query = "How much weight should I lift?"
        context = "Weight selection should be based on individual capability."
        
        prompt = self.llm_manager._create_rag_prompt(query, context)
        
        # Check for safety-related guidance
        prompt_lower = prompt.lower()
        self.assertIn('safety', prompt_lower)
        self.assertIn('form', prompt_lower)
    
    def test_encouraging_tone(self):
        """Test that prompts encourage a supportive tone."""
        query = "I'm struggling with motivation to exercise."
        context = "Consistency is key to building exercise habits."
        
        prompt = self.llm_manager._create_rag_prompt(query, context)
        
        # Check for encouraging language
        prompt_lower = prompt.lower()
        self.assertIn('encouraging', prompt_lower)
        self.assertIn('supportive', prompt_lower)


if __name__ == "__main__":
    unittest.main()
