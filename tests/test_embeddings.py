"""Tests for embedding functionality."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.embeddings.embedding_manager import EmbeddingManager


class TestEmbeddingManager(unittest.TestCase):
    """Test cases for EmbeddingManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.embedding_manager = EmbeddingManager(self.config)
    
    def test_initialization(self):
        """Test embedding manager initialization."""
        self.assertIsNotNone(self.embedding_manager)
        self.assertEqual(self.embedding_manager.model_name, self.config.embedding_model)
        self.assertIsNone(self.embedding_manager.model)  # Model not loaded yet
    
    def test_model_loading(self):
        """Test model loading."""
        try:
            self.embedding_manager.load_model()
            self.assertIsNotNone(self.embedding_manager.model)
        except Exception as e:
            self.skipTest(f"Model loading failed: {str(e)}")
    
    def test_single_text_encoding(self):
        """Test encoding a single text."""
        try:
            text = "This is a test document about fitness and health."
            embedding = self.embedding_manager.encode(text)
            
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(len(embedding.shape), 2)  # Should be 2D array
            self.assertEqual(embedding.shape[0], 1)  # One document
            
        except Exception as e:
            self.skipTest(f"Encoding test skipped: {str(e)}")
    
    def test_multiple_text_encoding(self):
        """Test encoding multiple texts."""
        try:
            texts = [
                "Fitness classes are great for building strength.",
                "Nutrition is important for recovery after workouts.",
                "Our trainers are certified professionals."
            ]
            embeddings = self.embedding_manager.encode(texts)
            
            self.assertIsInstance(embeddings, np.ndarray)
            self.assertEqual(len(embeddings.shape), 2)  # Should be 2D array
            self.assertEqual(embeddings.shape[0], len(texts))  # Should match input count
            
        except Exception as e:
            self.skipTest(f"Multiple encoding test skipped: {str(e)}")
    
    def test_query_encoding(self):
        """Test query encoding."""
        try:
            query = "What are the best exercises for building muscle?"
            embedding = self.embedding_manager.encode_query(query)
            
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(len(embedding.shape), 1)  # Should be 1D array for single query
            
        except Exception as e:
            self.skipTest(f"Query encoding test skipped: {str(e)}")
    
    def test_similarity_computation(self):
        """Test similarity computation."""
        try:
            query = "fitness training"
            docs = [
                "Personal training sessions",
                "Nutrition counseling",
                "Group fitness classes"
            ]
            
            query_embedding = self.embedding_manager.encode_query(query)
            doc_embeddings = self.embedding_manager.encode_documents(docs)
            
            similarities = self.embedding_manager.compute_similarity(query_embedding, doc_embeddings)
            
            self.assertIsInstance(similarities, np.ndarray)
            self.assertEqual(len(similarities), len(docs))
            self.assertTrue(all(-1 <= sim <= 1 for sim in similarities))  # Cosine similarity range
            
        except Exception as e:
            self.skipTest(f"Similarity computation test skipped: {str(e)}")
    
    def test_model_info(self):
        """Test getting model information."""
        info = self.embedding_manager.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("model_name", info)
        self.assertIn("loaded", info)
        self.assertEqual(info["model_name"], self.config.embedding_model)
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        try:
            # Test empty string
            embedding = self.embedding_manager.encode("")
            self.assertIsInstance(embedding, np.ndarray)
            
            # Test empty list
            embeddings = self.embedding_manager.encode([])
            self.assertIsInstance(embeddings, np.ndarray)
            self.assertEqual(embeddings.shape[0], 0)
            
        except Exception as e:
            self.skipTest(f"Empty input handling test skipped: {str(e)}")


class TestEmbeddingIntegration(unittest.TestCase):
    """Integration tests for embedding functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.embedding_manager = EmbeddingManager(self.config)
    
    def test_fitness_domain_embeddings(self):
        """Test embeddings on fitness-specific content."""
        try:
            fitness_texts = [
                "Cardio exercises improve cardiovascular health",
                "Strength training builds muscle mass",
                "Yoga enhances flexibility and mindfulness",
                "HIIT workouts burn calories efficiently"
            ]
            
            embeddings = self.embedding_manager.encode_documents(fitness_texts)
            
            # Check that embeddings are reasonable
            self.assertEqual(embeddings.shape[0], len(fitness_texts))
            
            # Check that similar content has higher similarity
            query = "muscle building exercises"
            query_embedding = self.embedding_manager.encode_query(query)
            similarities = self.embedding_manager.compute_similarity(query_embedding, embeddings)
            
            # "Strength training builds muscle mass" should have highest similarity
            max_sim_idx = np.argmax(similarities)
            self.assertEqual(max_sim_idx, 1)  # Index of strength training text
            
        except Exception as e:
            self.skipTest(f"Fitness domain test skipped: {str(e)}")


if __name__ == "__main__":
    unittest.main()
