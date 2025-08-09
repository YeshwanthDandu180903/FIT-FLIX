"""Main application entry point for FIT-FLIX RAG System."""

import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.retrieval.retriever import DocumentRetriever
from src.generation.llm_manager import LLMManager
from src.utils.document_loader import DocumentLoader
from src.embeddings.embedding_manager import EmbeddingManager


def initialize_system():
    """Initialize the RAG system components."""
    print("🚀 Initializing FIT-FLIX RAG System...")
    
    # Load configuration
    config = Config()
    
    # Initialize components
    document_loader = DocumentLoader(config)
    embedding_manager = EmbeddingManager(config)
    retriever = DocumentRetriever(config)
    llm_manager = LLMManager(config)
    
    return config, document_loader, embedding_manager, retriever, llm_manager


def main():
    """Main application function."""
    try:
        # Initialize system
        config, document_loader, embedding_manager, retriever, llm_manager = initialize_system()
        
        print("✅ System initialized successfully!")
        print(f"📊 Vector database path: {config.chroma_db_path}")
        print(f"🤖 Using LLM model: {config.llm_model}")
        print(f"🔍 Using embedding model: {config.embedding_model}")
        
        # Interactive loop
        print("\n💬 Welcome to FIT-FLIX RAG System!")
        print("Ask questions about fitness, nutrition, classes, and more.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("🔍 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Stay fit!")
                    break
                
                if not question:
                    continue
                
                print("🤔 Thinking...")
                
                # Retrieve relevant documents
                relevant_docs = retriever.retrieve(question)
                
                # Generate response
                response = llm_manager.generate_response(question, relevant_docs)
                
                print(f"\n🤖 Answer: {response}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Stay fit!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
