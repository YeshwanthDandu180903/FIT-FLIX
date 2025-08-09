"""Streamlit web interface for FIT-FLIX RAG System."""

import sys
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any
import time

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.retrieval.retriever import DocumentRetriever
from src.generation.llm_manager import LLMManager
from src.utils.document_loader import DocumentLoader
from src.utils.text_splitter import TextSplitter


class FitFlixStreamlitApp:
    """Streamlit web application for FIT-FLIX RAG system."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.config = Config()
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="FIT-FLIX RAG System",
            page_icon="🏋️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.chat_history = []
            st.session_state.retriever = None
            st.session_state.llm_manager = None
            st.session_state.document_loader = None
    
    @st.cache_resource
    def initialize_system(_self):
        """Initialize the RAG system components (cached).
        
        Returns:
            Tuple of (retriever, llm_manager, document_loader, success, message)
        """
        try:
            with st.spinner("🚀 Initializing FIT-FLIX RAG System..."):
                # Initialize components
                document_loader = DocumentLoader(_self.config)
                retriever = DocumentRetriever(_self.config)
                llm_manager = LLMManager(_self.config)
                
                # Initialize retriever
                retriever.initialize()
                
                # Check if documents exist in vector store
                stats = retriever.get_retrieval_stats()
                if stats.get('document_count', 0) == 0:
                    # Load and process documents
                    st.info("📚 Loading documents into vector store...")
                    documents = document_loader.load_all_documents()
                    
                    if not documents:
                        return None, None, None, False, "❌ No documents found in knowledge base directory"
                    
                    # Split documents
                    text_splitter = TextSplitter(_self.config)
                    chunked_docs = text_splitter.split_documents(documents)
                    
                    # Add to vector store
                    doc_contents = [doc['content'] for doc in chunked_docs]
                    doc_metadata = [doc['metadata'] for doc in chunked_docs]
                    retriever.add_documents(doc_contents, doc_metadata)
                    
                    st.success(f"✅ Added {len(chunked_docs)} document chunks to vector store")
                else:
                    st.success(f"✅ Loaded existing vector store with {stats['document_count']} documents")
                
                return retriever, llm_manager, document_loader, True, "System initialized successfully!"
                
        except Exception as e:
            error_msg = f"❌ Failed to initialize system: {str(e)}"
            return None, None, None, False, error_msg
    
    def render_sidebar(self):
        """Render the sidebar with system information and controls."""
        st.sidebar.title("🏋️ FIT-FLIX")
        st.sidebar.markdown("### AI Fitness Assistant")
        
        # System status
        if st.session_state.initialized:
            st.sidebar.success("✅ System Ready")
            
            # System info
            try:
                stats = st.session_state.retriever.get_retrieval_stats()
                st.sidebar.info(f"📊 Documents: {stats.get('document_count', 'N/A')}")
                st.sidebar.info(f"🤖 Model: {self.config.llm_model}")
                st.sidebar.info(f"🔍 Embedding: {self.config.embedding_model.split('/')[-1]}")
            except:
                st.sidebar.warning("Could not load system stats")
        else:
            st.sidebar.warning("⏳ System Initializing...")
        
        # Sample questions
        st.sidebar.markdown("### 💡 Try These Questions")
        sample_questions = [
            "What types of fitness classes do you offer?",
            "How much does a monthly membership cost?",
            "What are the gym's operating hours?",
            "Can you recommend a post-workout nutrition plan?",
            "What qualifications do your trainers have?",
            "Do you have equipment for strength training?",
            "How can I join the FIT-FLIX community?",
            "What should I eat before a workout?"
        ]
        
        for question in sample_questions:
            if st.sidebar.button(question, key=f"sample_{hash(question)}"):
                st.session_state.current_question = question
                st.experimental_rerun()
        
        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    def render_main_content(self):
        """Render the main content area."""
        st.title("🏋️ FIT-FLIX RAG System")
        st.markdown("### Your AI-powered fitness and wellness assistant")
        
        # Initialize system if needed
        if not st.session_state.initialized:
            retriever, llm_manager, document_loader, success, message = self.initialize_system()
            
            if success:
                st.session_state.retriever = retriever
                st.session_state.llm_manager = llm_manager
                st.session_state.document_loader = document_loader
                st.session_state.initialized = True
                st.success(message)
                st.experimental_rerun()
            else:
                st.error(message)
                return
        
        # Chat interface
        self.render_chat_interface()
    
    def render_chat_interface(self):
        """Render the chat interface."""
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation")
            
            for i, (question, answer, metadata) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**🙋 You:** {question}")
                    st.markdown(f"**🤖 FIT-FLIX:** {answer}")
                    
                    if metadata:
                        with st.expander("📊 Response Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Retrieval Time", f"{metadata.get('retrieval_time', 0):.2f}s")
                            with col2:
                                st.metric("Generation Time", f"{metadata.get('generation_time', 0):.2f}s")
                            with col3:
                                st.metric("Sources Found", metadata.get('source_count', 0))
                            
                            if metadata.get('sources'):
                                st.write("**Sources:**", ", ".join(metadata['sources']))
                    
                    st.divider()
        
        # Input area
        with st.container():
            # Check for sample question
            default_question = ""
            if hasattr(st.session_state, 'current_question'):
                default_question = st.session_state.current_question
                delattr(st.session_state, 'current_question')
            
            user_input = st.text_input(
                "Ask me anything about fitness, nutrition, classes, or membership:",
                value=default_question,
                placeholder="Type your question here...",
                key="user_input"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("Send", type="primary")
            
            if ask_button and user_input:
                self.process_question(user_input)
                st.experimental_rerun()
    
    def process_question(self, question: str):
        """Process user question and generate response.
        
        Args:
            question: User's question
        """
        if not question.strip():
            return
        
        try:
            with st.spinner("🤔 Thinking..."):
                # Retrieve relevant documents
                start_time = time.time()
                retrieved_docs = st.session_state.retriever.retrieve(question)
                retrieval_time = time.time() - start_time
                
                if not retrieved_docs:
                    response = "I couldn't find relevant information in our knowledge base. Could you please rephrase your question or ask about our fitness classes, nutrition, trainers, facilities, or membership options?"
                    metadata = {
                        'retrieval_time': retrieval_time,
                        'generation_time': 0,
                        'source_count': 0,
                        'sources': []
                    }
                else:
                    # Generate response
                    generation_start = time.time()
                    response = st.session_state.llm_manager.generate_response(question, retrieved_docs)
                    generation_time = time.time() - generation_start
                    
                    # Prepare metadata
                    sources = list(set([doc['metadata'].get('source', 'Unknown') 
                                      for doc in retrieved_docs[:3]]))
                    metadata = {
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'source_count': len(retrieved_docs),
                        'sources': sources
                    }
                
                # Add to chat history
                st.session_state.chat_history.append((question, response, metadata))
                
        except Exception as e:
            error_response = f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try again or contact support if the problem persists."
            metadata = {'error': True}
            st.session_state.chat_history.append((question, error_response, metadata))
    
    def render_about_section(self):
        """Render the about section."""
        with st.expander("ℹ️ About FIT-FLIX RAG System"):
            st.markdown("""
            **FIT-FLIX RAG System** is an AI-powered fitness and wellness assistant that uses 
            Retrieval-Augmented Generation (RAG) to provide accurate, contextual answers to your 
            fitness-related questions.
            
            **Features:**
            - 🏋️ Comprehensive fitness knowledge base
            - 🥗 Nutrition guidance and meal planning
            - 👥 Information about trainers and classes
            - 🏢 Facility details and membership options
            - 🤖 AI-powered responses with source attribution
            
            **How it works:**
            1. You ask a question about fitness, nutrition, or our services
            2. The system searches our knowledge base for relevant information
            3. An AI model generates a personalized response based on the retrieved context
            4. You get accurate, helpful information with source references
            
            **Technology Stack:**
            - Vector Database: ChromaDB
            - Embeddings: Sentence Transformers
            - Language Model: Google Gemini
            - Interface: Streamlit
            """)
    
    def run(self):
        """Run the Streamlit application."""
        # Render components
        self.render_sidebar()
        self.render_main_content()
        self.render_about_section()
        
        # Add custom CSS
        st.markdown("""
        <style>
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        </style>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    try:
        app = FitFlixStreamlitApp()
        app.run()
        
    except Exception as e:
        st.error(f"❌ Failed to launch app: {str(e)}")


if __name__ == "__main__":
    main()
