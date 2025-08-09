"""Streamlit web interface for FIT-FLIX RAG System."""

import sys
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.retrieval.retriever import DocumentRetriever
from src.generation.llm_manager import LLMManager
from src.utils.document_loader import DocumentLoader
from src.embeddings.embedding_manager import EmbeddingManager


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
                
                # Initialize retriever (this will create collection if it doesn't exist)
                retriever.initialize()
                
                # Check if documents exist in vector store
                stats = retriever.get_retrieval_stats()
                document_count = stats.get('document_count', 0)
                
                if document_count == 0:
                    # Load and process documents
                    st.info("📚 Loading documents into vector store...")
                    
                    # Load documents from knowledge base
                    knowledge_base_path = _self.config.knowledge_base_dir
                    if not knowledge_base_path.exists():
                        return None, None, None, False, f"❌ Knowledge base directory not found: {knowledge_base_path}"
                    
                    documents = document_loader.load_all_documents()
                    
                    if not documents:
                        return None, None, None, False, "❌ No documents found in knowledge base directory"
                    
                    # Split documents
                    from src.utils.text_splitter import TextSplitter
                    text_splitter = TextSplitter(_self.config)
                    chunked_docs = text_splitter.split_documents(documents)
                    
                    if not chunked_docs:
                        return None, None, None, False, "❌ No document chunks created"
                    
                    # Add to vector store
                    doc_contents = [doc['content'] for doc in chunked_docs]
                    doc_metadata = [doc['metadata'] for doc in chunked_docs]
                    doc_ids = [f"chunk_{i}" for i in range(len(chunked_docs))]
                    
                    retriever.add_documents(doc_contents, doc_metadata, doc_ids)
                    
                    st.success(f"✅ Added {len(chunked_docs)} document chunks to vector store")
                else:
                    st.success(f"✅ Loaded existing vector store with {document_count} documents")
                
                return retriever, llm_manager, document_loader, True, "System initialized successfully!"
                
        except Exception as e:
            error_msg = f"❌ Failed to initialize system: {str(e)}"
            st.error(error_msg)
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
                st.sidebar.metric("📚 Documents", stats.get('document_count', 0))
                st.sidebar.info(f"🤖 Model: {self.config.llm_model}")
            except:
                st.sidebar.warning("⚠️ Stats unavailable")
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
                self.process_question(question)
        
        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()  # Changed from st.experimental_rerun()
    
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
                st.rerun()
            else:
                st.error(message)
                return
        
        # Chat interface
        self.render_chat_interface()
    
    def render_chat_interface(self):
        """Render the chat interface."""
        # Display chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**🔍 You:** {chat['question']}")
                    st.markdown(f"**🤖 FIT-FLIX:** {chat['answer']}")
                    st.markdown("---")
        
        # Input area
        with st.form(key="question_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                question = st.text_input("Ask your fitness question:", key="user_question")
            
            with col2:
                submit_button = st.form_submit_button("Send", type="primary")
            
            if submit_button and question.strip():
                self.process_question(question)
                st.rerun()
    
    def process_question(self, question: str):
        """Process user question and generate response.
        
        Args:
            question: User's question
        """
        if not question.strip():
            st.warning("Please enter a question.")
            return
        
        try:
            with st.spinner("🤔 Thinking..."):
                # Retrieve relevant documents
                relevant_docs = st.session_state.retriever.retrieve(question)
                
                # Generate response
                response = st.session_state.llm_manager.generate_response(question, relevant_docs)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response
                })
                
                 # Clear the input box
                #st.session_state.user_input = ""  # Add this line
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
    
    def render_about_section(self):
        """Render the about section."""
        with st.expander("ℹ️ About FIT-FLIX RAG System"):
            st.markdown("""
            **FIT-FLIX RAG System** is an AI-powered fitness and wellness assistant that provides:
            
            - 🏋️ **Fitness guidance** and workout recommendations
            - 🥗 **Nutrition advice** and meal planning
            - 📅 **Class schedules** and facility information  
            - 👥 **Community** support and trainer details
            - 💡 **Expert tips** for achieving your fitness goals
            
            Built with advanced RAG (Retrieval-Augmented Generation) technology for accurate, 
            context-aware responses based on our comprehensive fitness knowledge base.
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