"""Gradio web interface for FIT-FLIX RAG System."""

import os
import sys
from pathlib import Path
import gradio as gr
from typing import List, Tuple, Optional
import tim

e
# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config
from src.retrieval.retriever import DocumentRetriever
from src.generation.llm_manager import LLMManager
from src.utils.document_loader import DocumentLoader
from src.utils.text_splitter import TextSplitter


class FitFlixGradioApp:
    """Gradio web application for FIT-FLIX RAG system."""
    
    def __init__(self):
        """Initialize the Gradio application."""
        self.config = Config()
        self.document_loader = None
        self.retriever = None
        self.llm_manager = None
        self.is_initialized = False
        self.initialization_error = None
    
    def initialize_system(self) -> Tuple[bool, str]:
        """Initialize the RAG system components.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            print("🚀 Initializing FIT-FLIX RAG System...")
            
            # Initialize components
            self.document_loader = DocumentLoader(self.config)
            self.retriever = DocumentRetriever(self.config)
            self.llm_manager = LLMManager(self.config)
            
            # Initialize retriever
            self.retriever.initialize()
            
            # Check if documents exist in vector store
            stats = self.retriever.get_retrieval_stats()
            if stats.get('document_count', 0) == 0:
                # Load and process documents
                print("📚 Loading documents into vector store...")
                documents = self.document_loader.load_all_documents()
                
                if not documents:
                    return False, "❌ No documents found in knowledge base directory"
                
                # Split documents
                text_splitter = TextSplitter(self.config)
                chunked_docs = text_splitter.split_documents(documents)
                
                # Add to vector store
                doc_contents = [doc['content'] for doc in chunked_docs]
                doc_metadata = [doc['metadata'] for doc in chunked_docs]
                self.retriever.add_documents(doc_contents, doc_metadata)
                
                print(f"✅ Added {len(chunked_docs)} document chunks to vector store")
            else:
                print(f"✅ Loaded existing vector store with {stats['document_count']} documents")
            
            self.is_initialized = True
            return True, f"✅ System initialized successfully! Ready to answer questions about fitness, nutrition, and wellness."
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize system: {str(e)}"
            print(error_msg)
            self.initialization_error = error_msg
            return False, error_msg
    
    def chat_with_rag(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Process chat message with RAG system.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not message.strip():
            return "", history
        
        # Check if system is initialized
        if not self.is_initialized:
            success, init_message = self.initialize_system()
            if not success:
                history.append([message, init_message])
                return "", history
        
        try:
            # Retrieve relevant documents
            print(f"🔍 Processing query: {message[:50]}...")
            start_time = time.time()
            
            retrieved_docs = self.retriever.retrieve(message)
            retrieval_time = time.time() - start_time
            
            if not retrieved_docs:
                response = "I couldn't find relevant information in our knowledge base. Could you please rephrase your question or ask about our fitness classes, nutrition, trainers, facilities, or membership options?"
                history.append([message, response])
                return "", history
            
            # Generate response
            generation_start = time.time()
            response = self.llm_manager.generate_response(message, retrieved_docs)
            generation_time = time.time() - generation_start
            
            # Add source information
            sources = list(set([doc['metadata'].get('source', 'Unknown') 
                              for doc in retrieved_docs[:3]]))
            source_info = f"\n\n*Sources: {', '.join(sources)} | Retrieved in {retrieval_time:.2f}s, Generated in {generation_time:.2f}s*"
            
            final_response = response + source_info
            history.append([message, final_response])
            
            print(f"✅ Response generated in {retrieval_time + generation_time:.2f}s")
            return "", history
            
        except Exception as e:
            error_response = f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try again or contact support if the problem persists."
            history.append([message, error_response])
            return "", history
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions for the interface.
        
        Returns:
            List of sample questions
        """
        return [
            "What types of fitness classes do you offer?",
            "How much does a monthly membership cost?",
            "What are the gym's operating hours?",
            "Can you recommend a good post-workout nutrition plan?",
            "What qualifications do your personal trainers have?",
            "Do you have equipment for strength training?",
            "How can I join the FIT-FLIX community?",
            "What should I eat before a workout?"
        ]
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="FIT-FLIX RAG System",
            theme=gr.themes.Soft(),
            css="""
            .header { text-align: center; margin-bottom: 2rem; }
            .chat-container { max-height: 600px; }
            .sample-questions { margin: 1rem 0; }
            """
        ) as interface:
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="header">
                    <h1>🏋️ FIT-FLIX RAG System</h1>
                    <p>Your AI-powered fitness and wellness assistant</p>
                    <p>Ask questions about our classes, nutrition, trainers, facilities, and more!</p>
                </div>
                """)
            
            # Main chat interface
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        value=[],
                        height=500,
                        show_copy_button=True,
                        bubble_full_width=False,
                        container=True,
                        elem_classes=["chat-container"]
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me anything about fitness, nutrition, classes, or membership...",
                            show_label=False,
                            scale=4,
                            container=False
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    gr.HTML("<h3>💡 Sample Questions</h3>")
                    
                    sample_questions = self.get_sample_questions()
                    for question in sample_questions:
                        question_btn = gr.Button(
                            question,
                            variant="secondary",
                            size="sm",
                            elem_classes=["sample-questions"]
                        )
                        question_btn.click(
                            fn=lambda q=question: q,
                            outputs=msg
                        )
            
            # System status
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f0f0f0; border-radius: 8px;">
                        <p><strong>System Status:</strong> Ready to help with your fitness journey! 💪</p>
                        <p><em>Powered by advanced AI and comprehensive fitness knowledge base</em></p>
                    </div>
                    """)
            
            # Event handlers
            def respond(message, history):
                return self.chat_with_rag(message, history)
            
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            
            # Clear button
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio application.
        
        Args:
            **kwargs: Gradio launch arguments
        """
        interface = self.create_interface()
        
        # Default launch settings
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": self.config.app_port,
            "share": False,
            "show_error": True,
            "quiet": False
        }
        launch_kwargs.update(kwargs)
        
        print(f"🚀 Launching FIT-FLIX Gradio App on port {launch_kwargs['server_port']}...")
        interface.launch(**launch_kwargs)


def main():
    """Main function to run the Gradio app."""
    try:
        app = FitFlixGradioApp()
        app.launch()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down FIT-FLIX Gradio App...")
    except Exception as e:
        print(f"❌ Failed to launch app: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
