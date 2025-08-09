# FIT-FLIX RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system for fitness and wellness knowledge, powered by advanced AI technologies.

## 🚀 Features

- **Intelligent Knowledge Retrieval**: Vector-based semantic search through fitness knowledge base
- **Multi-Modal LLM Support**: Compatible with Google Gemini, OpenAI GPT, and Hugging Face models
- **Interactive Web Interface**: Gradio and Streamlit applications for easy interaction
- **Scalable Architecture**: Modular design for easy maintenance and extension
- **Comprehensive Testing**: Unit tests for all components

## 📁 Project Structure

```
FIT-FLIX/
├── .env                           # Environment variables
├── .venv/                         # Virtual environment
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── data/                          # Data management
│   ├── knowledge_base/            # Raw knowledge documents
│   └── processed/                 # Processed/cleaned documents
│
├── vector_db/                     # Vector database storage
│   └── fitflix_chroma_db_gemini/
│
├── src/                           # Source code
│   ├── config.py                  # Configuration settings
│   ├── embeddings/                # Embedding-related code
│   ├── retrieval/                 # Retrieval logic
│   ├── generation/                # LLM generation
│   └── utils/                     # Utility functions
│
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── interfaces/                    # User interfaces
├── logs/                          # Application logs
└── app.py                        # Main application entry point
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd fitflix
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration

## 🚀 Quick Start

1. **Run the main application**:
   ```bash
   python app.py
   ```

2. **Launch Gradio interface**:
   ```bash
   python interfaces/gradio_app.py
   ```

3. **Launch Streamlit interface**:
   ```bash
   streamlit run interfaces/streamlit_app.py
   ```

## 📚 Usage

### Using the RAG System

```python
from src.config import Config
from src.retrieval.retriever import DocumentRetriever
from src.generation.llm_manager import LLMManager

# Initialize components
config = Config()
retriever = DocumentRetriever(config)
llm_manager = LLMManager(config)

# Query the system
question = "What are the benefits of strength training?"
relevant_docs = retriever.retrieve(question)
response = llm_manager.generate_response(question, relevant_docs)
print(response)
```

### Training Custom Embeddings

```python
from src.embeddings.embedding_manager import EmbeddingManager

embedding_manager = EmbeddingManager()
embedding_manager.train_custom_embeddings("data/knowledge_base/")
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_retrieval.py
pytest tests/test_generation.py
```

## 📊 Notebooks

- `notebooks/rag.ipynb` - Main RAG system demonstration
- `notebooks/data_exploration.ipynb` - Knowledge base analysis
- `notebooks/evaluation.ipynb` - Performance evaluation

## 🔧 Configuration

Key configuration options in `.env`:

- **API Keys**: OpenAI, Google AI, Hugging Face tokens
- **Model Settings**: Embedding models, LLM models, generation parameters
- **Database**: ChromaDB configuration
- **Retrieval**: Top-K results, chunk sizes, overlap settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues, please:
1. Check the FAQ in the knowledge base
2. Search existing issues
3. Create a new issue with detailed description

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced evaluation metrics
- [ ] Custom fine-tuning pipeline
- [ ] API documentation with Swagger
- [ ] Docker containerization
- [ ] Cloud deployment guides
