# app.py (This is what you would save and run with `streamlit run app.py`)

# imports
import os
import streamlit as st # Import Streamlit
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
import shutil

# --- Setup for RAG Chain (all this should be executed once when the app starts) ---

# Load environment variables
load_dotenv(override=True)

# Initialize LLM
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7) # Use your confirmed model name

# Data Loading and Chunking
knowledge_base_path = "knowledge-base"
markdown_file_glob = "**/*.md"
loader = DirectoryLoader(
    knowledge_base_path, glob=markdown_file_glob, loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

headers_to_split_on = [("#", "Header1"), ("##", "Header2")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = []
for doc in documents:
    split_docs = markdown_splitter.split_text(doc.page_content)
    for s_doc in split_docs:
        s_doc.metadata = {**doc.metadata, **s_doc.metadata}
    chunks.extend(split_docs)

# Initialize Embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create/Load Vector Store
db_persist_dir = "fitflix_chroma_db_gemini"
# Only create/load if it doesn't exist to save time on reruns
if not os.path.exists(db_persist_dir) or not os.listdir(db_persist_dir): # Check if directory is empty or doesn't exist
    print(f"Creating new vectorstore at '{db_persist_dir}'...")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=gemini_embeddings, persist_directory=db_persist_dir)
    vectorstore.persist()
else:
    print(f"Loading existing vectorstore from '{db_persist_dir}'...")
    vectorstore = Chroma(persist_directory=db_persist_dir, embedding_function=gemini_embeddings)

# Create Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define Prompt Template
prompt_template = """
You are an AI assistant specialized in information about Fitflix entities.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise, professional, and directly address the question.

Context:
{context}

Question: {question}
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Set up Conversational Memory (this is what LangChain's chain uses internally)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the Conversational Retrieval QA Chain
# This is your core RAG engine
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=gemini_llm,
    retriever=retriever,
    memory=memory, # This memory object is key
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# --- Streamlit UI and Chat Logic ---

st.title("Fitflix RAG Chatbot")
st.write("Ask me questions about Fitflix from the knowledge base.")

# Initialize chat history in Streamlit's session state
# This is for DISPLAYING the conversation in the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to chat history (for display)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response from the RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the qa_chain's invoke method
            # The qa_chain uses its internal 'memory' object to maintain context
            result = qa_chain.invoke({"question": prompt})
            full_response = result["answer"]
            st.markdown(full_response)
    # Add assistant response to chat history (for display)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Optional: Clear chat history button
if st.button("Clear Chat"):
    st.session_state.messages = []
    # Also clear LangChain's internal memory
    memory.clear()
    st.rerun() # Use st.rerun() instead of st.experimental_rerun() for newer Streamlit