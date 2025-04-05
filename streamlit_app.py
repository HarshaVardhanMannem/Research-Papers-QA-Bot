"""Streamlit app for the Research Papers QA Bot."""
import streamlit as st
from langchain.schema.output_parser import StrOutputParser
# Fix the import for RunnableAssign
from langchain.schema.runnable import RunnablePassthrough
import operator
from operator import itemgetter

# Import NVIDIA chat module
from langchain_nvidia_ai_endpoints import ChatNVIDIA


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
def get_api_key(key_name="NVIDIA_API_KEY"):
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(
            f"{key_name} not found in environment variables. "
            f"Please add it to your .env file: {key_name}=your-api-key"
        )
    return api_key

# Set API key from environment
os.environ["NVIDIA_API_KEY"] = get_api_key("NVIDIA_API_KEY")


from src.data.document_loader import (
    load_arxiv_documents, 
    preprocess_documents, 
    create_document_chunks,
    create_metadata_chunks
)
from src.embedding.embeddings import get_embedder
from src.retrieval.vector_store import (
    create_default_faiss,
    create_vector_stores, 
    aggregate_vector_stores,
    reorder_documents,
    docs_to_string
)
from src.prompts.chat_prompts import create_chat_prompt
from src.utils.helpers import print_runnable, save_memory_and_get_output
from config.settings import LLM_MODEL

# Set page config
st.set_page_config(
    page_title="Research Papers QA Bot",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def initialize_resources():
    """Initialize and cache all resources for the QA system."""
    # Load and process documents
    docs = load_arxiv_documents()
    docs = preprocess_documents(docs)
    docs_chunks = create_document_chunks(docs)
    extra_chunks, doc_string = create_metadata_chunks(docs_chunks)
    
    # Create vector stores
    vecstores = create_vector_stores(docs_chunks, extra_chunks)
    docstore = aggregate_vector_stores(vecstores)
    
    # Create conversation store
    convstore = create_default_faiss()
    
    # Initialize LLM
    llm = ChatNVIDIA(model=LLM_MODEL)
    
    # Create prompt
    chat_prompt = create_chat_prompt()
    
    print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")
    
    return {
        "docstore": docstore,
        "convstore": convstore,
        "llm": llm,
        "chat_prompt": chat_prompt,
        "doc_string": doc_string
    }

def main():
    # Page header
    st.title("📚 Research Papers QA Bot")
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Get resources
    resources = initialize_resources()
    docstore = resources["docstore"]
    convstore = resources["convstore"]
    llm = resources["llm"]
    chat_prompt = resources["chat_prompt"]
    doc_string = resources["doc_string"]
    
    # Display initial message if no messages yet
    if not st.session_state.messages:
        initial_msg = (
            "Hello! I am a document chat agent here to help you! "
            f"I have access to the following papers:\n{doc_string}\n\n"
            "How can I help you?"
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Create retrieval chain using RunnablePassthrough instead of RunnableAssign
    retrieval_chain = (
        RunnablePassthrough()
        | {
            "input": lambda x: x,
            "history": lambda x: docs_to_string(reorder_documents(convstore.as_retriever().invoke(x))),
            "context": lambda x: docs_to_string(reorder_documents(docstore.as_retriever().invoke(x)))
        }
    )
    
    # Create streaming chain
    stream_chain = chat_prompt | print_runnable() | llm | StrOutputParser()
    
    # Input for user question
    if user_input := st.chat_input("Ask a question about the papers"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Perform retrieval
            retrieval = retrieval_chain.invoke(user_input)
            
            # Stream response
            full_response = ""
            message_placeholder = st.empty()
            
            for chunk in stream_chain.stream(retrieval):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Save to conversation memory
            save_memory_and_get_output({'input': user_input, 'output': full_response}, convstore)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Sidebar with information
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This QA bot allows you to chat with research papers using RAG (Retrieval-Augmented Generation).
        
        The system includes papers on:
        - Transformers
        - BERT
        - RAG
        - MRKL
        - Mistral
        - LLM-as-a-Judge
        
        The bot uses NVIDIA's embedding model and Mixtral 8x22B for generating responses.
        """)

if __name__ == "__main__":
    main()