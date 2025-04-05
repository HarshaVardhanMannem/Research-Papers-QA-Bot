"""Streamlit app for the Research Papers QA Bot with dynamic paper upload."""
import streamlit as st
from langchain.schema.output_parser import StrOutputParser
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
    create_metadata_chunks,
    load_single_arxiv_document
)
from src.embedding.embeddings import get_embedder
from src.retrieval.vector_store import (
    create_default_faiss,
    create_vector_stores, 
    aggregate_vector_stores,
    reorder_documents,
    docs_to_string,
    add_documents_to_vector_store
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
        "doc_string": doc_string,
        "loaded_papers": [paper_id for doc_chunks in docs_chunks for paper_id in [getattr(chunk, 'metadata', {}).get('Source', '') for chunk in doc_chunks] if paper_id]
    }

def add_paper_to_resources(paper_id, resources):
    """Add a new paper to existing resources."""
    try:
        # Load single document
        new_doc = load_single_arxiv_document(paper_id)
        if not new_doc or len(new_doc) == 0:
            return False, f"Failed to load paper with ID: {paper_id}"
        
        # Preprocess
        new_doc = preprocess_documents([new_doc])
        new_doc_chunks = create_document_chunks(new_doc)
        
        # Validate chunks
        if not new_doc_chunks or not new_doc_chunks[0] or len(new_doc_chunks[0]) == 0:
            return False, f"Paper loaded but no valid content chunks were created for: {paper_id}"
        
        # Add to vector store
        add_documents_to_vector_store(resources["docstore"], new_doc_chunks[0])
        
        # Update document string
        metadata = getattr(new_doc_chunks[0][0], 'metadata', {})
        title = metadata.get('Title', 'Untitled')
        resources["doc_string"] += f"\n - {title}"
        resources["loaded_papers"].append(paper_id)
            
        return True, f"Successfully added paper: {title}"
    except Exception as e:
        return False, f"Error processing paper {paper_id}: {str(e)}"

def main():
    # Page header
    st.title("📚 Research Papers Q&A Assistant")
    
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
    
    # Paper upload section - place in sidebar
    with st.sidebar:
        st.title("Add New Papers")
        with st.form("upload_paper_form"):
            paper_id = st.text_input("ArXiv Paper ID (e.g., 1706.03762)", 
                                     help="Enter the ArXiv ID of the paper you want to add")
            
            # Add pattern hint and validation feedback
            if paper_id and not (paper_id.startswith(("arXiv:", "arxiv:")) or paper_id.count(".") == 1):
                st.info("Hint: ArXiv IDs typically look like '1706.03762' or 'arXiv:1706.03762'")
            
            # Check if paper already loaded
            already_loaded = paper_id in resources.get("loaded_papers", [])
            if already_loaded:
                st.warning("This paper is already loaded in the system.")
                
            submitted = st.form_submit_button("Add Paper")
            
            if submitted and paper_id and not already_loaded:
                with st.spinner(f"Loading and processing paper {paper_id}..."):
                    success, message = add_paper_to_resources(paper_id, resources)
                    if success:
                        st.success(message)
                        # Add system message noting the addition of a new paper
                        system_msg = f"System: {message}"
                        st.session_state.messages.append({"role": "assistant", "content": system_msg})
                    else:
                        st.error(message)
        
        # Display currently loaded papers
        st.subheader("Loaded Papers")
        paper_list = resources.get("loaded_papers", [])
        if paper_list:
            for idx, paper in enumerate(paper_list):
                st.text(f"{idx+1}. {paper}")
        else:
            st.text("No papers loaded yet.")
        
        # About section
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
        
        You can add new papers by entering their ArXiv IDs above.
        
        The bot uses NVIDIA's embedding model and Mixtral 8x22B for generating responses.
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Create retrieval chain using RunnablePassthrough
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
            # Validate user input is not empty
            if not user_input or user_input.strip() == "":
                error_message = "I cannot process an empty question. Please provide a valid question about the papers."
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.write(error_message)
            else:
                try:
                    # Perform retrieval
                    retrieval = retrieval_chain.invoke(user_input)
                    
                    # Check if we have any context returned
                    if not retrieval.get("context", "").strip():
                        fallback_context = "I don't have specific information about that in my knowledge base. I'll try to answer based on general knowledge about research papers."
                        retrieval["context"] = fallback_context
                    
                    # Stream response
                    full_response = ""
                    message_placeholder = st.empty()
                    
                    try:
                        for chunk in stream_chain.stream(retrieval):
                            if chunk:  # Ensure chunk is not empty
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                        
                        # Handle empty response case
                        if not full_response:
                            full_response = "I'm sorry, I couldn't generate a proper response. This might be due to the context being insufficient or the question being outside the scope of the loaded papers."
                    except Exception as e:
                        full_response = f"I encountered an error while generating the response: {str(e)}"
                    
                    message_placeholder.markdown(full_response)
                    
                    # Save to conversation memory
                    save_memory_and_get_output({'input': user_input, 'output': full_response}, convstore)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.write(error_message)

if __name__ == "__main__":
    main()