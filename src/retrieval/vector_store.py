"""Vector store creation and management utilities."""
from langchain.vectorstores import FAISS
from src.embedding.embeddings import get_embedder
import numpy as np

def create_default_faiss():
    """Create a default FAISS vector store."""
    embedder = get_embedder()
    return FAISS.from_texts(["dummy_text_for_initialization"], embedder)

def create_vector_store(docs_chunks, embedder):
    """Create a vector store from document chunks."""
    if not docs_chunks:
        return create_default_faiss()
    return FAISS.from_documents(docs_chunks, embedder)

def create_vector_stores(docs_chunks_list, extra_docs_list=None):
    """Create vector stores for multiple document chunk lists."""
    embedder = get_embedder()
    vector_stores = []
    
    # Process main document chunks
    for docs_chunks in docs_chunks_list:
        if docs_chunks:
            vector_store = create_vector_store(docs_chunks, embedder)
            vector_stores.append(vector_store)
    
    # Process extra documents if provided
    if extra_docs_list:
        for extra_doc in extra_docs_list:
            if isinstance(extra_doc, str) and extra_doc.strip():  # Ensure non-empty string
                vector_store = FAISS.from_texts([extra_doc], embedder)
                vector_stores.append(vector_store)
    
    return vector_stores

def aggregate_vector_stores(vector_stores):
    """Aggregate multiple vector stores into one."""
    if not vector_stores:
        return create_default_faiss()
    
    base_store = vector_stores[0]
    for store in vector_stores[1:]:
        base_store.merge_from(store)
    
    return base_store

def add_documents_to_vector_store(vector_store, docs_chunks):
    """Add new document chunks to an existing vector store."""
    if not docs_chunks:
        return vector_store
        
    embedder = get_embedder()
    
    # Create embeddings and add to store
    valid_chunks = [chunk for chunk in docs_chunks if hasattr(chunk, 'page_content') and chunk.page_content.strip()]
    
    if not valid_chunks:
        print("Warning: No valid chunks to add to vector store")
        return vector_store
        
    vector_store.add_documents(valid_chunks)
    return vector_store

def reorder_documents(docs):
    """Reorder documents by relevance score."""
    if not docs:
        return []
    return sorted(docs, key=lambda x: getattr(x, "relevance_score", 0), reverse=True)

def docs_to_string(docs):
    """Convert document list to string representation."""
    if not docs:
        return "No relevant documents found."
    
    result = []
    for doc in docs:
        content = getattr(doc, "page_content", "")
        if not content or content.strip() == "":
            continue
            
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source", "")
        title = metadata.get("Title", "")
        
        header = f"Source: {source}" if source else ""
        header += f" Title: {title}" if title else ""
        
        if header:
            result.append(f"{header}\n{content}\n")
        else:
            result.append(content)
    
    if not result:
        return "No relevant document content found."
    
    return "\n\n".join(result)