"""Vector store creation and management."""
from faiss import IndexFlatL2
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.document_transformers import LongContextReorder

from src.embedding.embeddings import get_embedder

def create_default_faiss():
    """Create an empty FAISS vector store."""
    embedder = get_embedder()
    embed_dims = len(embedder.embed_query("test"))
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def create_vector_stores(docs_chunks, extra_chunks):
    """Create vector stores from document chunks."""
    print("Constructing Vector Stores")
    embedder = get_embedder()
    
    # Create vector store for extra chunks
    vecstores = [FAISS.from_texts(extra_chunks, embedder)]
    
    # Create vector stores for document chunks
    for doc_chunks in docs_chunks:
        if doc_chunks:  # Make sure there are chunks to add
            vecstore = FAISS.from_documents(doc_chunks, embedder)
            vecstores.append(vecstore)
    
    return vecstores

def aggregate_vector_stores(vectorstores):
    """Merge all vector stores into a single one."""
    agg_vstore = create_default_faiss()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

def reorder_documents(docs):
    """Reorder documents to optimize context window usage."""
    reorderer = LongContextReorder()
    return reorderer.transform_documents(docs)

def docs_to_string(docs, title="Document"):
    """Convert document chunks to a formatted string."""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str