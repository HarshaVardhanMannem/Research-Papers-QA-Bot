"""Document loading and preprocessing utilities."""
import json
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    CHUNK_SEPARATORS, 
    MIN_CHUNK_LENGTH, 
    PAPER_IDS
)

def create_text_splitter():
    """Create and return a text splitter for chunking documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

def load_arxiv_documents():
    """Load documents from Arxiv based on configured paper IDs."""
    print("Loading Documents")
    docs = []
    for paper_id in PAPER_IDS:
        try:
            doc = ArxivLoader(query=paper_id).load()
            docs.append(doc)
        except Exception as e:
            print(f"Error loading paper {paper_id}: {e}")
    
    return docs

def preprocess_documents(docs):
    """Preprocess documents by truncating at References section."""
    for doc in docs:
        content = json.dumps(doc[0].page_content)
        if "References" in content:
            doc[0].page_content = content[:content.index("References")]
    return docs

def create_document_chunks(docs):
    """Split documents into chunks and filter out short chunks."""
    text_splitter = create_text_splitter()
    print("Chunking Documents")
    docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
    docs_chunks = [[c for c in dchunks if len(c.page_content) > MIN_CHUNK_LENGTH] for dchunks in docs_chunks]
    return docs_chunks

def create_metadata_chunks(docs_chunks):
    """Create additional chunks with document metadata."""
    doc_string = "Available Documents:"
    doc_metadata = []
    for chunks in docs_chunks:
        if chunks:
            metadata = getattr(chunks[0], 'metadata', {})
            doc_string += "\n - " + metadata.get('Title', 'Untitled')
            doc_metadata.append(str(metadata))
    
    return [doc_string] + doc_metadata, doc_string