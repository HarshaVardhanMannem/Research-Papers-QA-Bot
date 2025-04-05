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
            if not paper_id or paper_id.strip() == "":
                print("Skipping empty paper ID")
                continue
                
            doc = ArxivLoader(query=paper_id).load()
            if doc and len(doc) > 0:
                docs.append(doc)
            else:
                print(f"No document content returned for paper ID: {paper_id}")
        except Exception as e:
            print(f"Error loading paper {paper_id}: {e}")
    
    return docs

def load_single_arxiv_document(paper_id):
    """Load a single document from Arxiv based on paper ID."""
    if not paper_id or paper_id.strip() == "":
        print("Error: Empty paper ID provided")
        return None
        
    print(f"Loading paper {paper_id}")
    try:
        # Clean paper ID format if needed
        if paper_id.startswith(("arXiv:", "arxiv:")):
            paper_id = paper_id.split(":")[-1].strip()
            
        doc = ArxivLoader(query=paper_id).load()
        
        # Validate document content
        if not doc or len(doc) == 0:
            print(f"No document content returned for paper ID: {paper_id}")
            return None
            
        if not hasattr(doc[0], 'page_content') or not doc[0].page_content:
            print(f"Document has no page content for paper ID: {paper_id}")
            return None
            
        return doc
    except Exception as e:
        print(f"Error loading paper {paper_id}: {e}")
        return None

def preprocess_documents(docs):
    """Preprocess documents by truncating at References section."""
    processed_docs = []
    
    for doc in docs:
        if not doc or len(doc) == 0 or not hasattr(doc[0], 'page_content'):
            continue
            
        content = json.dumps(doc[0].page_content)
        if "References" in content:
            doc[0].page_content = content[:content.index("References")]
        processed_docs.append(doc)
        
    return processed_docs

def create_document_chunks(docs):
    """Split documents into chunks and filter out short chunks."""
    text_splitter = create_text_splitter()
    print("Chunking Documents")
    
    docs_chunks = []
    for doc in docs:
        if not doc or len(doc) == 0 or not hasattr(doc[0], 'page_content') or not doc[0].page_content:
            docs_chunks.append([])
            continue
            
        chunks = text_splitter.split_documents(doc)
        valid_chunks = [c for c in chunks if len(c.page_content) > MIN_CHUNK_LENGTH]
        docs_chunks.append(valid_chunks)
    
    return docs_chunks

def create_metadata_chunks(docs_chunks):
    """Create additional chunks with document metadata."""
    doc_string = "Available Documents:"
    doc_metadata = []
    
    for chunks in docs_chunks:
        if chunks and len(chunks) > 0:
            metadata = getattr(chunks[0], 'metadata', {})
            title = metadata.get('Title', 'Untitled')
            doc_string += f"\n - {title}"
            doc_metadata.append(str(metadata))
    
    if len(doc_metadata) == 0:
        doc_string += "\n - No documents loaded yet."
        
    return [doc_string] + doc_metadata, doc_string