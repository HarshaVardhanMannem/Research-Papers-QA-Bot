"""Embedding utilities for the QA system."""
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from config.settings import EMBEDDING_MODEL

def get_embedder():
    """Initialize and return the NVIDIA embeddings model."""
    # You can uncomment this to list available models
    # NVIDIAEmbeddings.get_available_models()
    return NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate="END")