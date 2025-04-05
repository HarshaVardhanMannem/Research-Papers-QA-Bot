# LLM-powered Research Papers QA Bot
Research paper QA bot using RAG agents with LLM

This application allows you to chat with research papers using RAG (Retrieval-Augmented Generation) techniques. The app loads papers from Arxiv, processes them into chunks, embeds them using NVIDIA's embedding model, and uses Mixtral 8x22B for generating responses.

## Project Structure

The project is organized as follows:

project-root/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── document_loader.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embeddings.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── chat_prompts.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
└── config/
    └── settings.py

## Features

- Document loading from Arxiv
- Text chunking and preprocessing
- Vector storage and retrieval using FAISS
- Conversation memory to maintain context
- Streamlit interface for easy interaction

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

1. Launch the app
2. Enter your question about any of the preloaded research papers
3. The system will retrieve relevant chunks and generate a response

## Papers Included

- "Attention Is All You Need" (Transformer paper)
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG paper)
- "Leveraging Modular and Recursive Knowledge for Large Language Model Reasoning" (MRKL paper)
- "Mistral 7B" (Mistral paper)
- "LLM-as-a-Judge"

## Acknowledgements

This project builds upon the materials and code provided in the NVIDIA [Building RAG Agents with LLMs] course. Significant portions of the code are derived from the course content, which has been modified and extended to create the full application. We would like to thank NVIDIA for providing the resources and knowledge that helped shape this project.

### Original Source:
NVIDIA [Building RAG Agents with LLMs]

This project also uses NVIDIA's LangChain library and AI Endpoints for document retrieval and question-answering, as well as their Mixtral 8x22B model for generating responses.

