# LLM-powered Research Papers QA Bot
Research paper QA bot using RAG agents with LLM

This application allows you to chat with research papers using RAG (Retrieval-Augmented Generation) techniques. The app loads papers from Arxiv, processes them into chunks, embeds them using NVIDIA's embedding model, stores them in a vector database, and uses Mixtral 8x22B for generating responses based on retrieved context.

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

-   Document loading from Arxiv using paper IDs.
-   **Dynamic Paper Addition:** Ability to add new research papers from Arxiv by providing their IDs directly within the application.
-   Text chunking and preprocessing for efficient retrieval.
-   Vector storage and retrieval using FAISS.
-   **Vector Store Updates:** The vector store is updated dynamically when new papers are added.
-   Conversation memory to maintain context during chat sessions.
-   Streamlit interface for easy interaction, including adding papers and asking questions.
-   Utilizes NVIDIA embedding models and Mixtral 8x22B via NVIDIA AI Endpoints.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Optional) Configure API Keys:** Ensure your NVIDIA API key is set up correctly. This might involve setting environment variables or configuring `config/settings.py` depending on your implementation for accessing NVIDIA AI Endpoints.
4.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

## Usage

1.  **Launch the app:** Run `streamlit run streamlit_app.py` in your terminal.
2.  **(Optional) Add a New Paper:**
    * Locate the input field in the Streamlit interface designed for adding Arxiv IDs.
    * Enter the Arxiv ID (e.g., `1706.03762`) of the paper you want to add.
    * Click the "Add Paper" or similar button to initiate the download, processing, embedding, and vector store update process. Wait for confirmation that the paper has been added successfully.
3.  **Ask Questions:**
    * Enter your question about any of the loaded research papers (including the ones you've added) in the chat input area.
    * The system will retrieve relevant text chunks from the vector store and use the LLM (Mixtral 8x22B) to generate a contextualized response.

## Default Papers Included

The following papers are typically pre-loaded or included as defaults in the initial setup. You can add more papers using the application's interface.

-   "Attention Is All You Need" (Transformer paper) - Arxiv ID: `1706.03762`
-   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - Arxiv ID: `1810.04805`
-   "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG paper) - Arxiv ID: `2005.11401`
-   "Leveraging Modular and Recursive Knowledge for Large Language Model Reasoning" (MRKL paper) - Arxiv ID: `2205.00050` (Note: Check Arxiv for exact MRKL paper ID if needed, this is an example)
-   "Mistral 7B" - Arxiv ID: `2310.06825`
-   "LLM-as-a-Judge" - Arxiv ID: `2306.05685`

*(Note: Please verify the exact Arxiv IDs if necessary)*

## Acknowledgements

This project builds upon the materials and code provided in the NVIDIA [Building RAG Agents with LLMs] course. Significant portions of the code related to the RAG pipeline are derived from the course content, which has been modified and extended to create the full application, including the dynamic paper addition feature. We would like to thank NVIDIA for providing the resources and knowledge that helped shape this project.

### Original Source:
NVIDIA [Building RAG Agents with LLMs]

This project also leverages NVIDIA technology, including LangChain integrations, AI Endpoints for document retrieval and question-answering (using embedding models and Mixtral 8x22B), and potentially other NVIDIA libraries.
