# External Tools and Libraries

This project uses several Python libraries to implement a Retrieval-Augmented Generation (RAG) system:

* Streamlit: Used to build the web-based interface for file upload, parameter adjustment, and interactive chat.

* NumPy: Handles embedding vector storage and cosine similarity calculation.

* OpenAI SDK: Connects to Cornell’s AI Gateway (https://api.ai.it.cornell.edu/v1) for both embeddings and chat completions.

* pypdf: Extracts text from uploaded PDF files.

* Built-in modules (os, io, math, time, typing): Used for file handling, environment configuration, and type annotations.

## Models and APIs

The RAG pipeline relies on:

* meta.llama-3.2-1b-instruct as the main large language model for conversational responses.

* google.text-embedding as the embedding model to generate dense vector representations of text for retrieval.

## GenAI Usage

ChatGPT was used to assist in:

* Designing the modular structure of the RAG system, including document chunking, embedding, retrieval, and chat response generation.

* Improving PDF parsing by recommending the use of the pypdf library.

* Clarifying the logic for cosine similarity computation and dynamic top-k retrieval.

* Refining the Streamlit user interface layout for better usability.

All AI assistance was limited to code structuring and explanation.
The actual implementation, debugging, and system integration were performed independently.
