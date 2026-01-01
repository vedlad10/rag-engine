# RAG Document Question Answering

This project implements a Retrieval-Augmented Generation (RAG) engine to answer user queries from PDFs and text documents with reduced hallucination.

## How it works
- Documents are split into chunks
- Embeddings are generated for each chunk
- FAISS is used for similarity search
- Retrieved context is passed to an LLM for answer generation

## Tech Stack
- LangChain
- FAISS
- Embeddings
- Google Colab

## Features
- PDF and text document ingestion
- Persistent vector store
- Context-grounded responses
