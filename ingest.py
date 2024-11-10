# ingest.py

import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from uuid import uuid4


# Pinecone setup
def ingest_file(file_path, embeddings, vector_store):
    # Skip non-PDF files
    if not file_path.lower().endswith(".pdf"):
        print(f"Skipping non-PDF file: {file_path}")
        return

    print(f"Starting to ingest file: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(loaded_documents)

    # Create embeddings for each document chunk
    uuids = [str(uuid4()) for _ in range(len(documents))]
    for doc, doc_id in zip(documents, uuids):
        embedding = embeddings.embed_query(
            doc.page_content
        )  # Updated to use page_content
        vector_store.upsert(vectors=[(doc_id, embedding, {"text": doc.page_content})])
    print(f"Finished ingesting file: {file_path}")
