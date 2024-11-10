import streamlit as st
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone, Index, ServerlessSpec
import pinecone
from models import Models
from ingest import ingest_file
import os
import time

# Load the environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize the Pinecone client
pinecone_client = Pinecone(api_key="6add807c-ec89-40b3-abff-1e98f950a5d1")

# Create the index if it doesn’t exist
index_name = "documents"
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=1024,  # Use the appropriate dimension for your embeddings model
        metric="cosine",  # You can also choose 'euclidean' or other metrics
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to the index
vector_store = pinecone_client.Index(index_name)

# Initialize Models
models = Models()

# Streamlit UI layout
st.title("PDF Chat Assistant with Model Selection")
st.sidebar.header("Settings")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    file_path = f"./data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")

# Model selection section
model_options = ["ollama-llama", "other_model"]  # Update with actual model names
selected_model = st.sidebar.selectbox("Choose the model", model_options)

# Initialize selected model
embeddings = models.get_embedding_model(selected_model)
llm = models.get_chat_model(selected_model)

# Ingest PDF if uploaded
if uploaded_file and st.sidebar.button("Ingest PDF"):
    st.write("Ingesting PDF... Please wait.")
    ingest_file(file_path, embeddings, vector_store)
    st.write("Ingestion completed.")

# Chat functionality
st.write("## Chat with PDF Content")
query = st.text_input("Ask a question about the content:")


# Custom retriever function using Pinecone
def retrieve_from_pinecone(
    query_text, vector_store, embeddings, top_k=3
):  # Reduce top_k
    # Generate query embedding
    query_embedding = embeddings.embed_query(query_text)

    # Query Pinecone for similar documents
    results = vector_store.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,  # Ensures metadata (like text) is included in results
    )

    # Filter out duplicate text blocks by converting to a set and back to a list
    unique_texts = list({match["metadata"]["text"] for match in results["matches"]})
    return unique_texts


# Add retry mechanism for querying
def query_with_retry(retrieve_func, query_text, llm, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            # Retrieve documents from Pinecone
            context_docs = retrieve_func(query_text, vector_store, embeddings)
            # Combine unique documents into context for the LLM chain
            context = " ".join(context_docs)

            # Display the context retrieved for debugging
            # st.write("Retrieved Context for Debugging:", context)

            # General-purpose message format for user queries
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the user's questions based on the provided context from the PDF. Use only the context information to answer as accurately as possible.",
                },
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": query_text},
            ]

            # Use invoke with properly formatted messages
            return llm.invoke(messages)

        except pinecone.exceptions.ServiceException as e:
            attempt += 1
            if attempt < retries:
                st.write("Encountered an error. Retrying...")
                time.sleep(2)  # Delay before retry
            else:
                st.error("Failed to retrieve answer after multiple attempts.")
                st.error(str(e))
                return None


# Execute the query with retry handling
if st.button("Submit") and query:
    result = query_with_retry(retrieve_from_pinecone, query, llm)
    print("result", result)

    if result:
        st.write("**Answer:** ", result.content)
