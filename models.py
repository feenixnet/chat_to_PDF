import os
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Add imports for additional models if needed.


class Models:
    def get_embedding_model(self, model_name):
        if model_name == "ollama-llama":
            return OllamaEmbeddings(model="mxbai-embed-large")
        # Add more conditional options for different models here if needed.

    def get_chat_model(self, model_name):
        if model_name == "ollama-llama":
            return ChatOllama(model="llama3.2", temperature=0)
        # Add additional model options here if available.
