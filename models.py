# models.py

import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Add imports for additional models if needed.


class Models:
    def get_embedding_model(self, model_name):
        if model_name == "ollama-llama":
            return OllamaEmbeddings(model="mxbai-embed-large")
        elif model_name == "distilbert":
            from langchain.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
        elif model_name == "cohere":
            from langchain.embeddings import CohereEmbeddings

            return CohereEmbeddings(model="cohere-embed-medium")

        # Add more conditional options for different models here if needed.
        raise ValueError(f"Embedding model {model_name} is not supported.")

    def get_chat_model(self, model_name):
        if model_name == "ollama-llama":
            return ChatOllama(model="llama3.2", temperature=0)
        elif model_name == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
        elif model_name == "gpt-neo-125M":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )

        # Add additional model options here if available.
        raise ValueError(f"Chat model {model_name} is not supported.")
