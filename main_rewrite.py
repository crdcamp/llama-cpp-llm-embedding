# %% Imports
from llama_cpp import Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from datetime import datetime
import os

# %% Model Params
context_window = 2048
model_path = "models/Qwen3-Embedding-8B-Q6_K.gguf"

# %% Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
