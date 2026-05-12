# %% Imports
from multiprocessing.sharedctypes import Value
import os
from posixpath import exists
from llama_cpp import Llama
import time
import numpy as np
import chromadb
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

# %% Model
context_window = 2048
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    verbose=True,
    n_ctx=context_window,
    n_batch=context_window # Need to double check if this is a good idea... Definitely not something that will scale lol
)

# %% Embed Function
def embed_file(file: str, context_window: int):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        # Document token length check
        text_token = text.encode('utf-8') # Some reason `.tokenize()` gets upset without this. Involves a utf encoding error I think
        text_token_len = len(llm.tokenize(text_token, add_bos=False))
        if text_token_len > context_window:
            print(f"Error: File `{file}` exceeded context window: {text_token_len}. Abandoning...\n\n")
            return None
        else: # Embed the file
            start_time = time.perf_counter()
            print(f"Processing File: `{file}` with token length: {text_token_len}")
            embeddings = llm.create_embedding(text)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"File `{file}` embedded in {elapsed_time:.2f} seconds at {text_token_len / (elapsed_time):.2f} tokens/second\n\n")

            return embeddings

# %% Chromadb
@register_embedding_function
class MyEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model, context_window: int):
        self.model = model
        self.context_window = context_window

    def __call__(self, input: Documents) -> Embeddings:
        results = []
        for doc in input:
            embedding = embed_file(doc, self.context_window)
            if embedding is None:
                raise ValueError(f"Failed to embed document: {doc}")
            results.append(embedding["data"][0]["embedding"])
        return results

    @staticmethod
    def name() -> str:
        return "my-ef"

    def get_config(self) -> Dict[str, Any]:
        return {"model": self.model, "context_window": self.context_window}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return MyEmbeddingFunction(config["model"], config["context_window"])

test_db_path = "test_db"
os.makedirs(test_db_path, exist_ok=True)

my_ef = MyEmbeddingFunction(model=llm, context_window=context_window)
client = chromadb.PersistentClient(path=test_db_path)
collection = client.get_or_create_collection(
    name="test-database",
    embedding_function=my_ef)

# %% Test doc
test_doc = "../data/summary/httpsawsamazoncomwhatisvectordatabases.md"

# %% Add to ChromaDB collection
