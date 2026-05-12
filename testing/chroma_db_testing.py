# %% Imports
import os
from pyexpat import model
from llama_cpp import Llama
import time
import chromadb
import numpy as np
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
import itertools
from datetime import datetime

# %% Model
context_window = 2048
model_path = "../models/Qwen3-Embedding-8B-Q6_K.gguf"

llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    verbose=True,
    n_ctx=context_window,
    n_batch=context_window # Need to double check if this is a good idea... Probably not a good idea...
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

# %% Test embedding
test_doc = "../data/summary/httpsawsamazoncomwhatisvectordatabases.md"
test_embedding = embed_file(test_doc, context_window=context_window)

# %% Inspect and access only embeddings
print(type(test_embedding))
for key, value in test_embedding.items():
    print(key)

# %% Custom embedding function
# Start here: https://docs.trychroma.com/docs/embeddings/embedding-functions#custom-embedding-functions
# From: https://github.com/chroma-core/chroma/issues/2409
@register_embedding_function
class LlamaCppEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model, model_path: str):
        self.model = model
        self.model_path = model_path  # store the path string separately for serialization

    def __call__(self, input: Documents) -> Embeddings:
        result = self.model.create_embedding(list(input))
        return [item['embedding'] for item in result['data']]

    @staticmethod
    def name() -> str:
        return "my-ef"

    def get_config(self) -> Dict[str, Any]:
        return dict(model_path=self.model_path)  # return the string, not the Llama object

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "LlamaCppEmbeddingFunction":
        model = Llama(model_path=config['model_path'], embedding=True)
        return LlamaCppEmbeddingFunction(model=model, model_path=config['model_path'])



# %%
all_test_embeddings = [item['embedding'] for item in test_embedding['data']]
# Flatten list (was a nested list before)
all_test_embeddings = list(itertools.chain.from_iterable(all_test_embeddings))
print(len(all_test_embeddings))

# %% Initialize ChromaDB
# We'll use PesistentClient outside of testing
# https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="test-collection",
    embedding_function=LlamaCppEmbeddingFunction(model=llm, model_path=model_path),
    metadata={
        "description": "A test collection for learning ChromaDB",
        "created": str(datetime.now())
    },
    # More info on configuration: https://docs.trychroma.com/docs/collections/configure#what-is-an-hnsw-index
    configuration={
        "hnsw": {
            "space": "cosine", # Turns out we don't need that cosine function
            "ef_construction": 100, # 100 is the default value
            "ef_search": 100, # 100 is the default value
        }
    }
)

# %% Insert embeddings into ChromaDB
collection.add(
    ids=["id1"], # Replace this with a uuid later: https://www.youtube.com/watch?v=yvsmkx-Jaj0&t=318s
    embeddings=all_test_embeddings,
    metadatas=[{"document": f"{test_doc}"}],
)

# %% Inspect and query
collection_result = client.get_collection(name="test-collection")
all_collection_results = client.list_collections()
print(collection_result)
print(all_collection_results)

"""
Since ChromaDB will use the collection's embedding function (https://docs.trychroma.com/docs/querying-collections/query-and-get#query),
this is likely an incorrect implementation.

With your embed function defined as `None`. this probably isn't
working correctly...
"""

collection.query(
    query_texts=["The meaning of a vector database"]
)

# %%
collection.get(ids=["id1"])
