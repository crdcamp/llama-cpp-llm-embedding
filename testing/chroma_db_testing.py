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

# %% Model Params
context_window = 2048
model_path = "../models/Qwen3-Embedding-8B-Q6_K.gguf"

# %% Load Model
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    verbose=True,
    n_ctx=context_window,
    n_batch=context_window # Need to double check if this is a good idea... Probably not a good idea...
)

# %% Custom embedding function for llama cpp
@register_embedding_function
class LlamaCppEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model, model_path: str):
        self.model = model
        self.model_path = model_path

    def __call__(self, input: Documents) -> Embeddings:
        result = self.model.create_embedding(list(input))
        return [item['embedding'] for item in result['data']]

    @staticmethod
    def name() -> str:
        return "my-ef"

    def get_config(self) -> Dict[str, Any]:
        return dict(model_path=self.model_path)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "LlamaCppEmbeddingFunction":
        model = Llama(model_path=config['model_path'], embedding=True)
        return LlamaCppEmbeddingFunction(model=model, model_path=config['model_path'])

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

# %% Create some test text to pass to the db
test_file = "../data/summary/httpsblogapifycomwhatisavectordatabase.md"
with open(test_file, 'r', encoding='utf-8') as f:
    text = f.read()

# %% Insert test file embeddings into ChromaDB
collection.add(
    ids=["id1"], # Replace this with a uuid later: https://www.youtube.com/watch?v=yvsmkx-Jaj0&t=318s
    documents=[text],
    metadatas=[{"source": test_file}],
)

# %% Inspect and query
collection_result = client.get_collection(name="test-collection")
all_collection_results = client.list_collections()
print(collection_result)
print(all_collection_results)

collection.query(
    query_texts=["The meaning of a vector database"]
)
