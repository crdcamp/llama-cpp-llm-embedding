# %% Imports
import os
from llama_cpp import Llama
import time
import chromadb
import itertools

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

# %% Test embedding
test_doc = "../data/summary/httpsawsamazoncomwhatisvectordatabases.md"
test_embedding = embed_file(test_doc, context_window=context_window)

# %% Inspect and access only embeddings
print(type(test_embedding))
for key, value in test_embedding.items():
    print(key)

# %%
all_test_embeddings = [item['embedding'] for item in test_embedding['data']]
# Flatten list (was a nested list before)
all_test_embeddings = list(itertools.chain.from_iterable(all_test_embeddings))
print(all_test_embeddings)

# %% Initialize ChromaDB
# We'll use PesistentClient outside of testing
# https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection(
    name="test-collection",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

# %% Figure out how to insert embeddings into ChromaDB
collection.add(
    ids=["id1"],
    embeddings=all_test_embeddings,
    metadatas=[{"document": f"{test_doc}"}],
)
