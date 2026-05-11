# %% Imports
from llama_cpp import Llama
import os
import time
import numpy as np

# %% Model
"""
Gonna abandon using text splitting on the documents for now.

I have yet to confirm but I think text splitting might mess with the
embedding results.

So, we're gonna just leave this as a stripped down idea where
the embedding function checks the length for our (short) context
window.
"""
context_window = 4096
llm = Llama(
    model_path="models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    verbose=True,
    n_ctx=context_window,
    n_batch=context_window # Need to double check if this is a good idea... Definitely not something that will scale lol
)

# %% Embed Function
def embed_file(file: str, context_window: int):
    start_time = time.perf_counter()
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        # Token length check
        text_token = text.encode('utf-8') # Some reason `.tokenize()` gets upset without this. Involves a utf encoding error I think
        text_token_len = len(llm.tokenize(text_token, add_bos=False))
        if text_token_len > context_window:
            print(f"Error: File {file} exceeded chunk size: {text_token_len}. Abandoning...")
            return None
        else: # Embed
            print(f"Processing File: {file}\nToken Length: {text_token_len}")
            embeddings = llm.create_embedding(text) # For some reason list comprehension won't work here
            end_time = time.perf_counter()
            print(f"File: {file} embedded in {end_time - start_time:.2f} seconds")
            return embeddings

# %% Testing doc
test_doc = "data/summary/httpsawsamazoncomwhatisvectordatabases.md"

# %% Embed
test_documents_embeddings = embed_file(test_doc, context_window=context_window)



# %% Multiple doc aggregate test embedding
test_agg_docs = ["data/summary/httpsawsamazoncomwhatisvectordatabases.md", "data/summary/httpsblogapifycomwhatisavectordatabase.md", "data/summary/httpsbrainyxcojournaljournal22.md"]
test_embeddings = []
for doc in test_agg_docs:
    embeddings = embed_file(doc, context_window=context_window)
    test_embeddings.append(embeddings)

# %% Inspect


# %% Cosine Similarity
def calculate_cosine_similarity(vector):
    vector = np.array(vector).flatten()
    print(vector.shape)

# %% Test cosine similarity
test_cosine_sim = calculate_cosine_similarity(test_documents_embeddings)
