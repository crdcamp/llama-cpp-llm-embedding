# %% Imports
from llama_cpp import Llama
import time
import numpy as np
import chromadb

# %% Model
"""
Gonna abandon using text splitting on the documents for now.

I have yet to confirm but I think text splitting might mess with the
embedding results.

So, we're gonna just leave this as a stripped down idea where
the embedding function checks the length for our (short) context
window.
"""

context_window = 2048
llm = Llama(
    model_path="models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    verbose=True,
    n_ctx=context_window,
    n_batch=context_window # Need to double check if this is a good idea... Definitely not something that will scale lol
)

# %% Embed Function
def embed_file(file: str, context_window: int):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        # Token length check
        text_token = text.encode('utf-8') # Some reason `.tokenize()` gets upset without this. Involves a utf encoding error I think
        text_token_len = len(llm.tokenize(text_token, add_bos=False))
        if text_token_len > context_window:
            print(f"Error: File {file} exceeded context window: {text_token_len}. Abandoning...\n\n")
            return None
        else: # Embed the file
            start_time = time.perf_counter()
            print(f"Processing File: {file}\nFile Token Length: {text_token_len}")
            embeddings = llm.create_embedding(text)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"File: embedded in {elapsed_time:.2f} seconds")
            print(f"Processed in {text_token_len / (elapsed_time):.2f} tokens/second\n\n")
            return embeddings, elapsed_time

# %% Multiple doc aggregate test embedding
test_agg_docs = ["data/summary/httpsawsamazoncomwhatisvectordatabases.md", "data/summary/httpsblogapifycomwhatisavectordatabase.md", "data/summary/httpsbrainyxcojournaljournal22.md"]
test_embeddings = []
embedding_times = []

# %% Testing embed function
for doc in test_agg_docs:
    result = embed_file(doc, context_window=context_window)
    if result is not None:
        embeddings, elapsed_time = result
        test_embeddings.append(embeddings)
        embedding_times.append(elapsed_time)

print(f"Total embedding time: {sum(embedding_times)}")
"""
You might wanna think about graphing these embeddings to be
sure you're doing this correctly
"""
# %% Chromadb
chroma_client = chromadb.Client()

# We'll look into naming these in a better way later
# Maybe just use the model to do it somehow?
# Idk...
client = chromadb.PersistentClient(path="db")


# %%


# %% Cosine Similarity
def calculate_cosine_similarity(array):
    vector = np.array(array).flatten()
    print(vector.shape)
