# %% Imports
from langchain_text_splitters.base import TokenTextSplitter
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
    n_batch=context_window # Need to double check if this is a good idea...
)

# %% Text splitter
# Not sure if I need it here given the token text length check...
# Hmmmm ......
text_splitter_chunk_size = 2048
text_splitter = TokenTextSplitter(
    chunk_size=text_splitter_chunk_size,
    chunk_overlap=100,
)

"""
Instead of a text splitter, I might just do a token
check here. I'm not sure how necessary the text split is in comparison to
the context of the text.

I'm still not even sure if the batches mess with context, or even in what way...
Needs some research
""";

# %% Embed Function
def embed_file(file: str, context_window: int):
    start_time = time.perf_counter()
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

        # Token length calculation
        text_token = text.encode('utf-8') # Some reason `.tokenize()` gets upset without this. Involves a utf encoding error I think
        text_token_len = len(llm.tokenize(text_token, add_bos=False))
        if text_token_len > text_splitter_chunk_size:
            print(f"Error: File {file} exceeded chunk size: {text_token_len}. Abandoning...")
            return
        else:
            del text_token, text_token_len # Won't be needing these from here on out
            # Split text
            split_text = text_splitter.create_documents([text])
            documents = [doc.page_content for doc in split_text]
            documents_embeddings = []

            for doc in documents:
                embeddings = llm.create_embedding(doc) # For some reason list comprehension won't work here
                documents_embeddings.extend(embeddings["data"])


    end_time = time.perf_counter()
    print(f"File: {file} embedded in {end_time - start_time:.2f} seconds")

    return documents_embeddings

# %% Testing doc
test_doc = "data/summary/httpsawsamazoncomwhatisvectordatabases.md"

# %% Embed
test_documents_embeddings = embed_file(test_doc, context_window=context_window)

"""
Need to do aggregate testing before continuing here
"""
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
