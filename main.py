# %% Imports
from langchain_text_splitters.base import TokenTextSplitter
from llama_cpp import Llama
import os
import time
import numpy as np

# %% Model
context_window = 20480
llm = Llama(
    model_path="models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    verbose=True,
    n_ctx=context_window,
    n_batch=context_window # IN ACTUAL USE CASE: Leave this at 512 and encode the text using batches instead
)

# %% Text splitter
text_splitter = TokenTextSplitter(
    chunk_size=2048,
    chunk_overlap=100,
)

# %% Embed Function
def embed_file(file: str, context_window: int):
    start_time = time.perf_counter()
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

        # Token length calculation
        token_text = text.encode('utf-8')
        text_token_len = len(llm.tokenize(token_text, add_bos=False))
        print(f"File: {file} token length: {text_token_len}")

        # Split text
        split_text = text_splitter.create_documents([text])
        documents = [doc.page_content for doc in split_text]
        documents_embeddings = []

        for doc in documents:
            embeddings = llm.create_embedding(doc) # For some reason list comprehension won't work here

            """
            Not entirely necessary to include this part below... but we'll include it just in case.
            The logic for referencing context window also might be incorrect.
            You'll also want to calculate the tokens earlier in the code as well
            and use this logic earlier, but we'll leave this as is for now
            """
            embedding_token_usage = embeddings['usage']['total_tokens']
            if embedding_token_usage <= context_window:
                documents_embeddings.extend(embeddings["data"])
                print(f"TOTAL TOKENS: {embeddings['usage']['total_tokens']}")
            else:
                print(f"File {file} exceeded context window. Skipping...")
                continue

    end_time = time.perf_counter()
    print(f"Documents embedded in {end_time - start_time:.2f} seconds")

    return documents_embeddings

# %% Testing doc
test_doc = "data/summary/httpsawsamazoncomwhatisvectordatabases.md"

# %% Embed
test_documents_embeddings = embed_file(test_doc, context_window=context_window)

# %% Inspecting
print("test_documents_embeddings PROPERTIES")
print("Type: ", type(test_documents_embeddings))
print("Length: ", len(test_documents_embeddings))
#print("Test vector sample:\n", test_documents_embeddings[0:1], "\n")


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
