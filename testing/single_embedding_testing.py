# %% Imports
from llama_cpp import Llama
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import numpy as np

# %% Test file
test_file = "../data/summary/httpswwwdatabrickscomblogwhatisvectordatabase.md"

# %% Model
context_window = 20480
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    verbose=True,
    n_ctx=context_window,
    n_batch=context_window # IN ACTUAL USE CASE: Leave this at 512 and encode the text using batches instead
)

# %% Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

# %% Conversion to function
def embed_file(file, context_window: int):
    start_time = time.perf_counter()
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        split_text = text_splitter.create_documents([text])
        documents = [doc.page_content for doc in split_text]
        documents_embeddings = []

        for doc in documents:
            embeddings = llm.create_embedding(doc)

            """
            Not entirely necessary to include this part below... but we'll include it just in case
            The logic for referencing context window might also be incorrect.
            You'll also want to calculate the tokens earlier in the code as well,
            and use this logic earlier, but we'll leave this as is for now
            """
            embedding_token_usage = embeddings['usage']['total_tokens']
            if embedding_token_usage < context_window:
                documents_embeddings.extend(embeddings["data"])
            else:
                print(f"File {file} exceeded context window. Skipping...")
                continue

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Documents embedded in {elapsed_time:.2f} seconds")

    return documents_embeddings

# %% Function testing
test_embed = embed_file(test_file, context_window)

# %% Cosine Similarity
def cosine_similarity(array):
    pass
