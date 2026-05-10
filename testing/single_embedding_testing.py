# %% Imports
from llama_cpp import Llama
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import numpy as np

# %% Test file
test_file = "../data/summary/httpswwwdatabrickscomblogwhatisvectordatabase.md"

# %% Model
context_length = 40960
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    n_ctx=context_length,
    n_batch=context_length # IN ACTUAL USE CASE: Leave this at 512 and encode the text using batches instead
)

# %% Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %% Test text split
with open(test_file, 'r', encoding='utf-8') as f:
    # Initial reading and data inspection
    test_content = f.read()
    test_documents = text_splitter.create_documents([test_content])
    print(f"Test documents type: {type(test_documents)}")

    contents = [doc.page_content for doc in test_documents]
    print(f"Contents type: {type(contents)}")

    for item in contents:
        embeddings = llm.create_embedding(item)
        print(f"Embeddings type: {type(embeddings)}")

# %% Conversion to function
def embed_file(file):
    start_time = time.perf_counter()
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        split_text = text_splitter.create_documents([text])
        documents = [doc.page_content for doc in split_text]
        print(type(documents))

        for doc in documents:
            embeddings = llm.create_embedding(doc)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Documents embedded in {elapsed_time:.2f} seconds")

test_embed = embed_file(test_file)
