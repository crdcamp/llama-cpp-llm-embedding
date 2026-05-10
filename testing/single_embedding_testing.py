# %% Imports
from llama_cpp import Llama
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import numpy as np

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
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %% Test text split
test_file = "../data/summary/httpswwwdatabrickscomblogwhatisvectordatabase.md"
with open(test_file, 'r', encoding='utf-8') as f:
    # Initial reading and data inspection
    test_content = f.read()
    test_documents = text_splitter.create_documents([test_content])
    print(f"Test documents type: {type(test_documents)}")
    contents = [doc.page_content for doc in test_documents]

    for item in contents:
        embeddings = llm.create_embedding(item)
        print(f"Embeddings type: {type(embeddings)}")
