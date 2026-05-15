# %% Imports
from llama_cpp import Llama
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from datetime import datetime
from test_embed import LlamaCppEmbeddingFunction
import pprint

# %% Single Test File
file = "../data/summary/httpsblogapifycomwhatisavectordatabase.md"
with open(file, 'r', encoding='utf-8') as f:
    text = f.read()
    print(type(text))

# %% Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %% Model Params
embed_model_path = "../models/Qwen3-Embedding-8B-Q6_K.gguf"
context_window = 2048
verbose=True

# %% Model
embed_model = Llama(
    model_path=embed_model_path,
    embedding=True,
    n_ctx=context_window,
    verbose=verbose
)

# %% In-Memory Chroma DB
client = chromadb.PersistentClient(path="test_chromadb")
collection = client.get_or_create_collection(
    name="text-splitter-testing",
    embedding_function=LlamaCppEmbeddingFunction(model=embed_model, model_path=embed_model_path),
    metadata={
        "description": "A test DB for trying out the text splitter function",
        "created": str(datetime.now())
    },
    configuration={
        "hnsw": {
            "space": "cosine", # Turns out we don't need that cosine function from earlier
            "ef_construction": 100, # 100 is the default value
            "ef_search": 100, # 100 is the default value
        }
    }
)

# %% Text splitter testing
split_texts = text_splitter.split_text(text)
print(type(split_texts))

# %% Get split texts into ze database
count = 0 # Temporary id naming convention. Will fix later
for item in split_texts:
    count += 1
    collection.add(
        ids=[str(count)],
        documents=[item],
        metadatas=[{"source": file}]
    )

# %% Query results
