# %% Imports
from llama_cpp import Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from datetime import datetime
import os
from embed import LlamaCppEmbeddingFunction

# %% Model Params
embed_model_path = "models/Qwen3-Embedding-8B-Q6_K.gguf"
context_window = 2048
verbose=True

# %% Model
embed_model = Llama(
    model_path=embed_model_path,
    embedding=True,
    n_ctx=context_window,
    verbose=verbose
)

# %% Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %% Chroma DB Setup
db_path = "chromadb"
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection(
    name="test-collection",
    embedding_function=LlamaCppEmbeddingFunction(model=embed_model, model_path=model_path),
    metadata={
        "description": "A test collection for learning ChromaDB",
        "created": str(datetime.now())
    },
    # More info on configuration: https://docs.trychroma.com/docs/collections/configure#what-is-an-hnsw-index
    configuration={
        "hnsw": {
            "space": "cosine", # Turns out we don't need that cosine function from earlier
            "ef_construction": 100, # 100 is the default value
            "ef_search": 100, # 100 is the default value
        }
    }
)
