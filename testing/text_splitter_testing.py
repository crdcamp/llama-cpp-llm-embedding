# %% Imports
from llama_cpp import Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from test_embed import LlamaCppEmbeddingFunction

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

# %% Text splitter function testing
split_texts = text_splitter.split_text(text)
print(type(split_texts))
print(split_texts)

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
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="text-splitter-testing",
    embedding_function=LlamaCppEmbeddingFunction(model=embed_model, model_path=embed_model_path),
    configuration={
        "hnsw": {
            "space": "cosine", # Turns out we don't need that cosine function from earlier
            "ef_construction": 100, # 100 is the default value
            "ef_search": 100, # 100 is the default value
        }
    }
)
