# %% Imports
from llama_cpp import Llama
import chromadb
from test_embed import LlamaCppEmbeddingFunction
import pprint

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

# %% ChromaDB
client = chromadb.PersistentClient(path="test_chromadb")
collection = client.get_collection(
    name="text-splitter-testing",
    embedding_function=LlamaCppEmbeddingFunction(model=embed_model, model_path=embed_model_path),
)

# %% Query
query = collection.query(
    query_texts=["What are the main purposes for a vector database?"],
    n_results=10
)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(query)
