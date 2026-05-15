# %% Imports
from llama_cpp import Llama
import chromadb
from embed import LlamaCppEmbeddingFunction
import pprint

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

# %% ChromaDB
db_path = "chromadb"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(
    name="vector-db",
    embedding_function=LlamaCppEmbeddingFunction(model=embed_model, model_path=embed_model_path),
)

# %% Query
input_query = "What are the main purposes for a vector database?"
query_results = collection.query(
    query_texts=[input_query],
    n_results=20
)

print("\n\nINPUT QUERY: ", input_query)

print("QUERY RESULT KEYS:")
for key, vaklue in query_results.items():
    print(key)

print("\nQUERY RESULTS:")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(query_results)

# %% Compare the results of each method: `cosine`,`l2`, and `ip`
