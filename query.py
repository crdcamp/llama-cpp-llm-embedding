# %% Imports
from llama_cpp import Llama
import chromadb
from main import LlamaCppEmbeddingFunction
import pprint
pp = pprint.PrettyPrinter(indent=4)

"""
NEED TO ADD TEXT SPLITTER TO EMBED FUNCTION
THIS NEEDS TO BE RESTRUCTURED.
THE EMBEDDING CLASS SHOULD PROBABLY BE IN IT'S OWN FILE
"""

# %%
model_path = "models/Qwen3-Embedding-8B-Q6_K.gguf"
llm = Llama(model_path=model_path, embedding=True, n_ctx=2048)

# %%
client = chromadb.PersistentClient(path="db")

# %%
collection = client.get_collection(
    name="test-collection",
    embedding_function=LlamaCppEmbeddingFunction(model=llm, model_path=model_path)
)

# %%
query = "What are the benefits of creating a Vector database?"
results = collection.query(
    query_texts=[query],
    include=["metadatas", "distances"],
    n_results=1
)

# %%
print(pp.pprint(results))

# %% Get the most similar document
