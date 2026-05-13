# %% Imports
from llama_cpp import Llama
import chromadb
from main import LlamaCppEmbeddingFunction

"""
THIS NEEDS TO BE RESTRUCTURED.
THE EMBEDDING CLASS SHOULD PROBABLY BE IN IT'S OWN FILE
"""

# %%
model_path = "models/Qwen3-Embedding-8B-Q6_K.gguf"
llm = Llama(model_path=model_path, embedding=True, n_ctx=2048)

# %%
client = chromadb.PersistentClient(path="db")
collection = client.get_collection(
    name="test-collection",
    embedding_function=LlamaCppEmbeddingFunction(model=llm, model_path=model_path)
)

# %%
results = collection.query(
    query_texts=["What are the uses of a Vector database?"],
    n_results=2
)

for key, value in results.items():
    print(key)

print(results['documents'])

# %%
