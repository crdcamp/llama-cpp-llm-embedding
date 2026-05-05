# Resource: https://codesignal.com/learn/courses/understanding-embeddings-and-vector-representations-3/lessons/comparing-vector-embedding-models-in-python-pgvector
# %% Imports
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# %% Model
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    n_ctx=40960,)

# %% Read documents
first_doc = "../data/summary/httpsawsamazoncomwhatisvectordatabases.md"
second_doc = "../data/summary/httpsblogapifycomwhatisavectordatabase.md"
with open(first_doc, 'r', encoding='utf-8') as f1, open(second_doc, 'r', encoding='utf-8') as f2:
    first_doc_str = f1.read()
    second_doc_str = f2.read()

# %% Embed both files
first_doc_embeddings = llm.create_embedding(first_doc_str)
second_doc_embeddings = llm.create_embedding(second_doc_str)

# %% Inspect embeddings type
print(type(first_doc_embeddings), "\n", type(second_doc_embeddings))
# %% Save dictionaries to inspect dict structure
os.makedirs("testing", exist_ok=True)
with open('testing/first_doc_embeddings_dict.json', 'w') as dict1, open('testing/second_doc_embeddings_dict.json', 'w') as dict2:
    json.dump(first_doc_embeddings, dict1)
    json.dump(second_doc_embeddings, dict2)

# %% Cosine Similarity
# Compare the cosine similarity between the two embedded documents
