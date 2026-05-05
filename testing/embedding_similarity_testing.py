# Resource: https://codesignal.com/learn/courses/understanding-embeddings-and-vector-representations-3/lessons/comparing-vector-embedding-models-in-python-pgvector
# %% Imports
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np

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
# %% Save to inspect dict structure
os.makedirs("embeddings", exist_ok=True)
with open('embeddings/first_doc_embeddings_dict.json', 'w') as dict1, open('embeddings/second_doc_embeddings_dict.json', 'w') as dict2:
    json.dump(first_doc_embeddings, dict1)
    json.dump(second_doc_embeddings, dict2)

# %% Inspect dict key structure
# We want to target `data`
for key, value in first_doc_embeddings.items():
    print(key)

# %% Convert embeddings to arrays
first_vec = np.array([item['embedding'] for item in first_doc_embeddings['data']])
second_vec = np.array([item['embedding'] for item in second_doc_embeddings['data']])

print(first_vec.shape, "\n", second_vec.shape)

# %% Cosine Similarity
# Compare the cosine similarity between the two embedded documents
