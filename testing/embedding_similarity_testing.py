# Resource: https://codesignal.com/learn/courses/understanding-embeddings-and-vector-representations-3/lessons/comparing-vector-embedding-models-in-python-pgvector
# %% Imports
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity

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
embeddings = llm.create_embedding([first_doc_str, second_doc_str])

# %% Cosine Similarity
# Compare the cosine similarity between the two embedded documents
