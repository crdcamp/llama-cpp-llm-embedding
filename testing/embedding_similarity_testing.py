# Resource: https://codesignal.com/learn/courses/understanding-embeddings-and-vector-representations-3/lessons/comparing-vector-embedding-models-in-python-pgvector
# %% Imports
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time

# %% Model
context_length = 40960
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    n_ctx=context_length,
    n_batch=context_length # IN ACTUAL USE CASE: Leave this at 512 and encode the text using batches instead
)

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

# %% Inspect dict key structure
# We want to target `data`
for key, value in first_doc_embeddings.items():
    print(key)

# %% Verify that token usage doesn't exceed context window
print("First doc embeddings usage: ", first_doc_embeddings['usage']['total_tokens'])
print("Second doc embeddings usage: ", second_doc_embeddings['usage']['total_tokens'])
print("Context length: ", context_length, "\n")

first_doc_tok_usage = first_doc_embeddings['usage']['total_tokens']
second_doc_tok_usage = second_doc_embeddings['usage']['total_tokens']

if first_doc_tok_usage > context_length or second_doc_tok_usage > context_length:
    print("Context window: EXCEEDED")
else:
    print("Context window: NOT EXCEEDED")

# %% Convert embeddings to arrays
# Using list comprehension for later. Not necessary here when we're dealing with single documents
first_vec = np.array([item['embedding'] for item in first_doc_embeddings['data']])
second_vec = np.array([item['embedding'] for item in second_doc_embeddings['data']])

print(first_vec.shape, "\n", second_vec.shape)

# %% Cosine Similarity
"""
Use this resource to calculate cosine similarity among all the vectors:
    https://danielcaraway.github.io/html/sklearn_cosine_similarity.html

    Example:
        calculate the entire cosie similarity matrix among X, Y, and Z
        `cos_sim = cosine_similarity([X, Y, Z])`
"""
# Compare the cosine similarity between the two embedded documents
cos_sim = cosine_similarity(first_vec, second_vec)
print(cos_sim)

# %% Embedding all documents and calculating cosine similarity
docs_dir = "../data/summary"
times = []

for file in os.listdir(docs_dir):
    if not file.endswith('.md'):
        continue

    path = os.path.join(docs_dir, file)
    start_time = time.perf_counter()

    try:
        with open(path, 'r', encoding='utf-8') as f:
            print(f"Processing file: {file}")
            content = f.read()
            embedding = llm.create_embedding(content)
            embedding_token_usage = embedding['usage']['total_tokens']

            if embedding_token_usage <= context_length:
                embedding_vector = np.array([item['embedding'] for item in embedding['data']])
                print(f"{file} successfully converted to vector")

            else:
                print(f"Error - Embedding exceeded context length: {embedding_token_usage} > {context_length}. Skipping...")

    except Exception as e:
        print(f"Failed to process {file}: {e}")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"Processed {file} in {total_time:.2f} seconds\n\n")
    times.append(total_time)

print(f"Total embedding time: {sum(times)}")
