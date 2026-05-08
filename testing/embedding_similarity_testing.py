# %% Imports
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import islice
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
print()
print(f"`object` key content: {first_doc_embeddings["object"]}")
print(f"`data` key type: {type(first_doc_embeddings["data"])}")
print(f"`model` key content: {first_doc_embeddings["model"]}")
print(f"`usage` key content: {first_doc_embeddings["usage"]}")

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

print(first_vec.shape, "\n", second_vec.shape, "\n")
print(f"First vector:\n{first_vec}\n")
print(f"Second vector:\n{second_vec}\n")

# %% Cosine Similarity
# Eliminate potentially unrelated from the dataset
"""
Use this resource to calculate cosine similarity among all the vectors:
    https://danielcaraway.github.io/html/sklearn_cosine_similarity.html

    Example:
        calculate the entire cosine similarity matrix among X, Y, and Z
        `cos_sim = cosine_similarity([X, Y, Z])`
"""
# Compare the cosine similarity between the two embedded documents

"""
This is undoubtedly incorrect...
Let's do some inspection while referencing this: https://danielcaraway.github.io/html/sklearn_cosine_similarity.html
to try and fix some things
"""

cos_sim = cosine_similarity(first_vec, second_vec)
print(cos_sim)

# %% Calculating cosine similarity for multiple docs
third_doc = "../data/summary/httpsbrainyxcojournaljournal22.md"
with open(third_doc, 'r', encoding='utf-8') as f3:
    third_doc_str = f3.read()
third_doc_embeddings = llm.create_embedding(third_doc_str)
third_vec = np.array([item['embedding'] for item in third_doc_embeddings['data']])
print(third_vec.shape)

# %% Inspect
vectors = [first_vec, second_vec, third_vec]
print("Before array flattening:")
for vec in vectors:
    print(vec.shape)

print("\nAfter array flattening:")

vectors = [v.flatten() for v in vectors]
for vec in vectors:
    print(vec.shape)
# %% Calculate
"""
Outputs the following error:
    ValueError: Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays.

Not sure what that means since all the arrays are shaped the same...
"""
mult_cos_sim = cosine_similarity(vectors)
print(mult_cos_sim)

# %% Save to inspect
