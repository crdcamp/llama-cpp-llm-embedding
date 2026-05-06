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

"""
THIS DESPARATELY NEEDS A FUNCTION TO SPLIT
THE PROCESSING INTO CHUNKS
"""
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
    times.append(total_time)
    print(f"Processed {file} in {total_time:.2f} seconds\n\n")

print("ALL DONE!")
print(f"Total embedding time: {sum(times):.2f} seconds")
