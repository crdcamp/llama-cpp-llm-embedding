# %% Imports
from llama_cpp import Llama
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# %% Model
context_length = 40960
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    n_ctx=context_length,
    n_batch=context_length # IN ACTUAL USE CASE: Leave this at 512 and encode the text using batches instead
)

# %% Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
# %% Chunk function
def chunk():
    pass

"""
DO NOT RUN THIS WITHOUT THE CHUNK FUNCTION.
YOU WILL FREEZE EVERYTHING!
"""
# %% Open documents
summary_dir = "../data/summary"
for file in os.listdir(summary_dir):
    if file.endswith('.md'):
        path = os.path.join(summary_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

            # Embedding
            embedding = llm.create_embedding(content)
            embedding_token_usage = embedding['usage']['total_tokens']

            if embedding_token_usage <= context_length:
                try:
                    embedding_vector = np.array([item["embedding"] for item in embedding["data"]]).flatten()
                    print(f"Successfully converted {file} to vector with shape: {embedding_vector.shape}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}. Skipping...")
            else:
                continue
