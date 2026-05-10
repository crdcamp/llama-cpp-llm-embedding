# %% Imports
from llama_cpp import Llama
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
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
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %% Split into chunks and embed
summary_dir = "../data/summary"
start_time = time.perf_counter()
for file in os.listdir(summary_dir):
    if file.endswith('.md'):
        path = os.path.join(summary_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            documents_embeddings = []
            documents = text_splitter.create_documents([text])
            print(type(documents))

            # for doc in documents:
            #     embeddings = llm.create_embedding(doc.page_content) # For some reason list comprehension won't work for this function
            #     documents_embeddings.extend(
            #         [
            #             (document, embeddings['embeddding'])
            #             for document, embeddings in zip(doc, embeddings['data'])
            #         ]
            #     )


end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"\nEmbedded documents in {elapsed_time:.2f} seconds")


# # Saving this for later
# #array = np.array([item['embedding'] for item in embeddings['data']])
# embedding_token_usage = embeddings['usage']['total_tokens']
# if embedding_token_usage <= context_length:
#     try:
#         embedding_vector = np.array([item["embedding"] for item in embeddings["data"]]).flatten()
#         print(f"Successfully converted {file} to vector with shape: {embedding_vector.shape}")
#     except Exception as e:
#         print(f"Error processing file {file}: {e}. Skipping...")
# else:
#     continue
