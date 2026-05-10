# %% Imports
from llama_cpp import Llama

# %% Model
llm = Llama(
    model_path="../models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    n_ctx=40960,)

# %% Open one file for single document embedding
first_doc = "../data/summary/httpsawsamazoncomwhatisvectordatabases.md"
with open(first_doc, 'r', encoding='utf-8') as f:
    first_doc_content = f.read()

print(type(first_doc_content))

# %% Create single document embedding
embeddings = llm.create_embedding(first_doc_content)

# %% Inspecting for assigning to variable
print(type(embeddings))
for key, value in embeddings.items():
    print(key)
print(embeddings)


# %% Read second document
second_doc = "../data/summary/httpsblogapifycomwhatisavectordatabase.md"
with open(second_doc, 'r', encoding='utf-8') as f2:
    second_doc_content = f2.read()

print(type(second_doc_content))

# %% Multiple document embedding
embeddings = llm.create_embedding([first_doc_content, second_doc_content])
