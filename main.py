# %% Imports
from llama_cpp import Llama

# %% Model
llm = Llama(
    model_path="models/Qwen3-Embedding-8B-Q6_K.gguf",
    embedding=True,
    n_ctx=20480,)

# %% Open one file for single document embedding
single_doc = "data/summary/httpsawsamazoncomwhatisvectordatabases.md"
with open(single_doc, 'r', encoding='utf-8') as f:
    single_doc_content = f.read()

print(type(single_doc_content))

# %% Create single document embedding
