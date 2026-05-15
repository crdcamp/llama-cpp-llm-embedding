# %% Imports
from llama_cpp import Llama
from embed import LlamaCppEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from datetime import datetime
import os

# %% Model Params
embed_model_path = "../../models/Qwen3-Embedding-8B-Q6_K.gguf"
context_window = 2048
verbose=True

# %% Model
embed_model = Llama(
    model_path=embed_model_path,
    embedding=True,
    n_ctx=context_window,
    verbose=verbose
)

# %% Chroma DB
db_path = "chromadb"
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(
    name="RAG and Vector Databases",
    embedding_function=LlamaCppEmbeddingFunction(model=embed_model, model_path=embed_model_path),
    metadata={
        "description": "My first vector database for learning RAG retrieval",
        "created": str(datetime.now())
    },
    configuration={
        "hnsw": {
            "space": "cosine", # Turns out we don't need that cosine function from earlier
            "ef_construction": 100, # 100 is the default value
            "ef_search": 100, # 100 is the default value
        }
    }
)

# %% Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %% Create embeddings for each file
documents_dir = "data/summary"
for doc in os.listdir(documents_dir):
    doc_path = os.path.join(documents_dir, doc)
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()
        split_texts = text_splitter.split_text(text)
        for item in split_texts:
            collection.upsert(
                ids=[],
                documents=[text],
                metadatas={"source": doc_path}
            )
