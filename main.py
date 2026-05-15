# %% Imports
from llama_cpp import Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from typing import Dict, Any
from chromadb.utils.embedding_functions import register_embedding_function
from chromadb import Documents, EmbeddingFunction, Embeddings
from datetime import datetime
import os

"""
USE THIS: https://docs.langchain.com/oss/python/integrations/splitters/split_html#using-htmlsemanticpreservingsplitter

WE'LL ADJUST THIS SO WE DONT HAVE TO PUT ALL
THIS IN A MAIN FUNCTION LATER

Also I think the IDs in the db should be assigned
as random numbers
"""

# %% Model Params
context_window = 2048
model_path = "models/Qwen3-Embedding-8B-Q6_K.gguf"

# %% Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %% Custom embedding function for llama cpp
@register_embedding_function
class LlamaCppEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model, model_path: str):
        self.model = model
        self.model_path = model_path

    def __call__(self, input: Documents) -> Embeddings:
        result = self.model.create_embedding(list(input))
        return [item['embedding'] for item in result['data']]

    @staticmethod
    def name() -> str:
        return "llama-cpp-embed-chroma"

    def get_config(self) -> Dict[str, Any]:
        return dict(model_path=self.model_path)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "LlamaCppEmbeddingFunction":
        model = Llama(model_path=config['model_path'], embedding=True)
        return LlamaCppEmbeddingFunction(model=model, model_path=config['model_path'])


if __name__ == "__main__":
    # %% Load Model
    llm = Llama(
        model_path=model_path,
        embedding=True,
        verbose=True,
        n_ctx=context_window,
        n_batch=context_window # Need to double check if this is a good idea... Probably not a good idea...
    )

    # %% Initialize ChromaDB
    db_path = "db"
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="test-collection",
        embedding_function=LlamaCppEmbeddingFunction(model=llm, model_path=model_path),
        metadata={
            "description": "A test collection for learning ChromaDB",
            "created": str(datetime.now())
        },
        # More info on configuration: https://docs.trychroma.com/docs/collections/configure#what-is-an-hnsw-index
        configuration={
            "hnsw": {
                "space": "cosine", # Turns out we don't need that cosine function from earlier
                "ef_construction": 100, # 100 is the default value
                "ef_search": 100, # 100 is the default value
            }
        }
    )

    # %% Open docs dir
    documents_dir = "data/summary"
    for doc in os.listdir(documents_dir):
        doc_path = os.path.join(documents_dir, doc)
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
            collection.upsert(
                ids=[doc_path],
                documents=[text],
                metadatas={"source": doc_path}
            )
