# %% Imports
from chromadb.utils.embedding_functions import register_embedding_function
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from llama_cpp import Llama
import chromadb

# %%
@register_embedding_function
class LlamaCppEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for llama.cpp to
    interact with Chroma DB
    """
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

client = chromadb.Client()
collection = client.get_or_create_collection(
    name="text-splitter-testing",
    embedding_function=LlamaCppEmbeddingFunction,
    configuration={
        "hnsw": {
            "space": "cosine", # Turns out we don't need that cosine function from earlier
            "ef_construction": 100, # 100 is the default value
            "ef_search": 100, # 100 is the default value
        }
    }
)
