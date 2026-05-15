# %% Imports
from llama_cpp import Llama
from embed import LlamaCppEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from datetime import datetime
import os
import uuid

# %% Model Params
embed_model_path = "models/Qwen3-Embedding-8B-Q6_K.gguf"
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

def get_or_create_collection(name: str, space: str, ef_construction: int, ef_search: int):
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=LlamaCppEmbeddingFunction(model=embed_model, model_path=embed_model_path),
        metadata={
            "description": "My first vector database for learning RAG retrieval",
            "created": str(datetime.now())
        },
        # I still have to do some testing to figure out the best configuration here
        # Parameters can be found here: https://docs.trychroma.com/docs/collections/configure#what-is-an-hnsw-index
        configuration={
            "hnsw": {
                "space": space, # Turns out we don't need that cosine function from earlier
                "ef_construction": ef_construction, # 100 is the default value
                "ef_search": ef_search, # 100 is the default value
            }
        }
    )

    return collection

l2_collection  = get_or_create_collection("l2-norm-collection", "l2", 100, 100)
inner_product_collection = get_or_create_collection("ip-collection", "ip", 100, 100)
cosine_collection = get_or_create_collection("cosine-sim-collection", "cosine", 100, 100)

collections_list = cosine_collection, l2_collection, cosine_collection`


# %% Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    length_function=len,
    is_separator_regex=False,
)

# %% Create embeddings for each file
# I'm uncertain if this is a good way to setup the db,
# but hey it's my first time
documents_dir = "data/summary"
for doc in os.listdir(documents_dir):
    doc_path = os.path.join(documents_dir, doc)
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()
        split_texts = text_splitter.split_text(text)
        for item in split_texts:
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_path}:{item}"))
            collection.upsert(
                ids=[chunk_id], # There might something better than UUIDs here, but we'll stick with them for now
                documents=[item],
                metadatas=[{"source": doc_path, "created": str(datetime.now())}]
            )
