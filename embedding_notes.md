# Embedding Notes

The starting point for these notes can be found [here](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

An embedding example can be found [here](https://github.com/ggml-org/llama.cpp/discussions/7712).

Let's begin.

# Relevant Embedding Commands

`--pooling {none,mean,cls,last,rank}`

pooling type for embeddings, **use model default if unspecified**
(env: LLAMA_ARG_POOLING)

`--embedding, --embeddings` 

restrict to only support embedding use case; **use only with dedicated embedding models** (default: disabled)
(env: LLAMA_ARG_EMBEDDINGS)

`--embd-gemma-default`

use default EmbeddingGemma model (note: can download weights from the internet)

# Example Commands

From the [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp/tree/master/examples/embedding)

```bash
./llama-embedding -m ./path/to/model --pooling mean --log-disable -p "Hello World!" 2>/dev/null
```

# Qwen3 Embedding

## Sentence Transformers Usage

From [Qwen3 Embedding model page](https://huggingface.co/Qwen/Qwen3-Embedding-8B#sentence-transformers-usage)

```python
# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-8B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.7493, 0.0751],
#         [0.0880, 0.6318]])

```
