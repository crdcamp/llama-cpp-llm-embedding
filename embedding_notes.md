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

# llama.cpp Docs

From [this link](https://llama-cpp-python.readthedocs.io/en/latest/#embeddings).

## Speculative Decoding

`llama-cpp-python` supports speculative decoding which allows the model to generate completions based on a draft model.

The fastest way to use speculative decoding is through the LlamaPromptLookupDecoding class.

Just pass this as a draft model to the Llama class during initialization.

```python
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

llama = Llama(
    model_path="path/to/model.gguf",
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10) # num_pred_tokens is the number of tokens to predict 10 is the default and generally good for gpu, 2 performs better for cpu-only machines.
)
```

## Embeddings

To generate text embeddings use `create_embedding` or `embed`. Note that you must pass `embedding=True` to the constructor upon model creation for these to work properly.

```python
import llama_cpp

llm = llama_cpp.Llama(model_path="path/to/model.gguf", embedding=True)

embeddings = llm.create_embedding("Hello, world!")

# or create multiple embeddings at once

embeddings = llm.create_embedding(["Hello, world!", "Goodbye, world!"])
```

# Qwen3 Embedding

## Sentence Transformers Usage

From [Qwen3 Embedding model page](https://huggingface.co/Qwen/Qwen3-Embedding-8B#sentence-transformers-usage)

**Note:** This will not work with GGUF file formats, but serves as a good outline.

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
