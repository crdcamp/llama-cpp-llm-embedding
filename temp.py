# %% Imports
from langchain_text_splitters.base import TokenTextSplitter

# %%
file = "data/summary/httpsblogapifycomwhatisavectordatabase.md"

# %% Text Splitter
text_splitter = TokenTextSplitter(
    chunk_size=2048,
    chunk_overlap=100,
)

# %% Show
with open(file, 'r', encoding='utf-8') as f:
    text = f.read()
    text_tokens = llm.tokenize(text, add_bos=False)
    split_text = text_splitter.create_documents([text])
    documents = [doc.page_content for doc in split_text]
    print(documents)
