# %%
# %pip install chromadb google-generativeai pandas python-dotenv

# %%
import json
import pandas as pd
import chromadb
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
import copy


# %%
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# %%
def flatten_metadata(meta_dict):
    """
    Converts any dict or list values in a metadata dictionary to JSON strings,
    as required by ChromaDB.
    """
    flat_meta = {}
    for key, value in meta_dict.items():
        if isinstance(value, (dict, list)):
            flat_meta[key] = json.dumps(value)
        else:
            flat_meta[key] = value
    return flat_meta

# %%
def load_and_prepare_docs(filepath="codebase_map.jsonl", max_lines=50,overlap_lines=5):
    """
    Loads the JSONL file and formats each entry for embedding.
    If a function's body exceeds max_lines, it's split into a parent document
    (signature only) and multiple child documents (body chunks).
    """
    documents = []
    metadata = []
    ids = []
    doc_counter = 1

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            content = ""

            # Handle Structs
            if data['type'] == 'struct':
                struct = data['struct']
                fields_str = "\n".join([f"  {field['name']} {field['type']} `{field.get('tag', '')}`" for field in struct['fields']])
                content = f"File: {data['file_path']}\nType: struct\nName: {struct['name']}\nFields:\n{fields_str}"
                documents.append(content)
                metadata.append(flatten_metadata(data)) # FLATTEN METADATA
                ids.append(str(doc_counter))
                doc_counter += 1

            # Handle Functions
            elif data['type'] == 'function':
                func = data['function']
                body_lines = func.get('body', '').split('\n')

                # If function body is larger than max_lines, chunk it
                if len(body_lines) > max_lines:
                    parent_content = f"File: {data['file_path']}\nType: function\nSignature: {func['signature']}\nSummary: This is a large function with its body broken into smaller chunks."
                    parent_id = str(doc_counter)
                    documents.append(parent_content)
                    
                    parent_meta = copy.deepcopy(data)
                    parent_meta['function']['body'] = "# BODY CHUNKED, SEE CHILD DOCUMENTS #"
                    metadata.append(flatten_metadata(parent_meta))
                    ids.append(parent_id)
                    doc_counter += 1

                    # 2. MODIFIED: Create overlapping child documents
                    step_size = max_lines - overlap_lines
                    chunk_num = 1
                    for i in range(0, len(body_lines), step_size):
                        # Define the chunk with overlap
                        chunk_text = "\n".join(body_lines[i:i + max_lines])
                        if not chunk_text: continue # Skip empty chunks

                        chunk_content = (
                            f"File: {data['file_path']}\n"
                            f"Type: function_chunk\n"
                            f"Parent Function: {func['signature']}\n"
                            f"Chunk {chunk_num}:\n---\n{chunk_text}"
                        )
                        documents.append(chunk_content)
                        child_meta = {
                            "file_path": data['file_path'],
                            "parent_function_name": func['name'],
                            "is_chunk": True
                        }
                        metadata.append(child_meta)
                        ids.append(f"{parent_id}_{chunk_num-1}")
                        doc_counter += 1
                        chunk_num += 1
                else:
                    # If function is not too long, process it normally
                    content = f"File: {data['file_path']}\nType: function\nSignature: {func['signature']}\nBody: {func['body']}"
                    documents.append(content)
                    metadata.append(flatten_metadata(data)) # FLATTEN METADATA
                    ids.append(str(doc_counter))
                    doc_counter += 1

    return documents, metadata, ids

print("Loading and preparing documents with hierarchical chunking...")
documents, metadata, ids = load_and_prepare_docs()
print(f"Loaded and processed into {len(documents)} documents.")

print(documents[68])

# %% [markdown]
# Embed and Store in VectorDB

# %%

"""Embed and Store in VectorDB"""

# Initialize ChromaDB client.
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "gocodebase_chunked" # Using a new name for the chunked data

# Delete the collection if it already exists to ensure a fresh start
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)
    print(f"Deleted existing collection: '{collection_name}'")

# Create a new, empty collection
collection = client.create_collection(name=collection_name)
print(f"Created a new collection: '{collection_name}'")

print("Embedding and indexing the codebase... This may take a moment.")
# Embed the documents in batches
batch_size = 50 # Increased batch size for efficiency
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    batch_meta = metadata[i:i+batch_size]

    # Using Google's embedding model
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=batch_docs,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = response['embedding']

    collection.add(
        embeddings=embeddings,
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )
    print(f"Indexed batch {i//batch_size + 1} of {len(documents)//batch_size + 1}...")
    time.sleep(1) # API rate limiting

print("Codebase successfully indexed in ChromaDB.")
item_count = collection.count()
print(f"The collection now has {item_count} items.")

# %%
# Add this cell to verify the contents of your ChromaDB collection
client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection(name="gocodebase_chunked")
    item_count = collection.count()
    print(f"The collection '{collection.name}' has {item_count} items.")

    if item_count > 0:
        print("\nHere's a sample of the data in the collection:")
        # Peek at the first 2 items to ensure they look correct
        sample = collection.peek(limit=2)
        print(sample['documents'])
except ValueError:
    print("The collection 'gocodebase_chunked' does not exist. Please run the indexing cell first.")

# %%
from IPython import embed
import google.generativeai as genai

# Make sure your key is configured via one of the methods above
# genai.configure(api_key="...")

# This loop will print available models if your key is valid
# It will fail with the same error if the key is still invalid
print("Verifying API Key by listing available models:")
embedContentList = []
generateContentList = []
for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    embedContentList.append(m.name)
  if "generateContent" in m.supported_generation_methods:
    generateContentList.append(m.name)

print("\nembedContentList: ", embedContentList)
print("\ngenerateContentList: ", generateContentList)

# %% [markdown]
# Take request, find relevent snippets from chroma nad pass to generative model

# %%

"""Take request, find relevent snippets from chroma nad pass to generative model"""

def query_rag(query: str, n_results: int = 5):
    """Performs the RAG process: query -> retrieve -> augment -> generate."""

    # 1. Retrieve relevant code snippets
    query_embedding_response = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="RETRIEVAL_QUERY"
    )

    results = collection.query(
        query_embeddings=[query_embedding_response['embedding']],
        n_results=n_results
    )

    retrieved_docs = results['documents'][0]
    context = "\n---\n".join(retrieved_docs)

    # 2. Augment: Create a prompt for the generative model
    prompt = f"""You are an expert Go programmer. Your task is to help a user modify their codebase.
Use the following relevant code snippets from the codebase as context to provide a complete and accurate answer.
Some snippets might be chunks of larger functions, indicated by 'Type: function_chunk'. Use the parent function signature to understand the context.

**CONTEXT FROM THE CODEBASE:**
---
{context}
---

**USER'S REQUEST:**
"{query}"

**YOUR TASK:**
Based on the user's request and the provided context, generate the necessary code changes.
- If a struct needs modification, show the new struct definition.
- If a function needs to be changed, provide the complete, updated function body.
- If new functions are needed, write them.
- Provide a brief, clear explanation of the changes you made.
- Present the final output in Go code blocks.
"""

    # 3. Generate the response
    model = genai.GenerativeModel('gemini-2.0-flash') # Using a more recent model
    response = model.generate_content(prompt)

    return response.text

# %% [markdown]
# Giving request
# 

# %%
"""Giving request"""

user_request = """
I need to add a 'likes' count to the Chirp model.
It should be an integer and default to 0.

Then, update the 'handlerChirpsCreate' function. After creating a chirp,
the response should include this new 'likes' field.
"""

from IPython.display import display, Markdown

# Get the suggested code change
suggested_change = query_rag(user_request)

# Create the full markdown string and display it
# The f-string combines the header and the response into a single markdown block
markdown_output = f"""
---
### SUGGESTED CODE CHANGE
---
{suggested_change}
"""

display(Markdown(markdown_output))


