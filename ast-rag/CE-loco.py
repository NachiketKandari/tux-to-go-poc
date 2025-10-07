# %% [markdown]
# # Local RAG Pipeline with Local Embeddings and Generation
#
# This script sets up a Retrieval-Augmented Generation (RAG) pipeline that runs entirely on your local machine.
# - **Embedding Model**: `Qwen/Qwen3-Embedding-0.6B` running via `sentence-transformers`.
# - **Vector Database**: ChromaDB for storing and retrieving document embeddings.
# - **Generation Model**: Llama 3.2 (or any other model) running via Ollama.

# %% [markdown]
# ## 1. Setup and Installations
# Install the necessary libraries for the pipeline. `sentence-transformers` and `torch` are added to run the local embedding model.

# %%
# For running generation via local model
%pip install ollama
# For running local embedding model
%pip install sentence-transformers torch
# Core libraries for RAG
%pip install chromadb pandas python-dotenv

# %%
import json
import pandas as pd
import chromadb
import time
import os
import copy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import ollama

# %% [markdown]
# ## 2. Load and Prepare Documents
# These functions parse a JSONL file containing codebase information, format it for embedding, and handle chunking for large functions.

# %%
def flatten_metadata(meta_dict):
    """
    Converts any dict or list values in a metadata dictionary to JSON strings,
    as required by ChromaDB.
    """
    flat_meta = {}
    for key, value in meta_dict.items():
        if isinstance(value, (dict, list)):
            try:
                # Attempt to serialize to JSON string
                flat_meta[key] = json.dumps(value)
            except TypeError:
                # Fallback for non-serializable objects
                flat_meta[key] = str(value)
        elif value is None:
            continue # Skip None values
        else:
            flat_meta[key] = value
    return flat_meta

# %%
def load_and_prepare_docs(filepath="codebase_map.jsonl", max_lines=50, overlap_lines=5):
    """
    Loads the JSONL file and formats each entry for embedding.
    Handles all code types: structs, functions, imports, constants, variables, and interfaces.
    If a function's body exceeds max_lines, it's split into a parent document
    and multiple child documents (body chunks).
    """
    documents = []
    metadata = []
    ids = []
    doc_counter = 1

    # Check if the file exists before trying to open it
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure the codebase map file is in the correct directory.")
        return [], [], []

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            content = ""
            doc_type = data['type']

            # Use a single metadata object for each simple entry
            current_meta = flatten_metadata(data)

            # Handle Structs
            if doc_type == 'struct':
                struct = data['struct']
                fields_str_parts = []
                for field in struct.get('fields', []):
                    tag_str = f"`{field.get('tag', '')}`" if field.get('tag') else ""
                    fields_str_parts.append(f"  {field.get('name')} {field.get('type')} {tag_str}".strip())
                fields_str = "\n".join(fields_str_parts)
                content = f"File: {data['file_path']}\nType: struct\nName: {struct['name']}\nFields:\n{fields_str}"

            # Handle Imports
            elif doc_type == 'import':
                imp = data['import']
                alias_str = f" as {imp['name']}" if imp.get('name') else ""
                content = f"File: {data['file_path']}\nType: import\nStatement: import {imp['path']}{alias_str}"

            # Handle Constants and Variables
            elif doc_type in ['constant', 'variable']:
                spec = data[doc_type]
                names_str = ", ".join(spec.get('names', []))
                type_str = f"\nType: {spec['type']}" if spec.get('type') else ""
                value_str = f"\nValue: {spec['value']}" if spec.get('value') else ""
                content = f"File: {data['file_path']}\nDeclaration: {doc_type}\nName(s): {names_str}{type_str}{value_str}"

            # Handle Interfaces
            elif doc_type == 'interface':
                interface = data['interface']
                methods_str = "\n".join([f"  {method['signature']}" for method in interface.get('methods', [])])
                content = f"File: {data['file_path']}\nType: interface\nName: {interface['name']}\nMethods:\n{methods_str}"

            # Handle Functions
            elif doc_type == 'function':
                func = data['function']
                body_lines = func.get('body', '').split('\n')

                if len(body_lines) > max_lines:
                    # Parent document for a large function
                    parent_content = f"File: {data['file_path']}\nType: function\nSignature: {func['signature']}\nSummary: This is a large function with its body broken into smaller chunks."
                    parent_id = str(doc_counter)
                    documents.append(parent_content)

                    parent_meta = copy.deepcopy(data)
                    parent_meta['function']['body'] = "# BODY CHUNKED, SEE CHILD DOCUMENTS #"
                    metadata.append(flatten_metadata(parent_meta))
                    ids.append(parent_id)
                    doc_counter += 1

                    # Child documents for each chunk
                    step_size = max_lines - overlap_lines
                    chunk_num = 1
                    for i in range(0, len(body_lines), step_size):
                        chunk_text = "\n".join(body_lines[i:i + max_lines])
                        if not chunk_text.strip(): continue

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
                            "is_chunk": True,
                            "chunk_number": chunk_num,
                            "parent_id": parent_id
                        }
                        metadata.append(child_meta)
                        ids.append(f"{parent_id}_{chunk_num}")
                        doc_counter += 1
                        chunk_num += 1
                else:
                    # Process normal-sized functions
                    content = f"File: {data['file_path']}\nType: function\nSignature: {func['signature']}\nBody: {func['body']}"

            # For all non-chunked types, add the document
            if content:
                documents.append(content)
                metadata.append(current_meta)
                ids.append(str(doc_counter))
                doc_counter += 1

    return documents, metadata, ids

# --- Main execution ---
print("Loading and preparing documents...")
# NOTE: Make sure you have a 'codebase_map.jsonl' file in the same directory.
documents, metadata, ids = load_and_prepare_docs(filepath="codebase_map.jsonl")

if documents:
    print(f"Loaded and processed {len(documents)} documents.")
    # Example: Print a few documents to see the formats
    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Document {i+1} ---\n{doc}")
else:
    print("No documents were loaded. Please check the file path and content.")

# %% [markdown]
# ## 3. Initialize Local Embedding Model
# We load the `Qwen/Qwen3-Embedding-0.6B` model from Hugging Face. This model will run on your local machine to generate embeddings.

# %%
print("Initializing local embedding model... This might take a moment.")
# Using trust_remote_code=True is required for some models on Hugging Face.
embedding_model = SentenceTransformer(
    'Qwen/Qwen3-Embedding-0.6B',
    trust_remote_code=True
)
print("Embedding model loaded successfully.")

# %% [markdown]
# ## 4. Embed and Store in VectorDB
# This cell takes the prepared documents, generates embeddings for them using the local Qwen model, and stores them in a local ChromaDB collection.

# %%
if documents:
    # Initialize ChromaDB client.
    client = chromadb.PersistentClient(path="./chroma_db_local")
    collection_name = "gocodebase_local_qwen"

    # Delete the collection if it already exists to ensure a fresh start
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: '{collection_name}'")

    collection = client.create_collection(name=collection_name)
    print(f"Created a new collection: '{collection_name}'")

    print("Embedding and indexing the codebase with local model... This may take a while.")
    batch_size = 20 # Using a smaller batch size for local processing
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]

        # Generate embeddings locally
        embeddings = embedding_model.encode(
            batch_docs,
            normalize_embeddings=True # Normalizing is good practice for retrieval
        ).tolist() # Convert to list for ChromaDB

        collection.add(
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"Indexed batch {i//batch_size + 1} of {len(documents)//batch_size + 1}...")

    print("Codebase successfully indexed in ChromaDB.")
    item_count = collection.count()
    print(f"The collection now has {item_count} items.")
else:
    print("Skipping embedding process as no documents were loaded.")

# %% [markdown]
# ## 5. Query the RAG Pipeline
# The `query_rag` function orchestrates the process:
# 1. Takes a user query.
# 2. Generates an embedding for the query using the local Qwen model.
# 3. Retrieves relevant documents from ChromaDB.
# 4. Constructs a detailed prompt with the retrieved context.
# 5. Sends the prompt to a local LLM (e.g., Llama 3.2) via Ollama to generate the final response.

# %%
def query_rag(query: str, n_results: int = 5):
    """Performs the RAG process using local models."""

    # 1. Retrieve relevant code snippets using the local embedding model
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    except NameError:
        return "The ChromaDB collection has not been initialized. Please run the embedding cell first."
    except Exception as e:
        return f"An error occurred during ChromaDB query: {e}"


    retrieved_docs = results['documents'][0]
    context = "\n---\n".join(retrieved_docs)

    # 2. Augment: Create a prompt for the local generative model
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
- If a const needs modification, show the new const.
- If a function needs to be changed, provide the complete, updated function body.
- If new functions are needed, write them.
- Provide a brief, clear explanation of the changes you made.
- Present the final output in Go code blocks.
"""
    # print("--- PROMPT SENT TO LLAMA ---")
    # print(prompt)
    # print("--------------------------")

    # 3. Generate the response from the local model via Ollama
    print("Sending request to local Llama3.2 model...")
    try:
        response = ollama.chat(
            model='llama3.2', # Make sure this model is available in Ollama
            messages=[
                {'role': 'user', 'content': prompt},
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama: {e}\n\nPlease ensure Ollama is running and the 'llama3.2' model is installed. You can run 'ollama run llama3.2' in your terminal."


# %% [markdown]
# ## 6. Give a Request and Get a Response
# Now, we provide a request to modify the codebase and display the generated response from the local RAG pipeline.

# %%
from IPython.display import display, Markdown

user_request = """
I need to add a 'likes' count to the Chirp model.
It should be an integer and default to 0.

Then, update the 'HandlerChirpsCreate' function. After creating a chirp,
the response should include this new 'likes' field.
"""

if 'collection' in locals() or 'collection' in globals():
    # Get the suggested code change
    suggested_change = query_rag(user_request)

    # Create the full markdown string and display it
    markdown_output = f"""
---
### SUGGESTED CODE CHANGE
---
{suggested_change}
"""

    display(Markdown(markdown_output))
else:
    print("Cannot run query as the ChromaDB collection was not created.")
