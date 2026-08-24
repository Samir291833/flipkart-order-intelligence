import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = "part3/chunks.json"
INDEX_DIR = "indexes"
INDEX_FILE = "indexes/policy_index.faiss"
METADATA_FILE = "indexes/policy_chunks.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKED POLICY DATA
# ============================================================

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]

print("Policy chunks:", len(chunks))


# ============================================================
# LOAD LOCAL SENTENCE TRANSFORMER
# ============================================================

model = SentenceTransformer(EMBEDDING_MODEL)

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = embeddings.astype("float32")

print("Embedding dimension:", embeddings.shape[1])


# ============================================================
# BUILD FAISS COSINE-SIMILARITY INDEX
# ============================================================
# With normalized vectors, inner product is equivalent to
# cosine similarity.

index = faiss.IndexFlatIP(embeddings.shape[1])

index.add(embeddings)

print("FAISS vectors:", index.ntotal)


# ============================================================
# SAVE INDEX
# ============================================================

os.makedirs(INDEX_DIR, exist_ok=True)

faiss.write_index(index, INDEX_FILE)

with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=2, ensure_ascii=False)

print("Index saved to:", INDEX_FILE)
print("Metadata saved to:", METADATA_FILE)