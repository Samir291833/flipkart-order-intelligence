import json

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_FILE = "indexes/policy_index.faiss"
METADATA_FILE = "indexes/policy_chunks.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# LOAD RETRIEVAL COMPONENTS
# ============================================================

model = SentenceTransformer(EMBEDDING_MODEL)

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# ============================================================
# POLICY RETRIEVAL
# ============================================================

def retrieve_policy(query, top_k=TOP_K):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position < 0:
            continue

        chunk = chunks[index_position].copy()

        chunk["similarity_score"] = float(score)

        results.append(chunk)

    return results


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    query = input("Enter your query: ")

    results = retrieve_policy(query)

    print("\nTop Retrieved Policy Chunks:")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(
            f"Similarity Score: "
            f"{result['similarity_score']:.4f}"
        )

        print(
            f"Document ID: "
            f"{result['document_id']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )

        print("-" * 70)