import json

from retrieve import retrieve_policy


# ============================================================
# CONFIGURATION
# ============================================================

SIMILARITY_THRESHOLD = 0.40

TOP_K = 5


# ============================================================
# MOCK LLM RESPONSE GENERATOR
# ============================================================
# This deterministic function uses only retrieved evidence.
# No API call or network connection is required.

def mock_llm_policy_answer(results):

    if not results:
        return {
            "answer": (
                "I could not find sufficiently relevant information "
                "in the policy knowledge base to answer this question."
            ),
            "source": "policy_kb",
            "confidence": 0.0
        }

    best_score = results[0]["similarity_score"]

    if best_score < SIMILARITY_THRESHOLD:

        return {
            "answer": (
                "I could not find sufficiently relevant information "
                "in the policy knowledge base to answer this question."
            ),
            "source": "policy_kb",
            "confidence": round(best_score, 4)
        }

    # Keep only sufficiently relevant evidence.
    grounded_results = [
        result
        for result in results
        if result["similarity_score"] >= SIMILARITY_THRESHOLD
    ]

    # Deduplicate identical parent documents.
    seen_documents = set()
    evidence = []

    for result in grounded_results:

        document_id = result["document_id"]

        if document_id not in seen_documents:

            evidence.append(result["text"])
            seen_documents.add(document_id)

    answer = " ".join(evidence)

    return {
        "answer": answer,
        "source": "policy_kb",
        "confidence": round(best_score, 4)
    }


# ============================================================
# POLICY QUESTION PROCESSING
# ============================================================

def answer_policy_question(query):

    results = retrieve_policy(
        query,
        top_k=TOP_K
    )

    print("\nRetrieved Policy Evidence:")
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

    best_score = (
        results[0]["similarity_score"]
        if results
        else 0.0
    )

    print("\nGroundedness Check:")
    print("=" * 70)

    print(
        f"Best similarity score: "
        f"{best_score:.4f}"
    )

    print(
        f"Similarity threshold: "
        f"{SIMILARITY_THRESHOLD:.4f}"
    )

    if best_score >= SIMILARITY_THRESHOLD:

        print("Groundedness result: PASSED")

    else:

        print("Groundedness result: FAILED")

        print(
            "Policy answer refused because the "
            "retrieved evidence is insufficient."
        )

    response = mock_llm_policy_answer(results)

    print("\nGenerated Answer:")
    print("=" * 70)

    print(
        json.dumps(
            response,
            indent=4
        )
    )

    return response


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    query = input("Enter your query: ")

    answer_policy_question(query)