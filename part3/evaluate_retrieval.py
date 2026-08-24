import sys
import os

# Allow imports from the part3 directory when executed
# from the project root.
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from retrieve import retrieve_policy


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================
# Evaluation is performed at DOCUMENT level.
# Multiple retrieved chunks belonging to the same document
# are deduplicated before Precision@3 and Recall@3 are
# calculated.


EVALUATION_QUERIES = [
    {
        "query": "How many days can I return apparel?",
        "relevant": ["POL001"]
    },
    {
        "query": "How long do I have to return electronics?",
        "relevant": ["POL003"]
    },
    {
        "query": "How many days do I have to return a home product?",
        "relevant": ["POL004"]
    },
    {
        "query": "What happens to a cash-on-delivery refund after a return?",
        "relevant": ["POL005"]
    },
    {
        "query": "Can an eligible return be collected from my address?",
        "relevant": ["POL009"]
    },
    {
        "query": "What should I do if my product arrives damaged?",
        "relevant": ["POL013"]
    }
]


def evaluate_query(item):

    query = item["query"]
    relevant = set(item["relevant"])

    results = retrieve_policy(
        query,
        top_k=3
    )

    retrieved_documents = []

    for result in results:

        document_id = result["document_id"]

        if document_id not in retrieved_documents:
            retrieved_documents.append(
                document_id
            )

    retrieved_set = set(
        retrieved_documents
    )

    true_positives = len(
        retrieved_set.intersection(
            relevant
        )
    )

    precision = (
        true_positives /
        len(retrieved_documents)
        if retrieved_documents
        else 0.0
    )

    recall = (
        true_positives /
        len(relevant)
        if relevant
        else 0.0
    )

    print("\n" + "-" * 70)
    print("Query:", query)
    print("Relevant documents:", sorted(relevant))
    print(
        "Top-3 retrieved documents:",
        retrieved_documents
    )

    print(
        f"Precision@3 = "
        f"{true_positives}/"
        f"{len(retrieved_documents)} "
        f"= {precision:.4f}"
    )

    print(
        f"Recall@3 = "
        f"{true_positives}/"
        f"{len(relevant)} "
        f"= {recall:.4f}"
    )

    return precision, recall


def main():

    print("=" * 70)
    print("PART 3 - DOCUMENT-LEVEL RETRIEVAL EVALUATION")
    print("=" * 70)

    precisions = []
    recalls = []

    for item in EVALUATION_QUERIES:

        precision, recall = evaluate_query(
            item
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

    average_precision = (
        sum(precisions) /
        len(precisions)
    )

    average_recall = (
        sum(recalls) /
        len(recalls)
    )

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Average Precision@3 = "
        f"{average_precision:.4f}"
    )

    print(
        f"Average Recall@3 = "
        f"{average_recall:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()