import json
import re

# Policy documents → sentence-wise chunks


with open("part3/policies.json", "r", encoding="utf-8") as file:
    policies = json.load(file)


chunks = []

for policy in policies:
    sentences = re.split(r"(?<=[.!?])\s+", policy["text"].strip())

    for sentence_number, sentence in enumerate(sentences, start=1):
        sentence = sentence.strip()

        if not sentence:
            continue

        chunks.append({
            "chunk_id": f"{policy['document_id']}_CH{sentence_number}",
            "document_id": policy["document_id"],
            "title": policy["title"],
            "text": sentence
        })


with open("part3/chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=2, ensure_ascii=False)


print("Policy documents:", len(policies))
print("Sentence chunks:", len(chunks))