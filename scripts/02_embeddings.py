"""Day 3: local embeddings + brute-force cosine search (no API key needed).

This is the baseline the IRVS project will later replace with Qdrant + BM25 hybrid search.
"""
import time

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"  # swap to bge-large-en-v1.5 for Project #2
# bge models expect this prefix on queries (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DOCS = [
    "Invoice INV-2041 from Acme Corp, total due $12,400, payment terms net 30.",
    "Purchase order PO-7781 for 40 laptops approved by procurement.",
    "AWR report shows high 'db file sequential read' waits on the ORDERS index.",
    "ASH sampling indicates row lock contention (enq: TX) during the batch window.",
    "Vendor bank details changed; flag for manual verification before payment.",
    "Tablespace USERS is 94% full; autoextend is disabled.",
    "Goods receipt note GRN-332 confirms delivery of 38 of 40 laptops.",
    "Top SQL by elapsed time is a full table scan on TRANSACTIONS.",
]

QUERIES = [
    "which invoice has a payment amount owed?",
    "why is the database slow on index reads?",
    "possible fraud risk in supplier payment",
    "storage running out of space",
    "quantity mismatch between order and delivery",
]


def main() -> None:
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL)
    print(f"loaded {MODEL} in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    doc_vecs = model.encode(DOCS, normalize_embeddings=True)  # (n_docs, dim)
    print(f"encoded {len(DOCS)} docs -> {doc_vecs.shape} in {time.perf_counter() - t0:.2f}s\n")

    q_vecs = model.encode([QUERY_PREFIX + q for q in QUERIES], normalize_embeddings=True)
    scores = q_vecs @ doc_vecs.T  # normalized vectors -> dot product == cosine similarity

    for q, row in zip(QUERIES, scores):
        print(f"Q: {q}")
        for i in np.argsort(-row)[:2]:
            print(f"   {row[i]:.3f}  {DOCS[i]}")
        print()


if __name__ == "__main__":
    main()
