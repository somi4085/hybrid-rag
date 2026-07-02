import sys
sys.path.append("src")
from evalaution.metrics import mean_reciprocal_rank, ndcg_score, faithfulness_score


def test_metrics():
    relevant_docs = ["Machine learning is a subset of artificial intelligence", "FAISS is a library for efficient similarity search"]
    retrieved_docs = ["FAISS is a library for efficient similarity search", "Machine learning is a subset of artificial intelligence"]

    answer = "Machine learning is a subset of artificial intelligence"
    context = relevant_docs

    print("MRR:", mean_reciprocal_rank(relevant_docs, retrieved_docs))
    print("ndcg_score:", ndcg_score(relevant_docs, retrieved_docs, k=5))
    print("faithful_score", faithfulness_score(answer, context))

test_metrics()
                     