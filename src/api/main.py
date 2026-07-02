import sys
sys.path.append("src")
from fastapi import FastAPI
from retrieval.hybrid_retriever import HybridRetriever 
from reranker.cohere_reranker import CohereReranker 
from config import Config


app = FastAPI()
retriever = HybridRetriever(Config.EMBEDDING_MODEL)      
reranker = CohereReranker(Config.COHERE_API_KEY)

sample_docs = [
    {"id": 1, "text": "Machine learning is a subset of artificial intelligence"},
    {"id": 2, "text": "Deep learning uses neural networks with many layers"},
    {"id": 3, "text": "Natural language processing helps computers understand text"},
    {"id": 4, "text": "FAISS is a library for efficient similarity search"},
    {"id": 5, "text": "BM25 is a ranking function used in information retrieval"}
]

retriever.build_index(sample_docs)

@app.post("/search")
def search(query: str, top_k: int = 10):
    try:
        result = retriever.search(query, top_k)
        indices = [r["index"] for r in result]
        matched_docs = [sample_docs[i] for i in indices]
        reranked = reranker.rerank(query, matched_docs, top_n=top_k)
        return reranked
    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}