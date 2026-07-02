from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np



class HybridRetriever:

    def __init__(self, model_name:str):
        self.dense = DenseRetriever(model_name)
        self.sparse = SparseRetriever()

    def build_index(self,documents:List[Dict]):
        self.dense.build_index(documents)
        self.sparse.build_index(documents)


    def search(self,query:str, top_k:int, alpha=0.5):
        dense_result = self.dense.search(query, top_k)
        sparse_result = self.sparse.search(query, top_k)
        combined_score = alpha * np.array(dense_result["distances"][0]) + (1-alpha) * np.array(sparse_result["scores"])
        top_k_indices = np.argsort(combined_score)[::-1][:top_k]
        return [{"index":int(i),"score": float(combined_score[i]) } for i in top_k_indices]

        