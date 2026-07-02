from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np


class SparseRetriever:

    def __init__(self):
        self.bm25 = None
        self.documents = None

    def build_index(self, documents: List[Dict]):
        tokenized_docs = [doc["text"].split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents


    def search(self,query:str, top_k:int):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        indices = np.argsort(scores)[::-1][:top_k]

        return {"scores":scores[indices].tolist(), "indices": indices.tolist()}