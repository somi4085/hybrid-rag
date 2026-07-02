import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, model_name:str):
        self.index = None
        self.model = SentenceTransformer(model_name)

    
    def build_index(self,documents: List[Dict]):
        embeddings = self.model.encode([doc["text"] for doc in documents]).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    
    def search(self,query:str, top_k:int):
        query_embeddings = self.model.encode(query).astype(np.float32).reshape(1,-1)
        distances, indices =  self.index.search(query_embeddings, top_k)

        return {"distances": distances, "indices": indices}
        




        