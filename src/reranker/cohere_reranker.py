import cohere
from typing import List, Dict


class CohereReranker:
    def __init__(self, api_key:str):
        self.client = cohere.Client(api_key)

    def rerank(self,query:str, documents:List[str], top_n:int):
        results = self.client.rerank(model = "rerank-english-v2.0", query=query, documents=[doc["text"] for doc in documents], top_n=top_n)

        return [{"text":documents[r.index]["text"], "relevance_score":r.relevance_score} for r in results.results]
        