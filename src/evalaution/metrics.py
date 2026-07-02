import numpy as np
from typing import List, Dict


def mean_reciprocal_rank(relevant_docs: List[str], retrieved_docs: List[str]):
    for index, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1/(index+1)
    return 0.0
    
def ndcg_score(relevant_docs, retrieved_docs, k:int):
    dcg = 0.0
    idcg = 0.0
    for index, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            dcg += 1/np.log2(index+2)

    for index, doc in enumerate(retrieved_docs[:k]):
            idcg += 1/np.log2(index+2)
    
    return dcg/idcg



def faithfulness_score(answer:str, context:List[str]):
    match = 0
    sentences = answer.split(".")
    for sentence in sentences:
        if any(sentence in ctx for ctx in context):
            match += 1
    return match/len(sentences)

