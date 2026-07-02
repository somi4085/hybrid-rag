import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    EMBEDDING_MODEL="all-MiniLM-L6-v2"
    TOP_K = 10
    