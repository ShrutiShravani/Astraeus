import chromadb
from langchain_openai import OpenAIEmbeddings
import os
import time
import chromadb
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

# Global placeholders
_collection = None
_embeddings = None


@retry(stop=stop_after_attempt(5),
wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: print(f"Retrying ChromaDB connection... Attempt {retry_state.attempt_number}")
)


def get_cache_collection():
    global _collection 
    if _collection is None:
        # Initialize Chroma once
        chroma_client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST"), 
            port=os.getenv("CHROMA_PORT")
        )
        
        chroma_client.heartbeat()
        _collection = chroma_client.get_or_create_collection(name="new_audits_v3")

    return _collection

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=key)
    return _embeddings