from src.agent.state import AgentState
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from src.utils.db import get_cache_collection, get_embeddings
import uuid
import time
from src.utils.monitoring import log_to_mlflow

collection = get_cache_collection()
embeddings = get_embeddings()

def semantic_cache_check_node(state:AgentState):
    start_ts= time.time()
    years = state.get("target_year") 
    if isinstance(years, list):
        year_to_use = sorted([int(year) for year in years])
    else:
        year_to_use = [(int(years))]
   
       
    company= state.get("target_company")
    print("checking for cache")
    current_turn = state.get("turn_count", 0) + 1
    query = state.get("query", [])
    
    #search_string = " | ".join(history) 
    if not query:
        print("is_cached:False")
        return {"is_cached": False}
    
    
    query_embedding = embeddings.embed_query(query)
    
    latency= round(time.time()-start_ts,3)
    
    where_conditions = {}
    if company and years:
        where_conditions = {
            "$and": [
                {"company": {"$eq": str(company).upper()}},
                {"year": {"$in": year_to_use}}
            ]
        }
     
    print(f"year:{year_to_use}")
    print(f"company:{company}")
 # Search for similar queries with a distance threshold (1 - distance = similarity)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        where=where_conditions,
        include=["metadatas", "documents", "distances"]
    )

    found_years=[]

    if results and results["metadatas"][0]:
        found_years=[m.get("year") for m in results["metadatas"][0]]

        
    if results["distances"][0] and results["distances"][0][0] < 0.10 and set(found_years)==set(year_to_use):  # 95% similarity
        cached_report = results["documents"][0][0]
        print("--- CACHE HIT: Bypassing LLM Nodes ---")
      
        node_metrics = {
        "node_benchmarks": {
            "python_repl": {
                "ttft": 0,  # No tokens in Python
                "latency": round(latency, 3),
                "input_tokens":0,
                "output_tokens":0,
                "tokens": 0,
                "cost": 0.0,
                "model": "None", 
                "tps": 0
            }
        }
    }
        log_to_mlflow("semantic_Cache", node_metrics, step=current_turn)
        return {
            "generation": cached_report,
            "turn_count": current_turn,
            "audit_status": "VERIFIED (FROM CACHE)",
            "is_cached": True,
            **node_metrics
        }
    
    return {"is_cached": False}

def finalize_audit(state: dict):
    """The Final Node: Runs after Human Review is complete"""
    decision = state.get("human_decision")
    raw_year = state.get("target_year") 
    if isinstance(raw_year, list):
        clean_year = int(raw_year[0])
    elif isinstance(raw_year, str):
        clean_year = int(raw_year)
    else:
        clean_year = int(raw_year)
    company= state.get("target_company")
    if state.get("is_cached") is True:
        print("--- SKIP CACHE SAVE: Data already exists in ChromaDB ---")
        return state
        
    if decision == "pass":
        # 1. Get the cache collection
        cache = get_cache_collection()
        
        # 2. Prepare the key (Query History)
        raw_history = state.get("query_history", [])
        semantic_key = " | ".join(raw_history)
        vector = embeddings.embed_query(semantic_key)
        
        # 3. SAVE TO CHROMADB
        cache.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            documents=[state.get("generation")],
            metadatas=[{
                "query_fingerprint": semantic_key,
                "timestamp": "2026-03-20",
                "status": "human_verified",
                "company": str(company).upper(), 
                "year": int(clean_year) if clean_year else 0

            }]
        )
        print("--- GRAPH COMPLETED: Report saved to Semantic Cache ---")
    
    return state

def check_chroma_health():
    try:
        # Get the count of all cached audits
        count = collection.count()
        
        # Check the size of the storage folder (if local)
        # Or check Docker stats if running in a container
        print(f"--- CHROMADB STATUS ---")
        print(f"Total Cached Audits: {count}")
        
        # If count > 10,000, your search latency might increase
        if count > 10000:
            print("WARNING: Cache size is large. Consider index optimization.")
        return True
    except Exception as e:
        print(f"CHROMA ERROR: {e}")
        return False