from qdrant_client import models
from src.agent.state import AgentState
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
import os
import re
import time
from src.utils.monitoring import log_to_mlflow
from deepeval.metrics import ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.utils.get_metrics import log_heavy_metrics

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
ranker = Ranker(model_name="rank-T5-flan", cache_dir="~/.cache")

#intialize redis
#cache_client= redis.Redis(host='localhost', port=6379, db=0,decode_responses=True)


#intialzie Qdrant
client = QdrantClient(url="http://localhost:6333")
# Dense: For semantic meaning
dense_encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Sparse: For keyword precision (PP&E, 2018, etc.)
sparse_encoder = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
"""
def forensic_cache_key(state:AgentState):
    #combine key identifiers
    key_str = f"{state['target_company']}_{state['target_year']}_{state['query']}"
    return hashlib.md5(key_str.lower().strip().encode()).hexdigest()
"""

def hybrid_retriever_node(state: AgentState):
    start_ts = time.time()
    query = state["query"]
    plan = state.get("plan", [])
    current_turn = state.get("turn_count", 0) + 1
    reflection = state.get("reflection_feedback")
    target_company = str(state.get("target_company", "")).upper()
    target_year = state.get("target_year")
  

    #cache_key= forensic_cache_key(state)
    #cached_payload = cache_client.get(cache_key)
    """
    if cached_payload:
        print("[CACHE HIT]: Serving forensic evidence from Redis.")
        # We return the exact same structure as a real retrieval
        return {
            "context":json.loads(cached_payload),
            "steps": ["Retrieved forensic context from Semantic Cache."]

        }
    """

    all_contexts = []
    seen_ids = set()

    critique_text = f"{reflection.critique}" if (reflection and reflection.needs_revision and reflection.target_node=="retriever") else ""
    
    # --- 2. DYNAMIC YEAR FILTERING (UNIVERSAL) ---
    filter_text = (" ".join(t.title for t in plan) if plan else query) + critique_text
    found_years = [int(y) for y in re.findall(r'\b(20\d{2})\b', filter_text)]
    
    for task_obj in plan:
        search_string = task_obj.title
        if critique_text:
            search_string = f"{search_string} focus on {critique_text}"# Add critique as a standalone search
    
        filter_conditions = []
        
        if target_company:
            filter_conditions.append(
                models.FieldCondition(key="metadata.company", match=models.MatchValue(value=target_company))
            )

        if found_years:
            if reflection and reflection.needs_revision and "missing" in reflection.critique.lower():
                filter_conditions.append(
                    models.FieldCondition(key="metadata.year", match=models.MatchAny(any=[max(found_years)]))
                )
            else:
                max_y, min_y = max(found_years), min(found_years)
                if (max_y - min_y) <= 2:
                    # Standard 3-year SEC window: Target the latest document
                    filter_conditions.append(
                        models.FieldCondition(key="metadata.year", match=models.MatchValue(value=max_y))
                    )
                else:
                    # Multi-document range: Use step-down logic
                    needed = []
                    curr = max_y
                    while curr >= min_y:
                        needed.append(curr)
                        curr -= 3
                    filter_conditions.append(
                        models.FieldCondition(key="metadata.year", match=models.MatchAny(any=needed))
                    )
        elif target_year:
            # Fallback to benchmark state
            filter_conditions.append(
                models.FieldCondition(key="metadata.year", match=models.MatchValue(value=int(state["target_year"])))
            )
        
        #document type filtering
        if task_obj.doc_source in ["10K","Transcript"]:
            filter_conditions.append(
                models.FieldCondition(key="metadata.doc_type", match=models.MatchValue(value=task_obj.doc_source))
            )

       
        # Embeddings
        dense_vector = list(dense_encoder.embed(search_string))[0].tolist()
        sparse_result = list(sparse_encoder.embed([search_string]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices.tolist(), 
            values=sparse_result.values.tolist()
        )
        
        # Search with RRF Fusion
        search_result = client.query_points(
            collection_name="financial_reports",
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense", limit=50),
                models.Prefetch(query=sparse_vector, using="sparse", limit=50)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(must=filter_conditions)
        )

        for point in search_result.points: 
            if point.id not in seen_ids:
                payload =point.payload
                text_content= payload.get("page_content","")
                
                #we inject the metadata into the text for the llm to see it later

                enriched_text =(
                f"[SOURCE:{task_obj.doc_source} | ZONE:{task_obj.zone} | RATIONALE:{task_obj.rationale}]\n"
                f"{text_content}"
               )
                
                all_contexts.append({
                    "id": point.id,
                    "text": enriched_text,
                    "source": task_obj.doc_source,     # 10K or Transcript
                    "zone": task_obj.zone,
                    "page_num": payload.get("metadata", {}).get("page", "N/A"),
                })
               
                seen_ids.add(point.id)
    
    # --- 4. RERANKING ---
    rerank_query = query
    if reflection and reflection.needs_revision:
        rerank_query = f"{rerank_query} (Correction: {reflection.critique})"
        
    rerank_request = RerankRequest(query=rerank_query, passages=all_contexts)
    # Return top 8 to ensure we catch enough rows from large tables
    #limit=12 if len(search_tasks)>1 else 8
    top_chunks = ranker.rerank(rerank_request)[:10]
    print(f"total_chunks retrieved")

    

    final_contexts = []
    for chunk in top_chunks:
        entry={
        "source":chunk["source"],
        "zone": chunk["zone"],
        "page": chunk["page_num"],
        "evidence":chunk["text"],
        "company": target_company,
        "year": target_year
    }

    
        final_contexts.append(entry)
    
    """
    retrieval_scores= log_heavy_metrics(
        query=state['query'],
        context=[c['evidence'] for c in final_contexts],
        prompt_version="2.1.0",
        current_turn=current_turn
    )
    """
    
 

    node_latency = time.time() - start_ts
    node_metrics = {
        "node_benchmarks": {
            "retriever": {
                "ttft": 0,
                "latency": round(node_latency, 3),
                "input_tokens":0,
                "output_tokens":0,
                "tokens": 0,
                "cost": 0.0,
                "model": "hybrid-search-engine",
                "tps": 0,
            }
        }
    }

    node_metrics["prompt_version"] ="None"
    log_to_mlflow("retriever",node_metrics,step=current_turn)

    # 2. Save the results to Redis for next time
    #cache_client.setex(cache_key,86400,json.dumps(final_contexts))
    return {
        "context": final_contexts, 
        "turn_count": current_turn,
        **node_metrics,
        "steps": [
            f" RETRIEVAL PHASE COMPLETE:",
            f"- Tasks Executed: {len(plan)} (Hybrid Search: Dense + Sparse SPLADE)",
            f"- Filters Applied: Company={target_company}, Years={target_year}, Sources={[t.doc_source for t in plan]}",
            f"- Reranking: Processed {len(all_contexts)} candidates down to {len(top_chunks)} high-relevance chunks using FlashRank.",
            f"- Coverage: {len([c for c in final_contexts if c['source'] == '10K'])} snippets from 10-K, {len([c for c in final_contexts if c['source'] == 'Transcript'])} from Transcripts."
        ]
    }