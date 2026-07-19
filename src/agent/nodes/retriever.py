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
from custom_logging import logger
import mlflow

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


def pre_filtering(initial_chunks, planner_tasks):
    scored_chunks = []
    
    for chunk in initial_chunks:
        # 1. Prepare chunk text once for efficiency
        text = chunk["evidence"].lower()
        max_overall_score = 0
        
        for task in planner_tasks:
            title = task.title.lower()
            
            # 2. Extract Year and Significant Words (Magnets)
            years_in_task = set(re.findall(r'\b(20\d{2})\b', title))
            # Get all words 4+ chars, excluding the year itself
            task_words = set(re.findall(r'\b[a-z]{4,}\b', title)) - years_in_task
            
            # 3. Calculate the Cluster Match (Intersection)
            matched_words = [w for w in task_words if w in text]
            match_count = len(matched_words)
            
            current_task_score = 0
            
            # 4. SCORING LOGIC (The "Senior" Threshold)
            # We only give a major boost if a Year matches AND we have a cluster of keywords
            year_match = any(y in text for y in years_in_task) if years_in_task else True
            
            if year_match:
                if match_count >= 3:
                    # CLUSTER HIT: This is likely the exact paragraph/table row
                    current_task_score = 100 + match_count
                elif match_count >= 1:
                    # PARTIAL HIT: Right year, but maybe not the exact metric
                    current_task_score = 50 + match_count
                else:
                    # YEAR ONLY: Only the year matched
                    current_task_score = 10
            else:
                # NO YEAR MATCH: Worth very little in a forensic query
                current_task_score = match_count

            # Keep the highest score this chunk achieved against any task
            if current_task_score > max_overall_score:
                max_overall_score = current_task_score
        
        scored_chunks.append((max_overall_score, chunk))

    # 5. SORT AND PICK THE TOP 4
    # Sort by score descending (highest first)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Logging for your debugging
    top_scores = [s[0] for s in scored_chunks[:4]]
    print(f"DEBUG: Top 4 Scores: {top_scores}")
    
    # Return the best 4. Even if none hit the '100' threshold, 
    # it still returns the 4 'best available' to prevent zero-chunk errors.
    return [item[1] for item in scored_chunks[:4]]
"""
def hybrid_retriever_node(state: AgentState):
    start_ts = time.time()
    query = state["query"]
    failed_tasks = state.get("failed_tasks", [])
    print("rotuihn  retrierv")
    plan = failed_tasks if failed_tasks else state.get("plan", [])
    current_turn = state.get("turn_count", 0) + 1
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

    
    # --- 2. DYNAMIC YEAR FILTERING (UNIVERSAL) ---
    filter_text = (" ".join(t.title for t in plan) if plan else query) 
    found_years = [int(y) for y in re.findall(r'\b(20\d{2})\b', filter_text)]
     

    for task_obj in plan:
        search_string = task_obj.title
        task_company = task_obj.extracted_company
        print(task_company)
        task_year = task_obj.extracted_year
        print(task_year)
    
    
        filter_conditions = []
        
        if task_company:
            filter_conditions.append(
                models.FieldCondition(key="metadata.company", match=models.MatchValue(value=task_company))
            )
       

        if task_year:
            filter_conditions.append(
                models.FieldCondition(key="metadata.year", match=models.MatchValue(value=int(task_year)))
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
                models.Prefetch(query=dense_vector, using="dense", limit=150),
                models.Prefetch(query=sparse_vector, using="sparse", limit=150)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(must=filter_conditions)
        )
        
        try:
            for point in search_result.points: 
                if point.id not in seen_ids:
                    payload =point.payload
                    if "metadata" in payload:
                        print(f"metadata keys{payload['metadata'].keys()}")
                    text_content= payload.get("page_content","")
                    p_company = payload.get("company") or payload.get("metadata", {}).get("company") or "N/A"  # Not "metadata.company", just "company"
                    p_year = payload.get("year") or  payload.get("metadata", {}).get("year") or "N/A"
                    p_source = payload.get("source") or payload.get("metadata", {}).get("source") or "N/A"    # This is your filename (e.g., NIKE_10K_2020.pdf)
                    p_doc_type = payload.get("doc_type") or payload.get("metadata", {}).get("doc_type") or "N/A"
                    p_page = payload.get("page") or payload.get("metadata", {}).get("page") or "N/A"
                    
                    #we inject the metadata into the text for the llm to see it later

                    enriched_text =(
                    f"--- [COORD START] ---\n"
                    f"COMPANY_NAME: {p_company}\n"
                    f"YEAR: {p_year}\n"
                    f"SOURCE_ID: {p_source}\n" # This is the full filename
                    f"DOC_TYPE: {p_doc_type}\n"
                    f"PAGE: {p_page}\n"
                    f"CONTENT: {text_content}\n"
                    f"--- [COORD END] ---\n"
                )
                        
                    
                    all_contexts.append({
                        "id": point.id,
                        "source": p_source,
                        "company": p_company, # Explicitly pass these for pre_filtering
                        "year": p_year,#
                        "doc_type":p_doc_type,
                        "page":p_page,
                        "text": enriched_text,
        
                    })
                
                    seen_ids.add(point.id)
        except Exception as e:
            logger.exception(e)
            return {
            "retriever_failed": True
        }
    
    # --- 4. RERANKING ---
    rerank_query = query

    # --- 4. RERANKING ---
    if not all_contexts:
        print("WARNING: No contexts found in Qdrant. Skipping rerank.")
        top_chunks = []
        
    try:
        rerank_request = RerankRequest(query=rerank_query, passages=all_contexts)
        # Return top 8 to ensure we catch enough rows from large tables
        #limit=12 if len(search_tasks)>1 else 8
        top_chunks = ranker.rerank(rerank_request)[:10]
    except Exception as e:
            logger.exception(e)
            return {
            "retriever_failed": True
        }
    


    initial_contexts = []
    final_contexts=[]
    entry={}
    for chunk in top_chunks:
        entry={
        "company": chunk["company"], # Use the one tagged during retrieval
        "year": chunk["year"],
        "source":chunk["source"],
        "page": chunk["page"],
        "evidence":chunk["text"]
    }  
        final_contexts.append(entry)
        print(f"length of initial_contexts:{len(final_contexts)}")
    
    #final_contexts = pre_filtering(initial_contexts, plan)
    print(f"[DEBUG] final_cotnexts_count:{len(final_contexts)}")
    print(f"[DEBUG] {[c['source']for c in final_contexts]}")
    print(f"[RETRIEVAL DEBUG] Sources: {[(c['source'], c['year'], c['page']) for c in final_contexts]}")


    for i, ctx in enumerate(final_contexts):
        print(f"\n[CHUNK {i}]")
        print(f"  Source: {ctx['source']} | Year: {ctx['year']} | Page: {ctx['page']}")
        # Show first 200 chars — does it contain the metric?
        print(f"  Content preview: {ctx['evidence'][200:400]}")
        print(f"{'='*50}\n")

 
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

    try:
        log_to_mlflow("retriever",node_metrics,step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")

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