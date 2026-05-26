from src.agent.state import AgentState,Retriever_feedback
import time
from src.utils.monitoring import log_to_mlflow
from langchain_openai import ChatOpenAI
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
from src.utils.monitoring import log_to_mlflow
import mlflow
from src.utils.prompt_manager import PromptManager
from langchain_core.prompts import ChatPromptTemplate
import asyncio
from langchain_core.runnables import RunnableConfig

promptloader= PromptManager()
perf_cb=PerformanceCallback()
llm_pro = ChatOpenAI(model="gpt-4o-mini", 
    temperature=0,streaming=True, max_retries=3)

# Snapshot Version (Secondary Backup)
llm_pro_snap = ChatOpenAI(model="gpt-4o", streaming=True,
    temperature=0)

# Cheap & Fast (Final Safety Net)
llm_mini = ChatOpenAI(model="gpt-4.1-mini",streaming=True,
    temperature=0, max_retries=5)

# 2. THE CHAIN OF COMMAND
# For heavy tasks (Generator/Auditor): 
# Try GPT-4o -> Try Snapshot -> Try Mini (Fail-safe)
resilient_pro = llm_pro.with_fallbacks([llm_pro_snap, llm_mini])

def query_aware_compression(context,query,planner_tasks):
    """
    Compresses context to only the parts relevant to the query and tasks.
    """

    compression_prompt = f"""
   ### MISSION
    You are a High-Density Forensic Data Compressor. Your goal is to strip away prose and fluff while preserving 100% of the factual, numerical, and evidentiary value of the provided context.

    ### RULES
    1. **NO FILTERING:** Do not decide what is relevant to the query. Keep ALL raw data, metrics, years, and financial figures found in the context.
    2. **FORMATTING:** Use bullet points for high-density facts. 
    3. **LEGAL/AUDIT INTEGRITY:** Maintain all Source and Page references for every fact.
    4. **DELETE:** Remove introductory sentences ("The document states..."), transition phrases, and marketing adjectives.
    5. **PRESERVE:** Keep all explanations of accounting methods or 'notes to financial statements'.

    [QUERY]: {query}
    [CONTEXT]: {context}

    Output the compressed, high-density evidence blocks below:
    """

    response =  llm_pro.ainvoke(compression_prompt)
    return response.content

def retrieval_auditor_node(state:AgentState):
    #remove config: RunnableConfig from args above
    start_ts=time.time()
    current_query = state["query"]
    current_turn = state.get("turn_count", 0) + 1
    structured_llm =  resilient_pro.with_structured_output(Retriever_feedback,include_raw=True)
    existing_audit_wiki=state.get("audit_wiki",[])
    context= state.get("context")
    evidence_found=[]
    combined_critique=[]
    existing_final_context = state.get("final_context", [])
    

    plan = state.get("plan", [])
    
    planner_tasks= "\n".join([f"Task{i+1} {t.title} | Source: {t.doc_source}" for i, t in enumerate(plan)]) 
    
    print(f"Debug-----Stratign compresstion {time.time()}")
    compressed_evidence= query_aware_compression(context,current_query,planner_tasks)
    print(f"compressed_evidence_summary:{compressed_evidence}")

    node_config = promptloader.prompts.get('retriever_auditor', {})
    raw_template = node_config.get('retriever_auditor_prompt')
    prompt_version = node_config.get('version', '1.0.0')
    current_attempts = state.get("retriever_audit_attempts", 0)

    prompt = raw_template.format(
        planner_tasks=planner_tasks, 
        context=compressed_evidence,
    )
    print(f"!!! DISPATCHING: {planner_tasks.title}")
  
    # .ainvoke is the ASYNC version of .invoke
    response = structured_llm.ainvoke(
        prompt, 
        config={"callbacks": [perf_cb]}
    )
    print(f"DEBUG: Task {planner_tasks.title} finished at {time.time()}")
 

    # 3. MERGE THE RESULTS
    evidence_found = [] 
    output = response["parsed"]

    found_evidence_list = output.found_evidence

    for item in found_evidence_list:
        item.status=item.status.lower()
        if item.status!="missing":
            print(f"Verified:{item.task_name} | {item.company} | {item.year} | {item.evidence} in {item.source} p.{item.page} {item.quote}")
            evidence_found.append(item)
          
    all_evidence = existing_audit_wiki + evidence_found
  
    verified_pages={str(item.page) for item in evidence_found}

    #old_verified_pages= {str(item.page) for item in audit_wiki}
    new_pruned = [chunk for chunk in context if str(chunk.get("page")) in verified_pages]
    
    all_contexts = existing_final_context + new_pruned
    
     
    # 4. CAPTURE METRICS (The Senior-Level Step)
    metrics_getter = get_node_metrics(
        "retrieval_auditor",
        response, 
        perf_cb, 
        start_ts,
    )

    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version
    log_to_mlflow("retrieval_auditor",node_results,step=current_turn)

    needs_revision=output.needs_revision
    print(f"need_revision{needs_revision}")
    print(output.retriever_critique)

    all_evidence = existing_audit_wiki + evidence_found


    if needs_revision is True:
        print("needs_revision")
        current_attempts += 1
        
    else:
        print("Audit complete. Passing full accumulated context to Generator.")
  
    return{
        "turn_count": current_turn,
        "audit_wiki":all_evidence,
        **node_results,
        "context": context,
        "final_context": all_contexts,
        "retriever_feedback": output,
        "retriever_audit_attempts": current_attempts,
        "critique":   combined_critique,
        "steps": [
            f"Audit attempt: {current_attempts}",
            f"Feedback: {combined_critique[:100]}...",
            f"Retrieved_Evidence:{found_evidence_list}"
        ]
    }

