from src.agent.state import AgentState, Reflection
from langchain_openai import ChatOpenAI
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
import time
from src.utils.monitoring import log_to_mlflow
import mlflow
from src.utils.prompt_manager import PromptManager
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from custom_logging import logger


from src.utils import monitoring


promptloader= PromptManager()

perf_cb=PerformanceCallback()


llm_pro = ChatOpenAI(model="gpt-4o-mini",streaming=True,
    temperature=0, max_retries=3)

# Snapshot Version (Secondary Backup)
llm_pro_snap = ChatOpenAI(model="gpt-4o",streaming=True, 
   temperature=0)

# Cheap & Fast (Final Safety Net)
llm_mini = ChatOpenAI(model="gpt-4.1-mini",streaming=True, 
    temperature=0, max_retries=5)

# 2. THE CHAIN OF COMMAND
# For heavy tasks (Generator/Auditor): 
# Try GPT-4o -> Try Snapshot -> Try Mini (Fail-safe)
resilient_pro = llm_pro.with_fallbacks([llm_pro_snap, llm_mini])
# For light tasks (Security/Planner):
# Try Mini -> Try Pro
resilient_mini = llm_mini.with_fallbacks([llm_pro])

def audit_engine(state: AgentState):
    start_ts=time.time()
    query_type =state.get("type")
    current_turn = state.get("turn_count", 0) + 1
    current_attempts = state.get("audit_attempts", 0)
    current_query = state.get("query")
    planner_tasks = state.get("plan")
    generated_report= state.get("generation")
    wiki_archive = state.get("wiki_archive", "")
    audit_wiki= state.get("audit_wiki","")
    is_follow_up = state.get("is_follow_up", False)
    query_history= state.get("query_history")

    try:
        node_config = promptloader.prompts.get('audit_engine', {})
        raw_template = node_config.get('audit_engine_prompt')
        prompt_version = node_config.get('version', '1.0.0')
    except Exception as e:
        logger.exception(e)
        return {
             "audit_engine_failed": True
       
        }

    context_str = ""
    wiki_str = "\n".join([f"---[COORD:  Company: {item.company} | Year: {item.year} | Source: {item.source} | PAGE: {item.page}] ---\nVERIFIED METRIC: {item.evidence})" for item in audit_wiki])
    context_str = f""" ### Evidences so far (ARCHIVE)
        {wiki_archive if wiki_archive else "No historical archive yet."}
        ### ACTIVE INVESTIGATION (WIKI)
        {wiki_str if wiki_str else "No new findings this turn."}
        """

    
    math_val = state.get('calculation_result')
    math_info = f"CALCULATED_MATH: {math_val}" if math_val is not None else "CALCULATED_MATH: N/A"
    
  
    mode_instruction = f"""
    ### MODE:
    - Status: {"Follow-up turn" if is_follow_up else "Fresh audit"}
    - Current Query: {current_query}
    - Query History: {query_history if is_follow_up else "N/A"}
    - Evidence Rules: Every fact in the report MUST be 
    supported by either ARCHIVE or WIKI.
    - Hallucination Policy: Any claim not found in 
    ARCHIVE or WIKI is a hallucination.
    - The report must materially answer: {current_query}
    """
   
    

    # ADVANCED PROMPT: Separates Data Validation from Writing Validation
    audit_prompt = raw_template.format(
        mode_instruction=mode_instruction,
        math_val=math_val,
        math_info=math_info,
        planner_tasks=planner_tasks,
        context_str=context_str,
        generated_report=generated_report,
        current_query=current_query,
        query_history= query_history[-1] if query_history else "No prior context"
   
    )

    user_content = f"Please audit the following report against the provided context for query: {current_query}"
    
    structured_llm = resilient_pro.with_structured_output(Reflection, include_raw=True) # Replace with your Pydantic class
    
    try:
        response = structured_llm.invoke(
        [
            SystemMessage(content=audit_prompt),
            HumanMessage(content=user_content)
        ],
        config={"callbacks": [perf_cb]}
    )

    except Exception:
        logger.exception(
            f"Audit Engine LLM invocation failed | prompt_version={prompt_version}"
        )
        return {
            "audit_engine_failed": True
        }

    output = response["parsed"]

    scores={
    "math_score":output.math_score,
    "traceability_score":output.traceability_score,
    "hallucination_score":output.hallucination_score
    }

    current_run_id = monitoring.ACTIVE_AUDIT_RUN_ID
    
    if current_run_id is None:
        print("Warning: Guardrail sees no Active Run ID")
    else:
        for key,value in scores.items():
            monitoring.client.log_metric(
                run_id=current_run_id, 
                key=key, 
                value=float(value if value is not None else 0),
                step=current_turn
            )
    
   
    metrics_getter = get_node_metrics(
        "audit_engine",
        response,
        perf_cb,
        start_ts
        
    )
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version

    try:
        log_to_mlflow("audit_engine", node_results, step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")
        
    needs_revision = output.needs_revision
    
    
    print(output.math_score,output.traceability_score,output.hallucination_score,output.divergence_score)
    
    is_factually_safe:bool= output.hallucination_score>=4
    is_math_accurate:bool = output.math_score>=4
    is_traceable_enough:bool = output.traceability_score>=3


    if all([is_factually_safe, is_math_accurate, is_traceable_enough]):
        needs_revision = False
        target_node = "human_review"
        audit_status = "PASSED"
    else:
        needs_revision=True
        target_node="generator"

    
    if needs_revision:
        current_attempts += 1
        if current_attempts >= 3:
            target_node = "human_review"
            needs_revision = False
            audit_status = "MAX_AUDIT_ATTEMPTS_REACHED_ESCALATED_TO_HUMAN_REVIEW"
    

    
    if target_node == "human_review":
        audit_status = "VERIFIED_BY_AUDITOR"
    else:
        audit_status = "NEEDS_CORRECTION_ROUTING_TO_GENERATOR"

    return {
        "turn_count": current_turn,
        **node_results,
        "reflection_feedback": output,
        "audit_status": audit_status,
        "target_node": target_node,
        "audit_attempts": current_attempts,
        "critique": output.critique,
        "steps": [
            f"decision:{output.decision}"
            f"Audit attempt: {current_attempts}",
            f"Scores: H:{output.hallucination_score} M:{output.math_score} T:{output.traceability_score} D:{output.divergence_score}",
            f"Target Node: {target_node}",
            f"Feedback: {output.critique[:100]}..."
        ]
    }