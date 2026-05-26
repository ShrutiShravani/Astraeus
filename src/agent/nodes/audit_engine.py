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
    node_config = promptloader.prompts.get('audit_engine', {})
    raw_template = node_config.get('audit_engine_prompt')
    prompt_version = node_config.get('version', '1.0.0')
    

    context_str = ""
    context = state.get("context_history",[])
    for c in context:
        context_str += f"--- [COORD: {c['source']} | PAGE: {c['page']}] ---\n{c['evidence']}\n\n"
    
    math_val = state.get('calculation_result')
    math_info = f"CALCULATED_MATH: {math_val}" if math_val is not None else "CALCULATED_MATH: N/A"
    

    mode_instruction = f"""
    ### MODE: INITIAL AUDIT
    - This is the FIRST turn.
    - Audit the report ONLY against the CURRENT QUERY: "{current_query}".
    - Every material fact, figure, conclusion, and calculation in the report MUST be supported by the CURRENT EVIDENCE.
    - If the report contains unsupported claims not found in the CURRENT EVIDENCE, treat them as hallucinations.
    - The report must materially answer the CURRENT QUERY.
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
   
    )

    user_content = f"Please audit the following report against the provided context for query: {current_query}"
    
    structured_llm = resilient_pro.with_structured_output(Reflection, include_raw=True) # Replace with your Pydantic class

    response = structured_llm.invoke(
    [
        SystemMessage(content=audit_prompt),
        HumanMessage(content=user_content)
    ],
    config={"callbacks": [perf_cb]}
)

    output = response["parsed"]
    scores={
    "math_score":output.math_score,
    "traceability_score":output.traceability_score,
    "hallucination_score":output.hallucination_score,
    "divergence_score":output.divergence_score
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
    log_to_mlflow("audit_engine", node_results, step=current_turn)

    needs_revision = output.needs_revision
    
    
    print(output.math_score,output.traceability_score,output.hallucination_score,output.divergence_score)
    
    is_factually_safe:bool= output.hallucination_score>=4
    is_math_accurate:bool = output.math_score>=4
    is_traceable_enough:bool = output.traceability_score>=3
    is_divergence:bool =output.divergence_score>=4


    
    if query_type=="C":
        if all([is_factually_safe,is_math_accurate,is_traceable_enough,is_divergence]):
            needs_revision=False
            target_node = "human_review"
            audit_status = "PASSED"
        
        else:
            needs_revision=True
            target_node="generator"
    else:
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