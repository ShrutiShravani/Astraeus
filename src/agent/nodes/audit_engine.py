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
    current_turn = state.get("turn_count", 0) + 1
    current_attempts = state.get("audit_attempts", 0)
    current_query = state.get("query")
    planner_tasks = state.get("plan")
    query_history= state.get("query_history")
    past_queries = query_history[:-1] if len(query_history) > 1 else []
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
    
    if len(past_queries)==0:
        mode_instruction = f"""
        ### MODE: INITIAL AUDIT
        - This is the FIRST turn.
        - Audit the report ONLY against the CURRENT QUERY: "{current_query}".
        - Every material fact, figure, conclusion, and calculation in the report MUST be supported by the CURRENT EVIDENCE.
        - If the report contains unsupported claims not found in the CURRENT EVIDENCE, treat them as hallucinations.
        - The report must materially answer the CURRENT QUERY.
        """
    else:
        mode_instruction = f"""
            ### MODE: CUMULATIVE FOLLOW-UP AUDIT
            - This is a FOLLOW-UP turn.
            - The GENERATED_REPORT is a cumulative document that may include findings from:
            - PAST QUERIES: {past_queries}
            - CURRENT QUERY: {current_query}

            ### CUMULATIVE AUDIT RULE
            - The auditor must verify that the final report materially answers:
            1. the CURRENT QUERY, and
            2. any still-relevant PAST QUERIES if they are intentionally preserved in the cumulative report.

            ### EVIDENCE MATCHING RULE
            - For claims related to the CURRENT QUERY:
            - they MUST be supported by the CURRENT EVIDENCE.
            - For claims carried forward from PAST QUERIES:
            - they may be supported by previously validated past evidence and do NOT need to appear again in the CURRENT EVIDENCE.
            - Do NOT mark a past finding as hallucinated merely because it is absent from the CURRENT EVIDENCE, if it belongs to a past query and is being legitimately preserved.

            ### CONSISTENCY RULE
            - Past findings should remain in the cumulative report unless:
            - they contradict newer evidence,
            - they are no longer relevant, or
            - they are materially misstated in the current report.

            ### FINAL FOLLOW-UP REQUIREMENT
            - The cumulative report should correctly integrate both past and current findings.
            - If the user request or workflow expects the report to answer both past and current queries together, validate coverage for both.
            - If the current turn only requires an incremental addition, do not reject merely because the full past analysis is not restated verbatim, as long as preserved past findings remain correct.
        """
    
    audit_unit_tests = f"""
    ### MANDATORY VALIDATION TESTS:
    - [MATH]: Is the result exactly {math_info}? (Mandatory for TYPE A/C).
    - [DIVERGENCE]: For TYPE C, does the ANALYSIS section explicitly contrast Transcript optimism vs 10-K risk?Does the report identify if the Transcript's positive outlook is contradicted by specific 'Risk Factors' or 'Notes' in the 10-K? (e.g., CEO says 'Growth' but 10-K says 'Market share loss').
    - [CUMULATIVE]: Are PAST math calculations and current math calcualtions: {past_queries} {current_query}both present in report?
    - [FIDELITY]: Does the 'EVIDENCE TABLE' source/page match the provided [COORD] headers in {context_str} exactly?
    - [TEST_CUMULATIVE]: Are findings from {past_queries} still accurately preserved in the report?
    - [COVERAGE]: Does it answer the CURRENT QUERY: "{current_query}" AND preserve relevant PAST QUERIES: {past_queries}?
    """

    # ADVANCED PROMPT: Separates Data Validation from Writing Validation
    audit_prompt = raw_template.format(
        mode_instruction=mode_instruction,
        math_val=math_val,
        math_info=math_info,
        planner_tasks=planner_tasks,
        audit_unit_tests=audit_unit_tests,
        context_str=context_str,
        generated_report=generated_report,
        current_query=current_query,
        past_queries=past_queries
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

    mlflow.log_metric("hallucination_score", output.hallucination_score, step=current_turn)
    mlflow.log_metric("divergence_score", output.divergence_score, step=current_turn)
    mlflow.log_metric("traceability_score", output.traceability_score, step=current_turn)
    mlflow.log_metric("math_score", output.math_score, step=current_turn)

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
    target_node = output.target_node

    if target_node == "human_review":
        audit_status = "VERIFIED_BY_AUDITOR"
    else:
        audit_status = "NEEDS_CORRECTION_ROUTING_TO_GENERATOR"

    if (output.math_score < 1.0 or output.traceability_score < 1.0 or output.hallucination_score < 1.0):
        needs_revision = True
        target_node = "generator"
        audit_status = "FORCED_REJECTION_BY_SCORE_INTEGRITY"

    if needs_revision:
        current_attempts += 1
        if current_attempts >= 3:
            target_node = "human_review"
            needs_revision = False
            audit_status = "MAX_AUDIT_ATTEMPTS_REACHED_ESCALATED_TO_HUMAN_REVIEW"
    else:
        target_node = "human_review"
        audit_status = "VERIFIED_BY_AUDITOR"

    return {
        "turn_count": current_turn,
        **node_results,
        "reflection_feedback": output,
        "audit_status": audit_status,
        "target_node": target_node,
        "audit_attempts": current_attempts,
        "critique": output.critique,
        "steps": [
            f"Audit attempt: {current_attempts}",
            f"Target Node: {target_node}",
            f"Feedback: {output.critique[:100]}..."
        ]
    }