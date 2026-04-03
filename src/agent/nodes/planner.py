from torch import monitor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import time
from src.agent.state import AgentState, Planner
from dotenv import load_dotenv
import os
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
from src.utils.monitoring import log_to_mlflow
from src.utils.prompt_manager import PromptManager

promptloader= PromptManager()

perf_cb=PerformanceCallback()


# Primary: Cheap & Fast
llm_mini = ChatOpenAI(model="gpt-4o-mini",streaming=True,temperature=0, max_retries=5)

# Backup: Smart & High-Limit (The Fallback)
llm_gpt4o = ChatOpenAI(model="gpt-4o",streaming=True, temperature=0)

# Resilient Brain for everyone
resilient_brain = llm_mini.with_fallbacks([llm_gpt4o])

def planner_node(state:AgentState):
    start_ts=time.time()
    current_query = state["query"]
    current_turn = state.get("turn_count", 0) + 1
    history = state.get("query_history", [])
    prev_company = state.get("target_company")
    prev_year = state.get("target_year")
    retrieved_feedback= state.get("retriever_feedback",[])
    node_config = promptloader.prompts.get('planner', {})
    raw_template = node_config.get('planner_prompt')
    prompt_version = node_config.get('version', '1.0.0')
    planner_instruction = ""

    if retrieved_feedback:
        planner_instruction=f"""
        You are in a REVISION LOOP because the previous retrieval failed audit.
    ### REVISION MODE: TARGETED CORRECTION
    You are correcting a failed retrieval for the CURRENT query: "{state.get('query')}".
    The Auditor found a gap in the evidence.
    
    LAST AUDIT CRITIQUE:  {retrieved_feedback.retriever_critique}

    INSTRUCTION:
    1. Do NOT change the Target Company or Year. Keep Company: {prev_company} and Year: {prev_year}.
    2. Broaden keywords or shift Zones as requested by the Auditor.
    2. Change the 'Keywords' or 'Zone' specifically to address the [RETRIEVAL_GAP].
    3. If the Auditor said 'Transcript is missing', your NEW plan must have a Transcript task.

    DIRECTIONS:
    1. If the critique says 'Missing doc_source', you MUST add a task where doc_source='Transcript'or "10K".
    2. If the critique says 'Missing specific line item', you MUST broaden your keywords or target the specific 'Item' (e.g., Item 8 for tables).
    3. Do NOT simply repeat the old plan. If you repeat the old plan, the audit will fail again.
    
    """
    if len(history) > 1:
        return {
            "target_node": "END", # Or route to a 'limit_exceeded' node
            "steps": ["RECURSION_LIMIT: Max follow-up depth reached (2/2)."]
        }
    existing_report = state.get("generation", "No previous report exists.")
    if "1. EXECUTIVE SUMMARY" in existing_report:
        existing_report.split("1. EXECUTIVE SUMMARY")[-1].split("2. ANALYSIS")[0]
    else:
        report_summary = "No previous summary found. Treat this as a new investigation."

    structured_planner= resilient_brain.with_structured_output(Planner,include_raw=True)
    planner_prompt  = raw_template.format(
        planner_instruction=planner_instruction,
        history=history,
        report_summary=report_summary,
        current_query=current_query,
        )

   
    # Invoke the LLM directly with the string
    raw_result = structured_planner.invoke(planner_prompt,config={"callbacks": [perf_cb]})
    # 5. Extracting data from the 'include_raw' format
    plan_output = raw_result["parsed"] # This is your 'Planner' object
   

    metrics_getter= get_node_metrics("planner",raw_result,perf_cb,start_ts)
    
   
    print(f"Plan Created: {plan_output.tasks}")

    #JOIN ALL TASKS 
    task_rationales="|".join([t.rationale for t in plan_output.tasks])
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version

    log_to_mlflow("planner",node_results,step=current_turn)
 
    print(plan_output)
    print(plan_output.extracted_company.upper())
    print(plan_output.extracted_year)
  
    return {
        "plan": plan_output.tasks,
        "turn_count": current_turn,

        **node_results,
        "type": plan_output.type,
        "target_company": plan_output.extracted_company.upper(), # Ensure this isn't None!
        "target_year": plan_output.extracted_year,
        "steps": ["Created search plan for query: " + state["query"],f"query_type:{plan_output.type}",f"Audit Strategy: {plan_output.reasoning}", 
            f"Task Logic: {task_rationales}"]
    }


