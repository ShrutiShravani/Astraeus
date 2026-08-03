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
from custom_logging import logger



promptloader= PromptManager()

perf_cb=PerformanceCallback()


# Primary: Cheap & Fast 
llm_mini = ChatOpenAI(model="gpt-4o-mini",streaming=True,temperature=0, max_retries=5)

# Backup: Smart & High-Limit (The Fallback)
llm_gpt4o = ChatOpenAI(model="gpt-4o",streaming=True, temperature=0)

# Resilient Brain for everyone
resilient_brain = llm_mini.with_fallbacks([llm_gpt4o])
max_attempts=2

def planner_node(state:AgentState):
    start_ts=time.time()
    failed_tasks = state.get("failed_tasks", [])
    current_query = failed_tasks if failed_tasks else state["query"]
    current_turn = state.get("turn_count", 0) + 1
    history = state.get("query_history", [])
    prev_company = state.get("target_company",[])
    print(prev_company)
    prev_year = state.get("target_year",[])
    print(prev_year)
    
    retrieved_feedback= state.get("retriever_feedback",[])
    print(retrieved_feedback)
  
    if retrieved_feedback:
        critique_text = getattr(retrieved_feedback, "retriever_critique", "") or "No specific critique provided."
        failed_titles = ", ".join([t.title for t in failed_tasks]) if failed_tasks else "the previously failed tasks"
 
        planner_instruction += f"""
    
        RETRY MODE — PRIOR RETRIEVAL FAILED FOR SOME TASKS:
        The following task(s) could not be found in the last retrieval
        attempt: {failed_titles}
    
        Auditor feedback on why retrieval failed:
        "{critique_text}"
    
        RULES FOR THIS RETRY:
        - Create tasks ONLY for the failed task(s) listed above. 
        - Use the auditor's feedback to adjust HOW you search, not just
        repeat the same failed task unchanged. For example: if the
        feedback suggests the metric is discussed narratively rather
        than in a table, change `zone` from "Item 8" to "Item 7".
        If it suggests management commentary rather than filed
        numbers, consider `doc_source="Transcript"` instead of "10K".
        - Keep extracted_company and extracted_year the same as before
        UNLESS the feedback specifically indicates they were wrong.
        """

    
    try:
        node_config = promptloader.prompts.get('planner', {})
        raw_template = node_config.get('planner_prompt')
        if not raw_template:
            raise ValueError("planner prompt not found")
        prompt_version = node_config.get('version', '1.0.0')
    except Exception as e:
        logger.exception(e)
        return {
             "planner_failed": True
       
        }

    audit_wiki=state.get("audit_wiki",[])
    already_verified_facts= ",".join([f"Year: {item.year} | Company: {item.company} | Task: {item.task_name} | Evidence :{item.evidence} in {item.source} p.{item.page}"for item in audit_wiki])

        
    if len(history) > 5:
        print("length exceeded")
        return {
            "target_node": "END", # Or route to a 'limit_exceeded' node
            "steps": ["RECURSION_LIMIT: Max follow-up depth reached (2/2)."]
        }
  
    structured_planner= resilient_brain.with_structured_output(Planner,include_raw=True)
    planner_prompt  = raw_template.format(
        planner_instruction=planner_instruction,
        report_summary="DEPRECATED: Refer to audit_wiki for facts",
        current_query=current_query,
        audit_wiki=already_verified_facts, # <--- THIS IS THE "FACT BRIDGE"
        locked_company=prev_company,
        locked_year=prev_year
    )

   
    # Invoke the LLM directly with the string
    for attempts in range(1, max_attempts + 1):
        try:
            raw_result = structured_planner.invoke(planner_prompt,config={"callbacks": [perf_cb]})
            break
        except Exception:
            logger.exception(
                f"Planner LLM invocation failed | prompt_version={prompt_version}"
            )
            if attempts ==max_attempts:
                return {
                    "planner_failed": True
                }


    # 5. Extracting data from the 'include_raw' format
    plan_output = raw_result["parsed"] # This is your 'Planner' object
   

    metrics_getter= get_node_metrics("planner",raw_result,perf_cb,start_ts)
    
   
    print(f"Plan Created: {plan_output.tasks}")

    if not plan_output.tasks: 
        logger.error(
        f"Planner returned empty task list | prompt_version={prompt_version}"
        )
        return {
            "planner_failed": True
        }

    #JOIN ALL TASKS 
    
    all_extracted_year=list(set([t.extracted_year for t in plan_output.tasks]))
    all_extracted_company=list(set([t.extracted_company for t in plan_output.tasks]))
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version
    
    try:
        log_to_mlflow("planner",node_results,step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")
 
    print(plan_output)

  
    return {
        "plan": plan_output.tasks,
        "turn_count": current_turn,
        **node_results,
        "type": plan_output.type,
        "target_company": all_extracted_company, # Ensure this isn't None!
        "target_year":  all_extracted_year,
        "steps": ["Created search plan for query: " + state["query"],f"query_type:{plan_output.type}"]
    }


