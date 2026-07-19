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

def planner_node(state:AgentState):
    start_ts=time.time()
    current_query = state["query"]
    current_turn = state.get("turn_count", 0) + 1
    history = state.get("query_history", [])
    prev_company = state.get("target_company",[])
    print(prev_company)
    prev_year = state.get("target_year",[])
    print(prev_year)
    #current_existing_report = state.get("report_history", "")
    is_follow_up=state.get("is_follow_up")
    planner_instruction = ""

    if state.get("human_decision") == "is_investigate":
        # Keep only the relevant context
        history = [history] if history else []  
        planner_instruction = f"INVESTIGATE MODE:You are analyzing a thread of financial investigation.History of queries {history}.This was for {prev_company} {prev_year}."
      
    else:
        history = [] # Clean slate
        planner_instruction = "You are starting a fresh forensic audit."

    retrieved_feedback= state.get("retriever_feedback",[])
    print(retrieved_feedback)
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
    
        
    if is_follow_up:

        planner_instruction = f"""
        FOLLOW-UP MODE: GAP ANALYSIS

        CURRENT STATE:
        - Verified Facts: {already_verified_facts}
        - User Query: {current_query}

        RULES:

            YOUR ONLY JOB:
            Compare the USER QUERY against VERIFIED FACTS.

            STEP 1 — CHECK WIKI FIRST:
            For each metric the query needs, check if it 
            exists in VERIFIED  FACTS with the exact 
            company and year.

            STEP 2 — DECIDE:
            A) If ALL needed metrics are in the verified facts:
            → return an empty task list []. Do not re-retrieve data you already possess.
            → This tells the system to calculate from wiki

            B) If SOME metrics are missing from verified facts:
            → Create tasks ONLY for the missing metrics
            → Do NOT create tasks for metrics already in verified facts

            C) If query asks for a ratio (margin, growth rate):
            → Check if components are in verified facts
            → If yes → return an empty task list[]
            → If no → create tasks for missing components only

            STRICT RULE: 
            If a metric + year + company combination exists 
            in VERIFIED FACTS, you are FORBIDDEN from 
            creating a retrieval task for it. Violation = 
            wasted compute and audit failure.
            """

    else:
        planner_instruction=f"Analyze the user query,classify a financial query type and  build a comprehensive step by step task plan."
        
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
    try:
        raw_result = structured_planner.invoke(planner_prompt,config={"callbacks": [perf_cb]})
    except Exception:
        logger.exception(
            f"Planner LLM invocation failed | prompt_version={prompt_version}"
        )
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


