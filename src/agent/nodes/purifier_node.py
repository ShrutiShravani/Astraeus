from torch import monitor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import time
from src.agent.state import AgentState, PurifierOutput
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

def prompt_purifier_node(state:AgentState):
    start_ts=time.time()
    current_query = state["query"]
    current_turn = state.get("turn_count", 0) + 1
    history = state.get("query_history", [])
   
    print("loading prompt")
    try:
        node_config = promptloader.prompts.get('query_purifier', {})
        raw_template = node_config.get('query_purifier_prompt')
        if not raw_template:
            raise ValueError("purifier prompt not found")
        prompt_version = node_config.get('version', '1.0.0')
    except Exception as e:
        logger.exception(e)
        return {
             "prompt_purifier_failed": True
       
        }


    # Run your Purifier prompt here
    
    structured_purifier = resilient_brain.with_structured_output(PurifierOutput,include_raw=True)

    purifier_prompt  = raw_template.format(
        history=history[-1] if history else "No prior context",
        current_query=current_query

        )
    
    try:
        response = structured_purifier.invoke(purifier_prompt,config={"callbacks": [perf_cb]})

    except Exception:
        logger.exception(
            f"Planner LLM invocation failed | prompt_version={prompt_version}"
        )
        return {
            "prompt_purifier_failed": True
        }
    
    result = response["parsed"]
    print(result)
    metrics_getter= get_node_metrics("prompt_purifier",response,perf_cb,start_ts)
    
    
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version
    
    try:
        log_to_mlflow("prompt_purifier",node_results,step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")
 
  

    if result.action == "clarify":
        # By setting ask_user to True, your main.py loop will catch this
        return {"ask_user": True, 
        "clarification_question": result.clarification_question,
        "turn_count": current_turn,

        **node_results,
        "steps": ["Clarifictaion required for query: " + state["query"],f"clarifictaion_question:{result.clarification_question}"]}
    
    # Otherwise, return the rewritten query
    elif result.action=="rewritten_query":
        return {"query": result.rewritten_query, 
        "turn_count": current_turn,

            **node_results,
            "steps": ["Rewriting original query : " + state["query"],f"new_query:{result.rewritten_query}"]}
    else:
        return {"query": current_query, 
              "turn_count": current_turn,
            **node_results,
            "steps": ["Using original query : " + state["query"]]}