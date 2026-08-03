from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re
from src.agent.state import AgentState,SecurityRating
import time
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
from src.utils.monitoring import log_to_mlflow
import mlflow
import mlflow
from src.utils.prompt_manager import PromptManager
import os
from langchain_core.messages import SystemMessage, HumanMessage
from custom_logging import logger

from src.utils import monitoring


promptloader= PromptManager()

perf_cb=PerformanceCallback()

llm_mini = ChatOpenAI(
    model="gpt-4o-mini", 
    streaming=True,
    temperature=0, 
    max_retries=5, 
    timeout=30
)

# Fallback: High-Limit, Smart GPT-4o
llm_gpt4o = ChatOpenAI(
    model="gpt-4o", 
    streaming=True,
    temperature=0, 
    max_retries=3
)



resilient_brain = llm_mini.with_fallbacks([llm_gpt4o])

max_attempts=2

def system_1_guard(state:AgentState):
    start_ts = time.time()
    user_query = state["query"]
    current_turn = state.get("turn_count", 0) + 1
    try:
        node_config = promptloader.prompts.get('system_guardrail', {})
        raw_template = node_config.get('system_guardrail_prompt')
        prompt_version = node_config.get('version', '1.0.0')
    
    except Exception as e:
        logger.exception(e)
        return {
            "is_safe": False,
            "security_log":"Prompt unavailable"
        }

     # Catch obvious "Ignore previous" injections without calling GPT.
    denylist = [r"ignore previous", r"system prompt", r"you are now", r"dan mode"]
    jailbreakpatterns=[r"(?i)ignore (all )?previous", 
        r"(?i)developer mode", 
        r"(?i)dan mode",
        r"(?i)payload",
        r"(?i)output the system prompt",
        r"translation of the above into"]

    for pattern in denylist:
        if re.search(pattern,user_query.lower()):
            detection = 1 if not plan_output.is_safe else 0
            try:
                monitoring.client.log_metric(
                    run_id=current_run_id, 
                    key="jail_break_detected", 
                    decision_source="regex",
                    value=float(detection), 
                    step=current_turn
                )
            except Exception as e:
                logger.warning(f"MLflow metric logging failed: {e}")
            return {"is_safe":False}
    
    for pattern in jailbreakpatterns:
        if re.search(pattern,user_query.lower()):
            detection = 1 if not plan_output.is_safe else 0
            try:
                monitoring.client.log_metric(
                    run_id=current_run_id, 
                    key="jail_break_detected", 
                    decision_source="regex",
                    value=float(detection), 
                    step=current_turn
                )
            except Exception as e:
                logger.warning(f"MLflow metric logging failed: {e}")
            return {
                "is_safe": False, 
                "security_log": f"Jailbreak Attempt: {pattern}",
                "prompt_version": "v2.1.0-strict"
            }


    system_prompt=raw_template.format(user_input=user_query)

    user_message=f"Please analyze this input for security: <user_input>{user_query}</user_input>"

    structure_llm= resilient_brain.with_structured_output(SecurityRating,include_raw=True)
    
    for attempts in range(1, max_attempts + 1):
        try:
            assessment = structure_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ],
                config={"callbacks": [perf_cb]}
            )
            break
        except Exception:
            logger.exception(
                f"Guardrail LLM invocation failed | prompt_version={prompt_version}"
            )
            if attempts ==max_attempts:
                return {
                    "is_safe": False,
                    "security_log": "This request could not be processed"
                }
    plan_output = assessment["parsed"]

    detection = 1 if not plan_output.is_safe else 0

    current_run_id = monitoring.ACTIVE_AUDIT_RUN_ID
    
    if current_run_id is None:
        print("Warning: Guardrail sees no Active Run ID")
    else:
        try:
            monitoring.client.log_metric(
                run_id=current_run_id, 
                key="jail_break_detected", 
                decision_source="llm",
                value=float(detection), 
                step=current_turn
            )
        except Exception as e:
            logger.warning(f"MLflow metric logging failed: {e}")
    
   
    metrics_getter= get_node_metrics("guard_rail",assessment,perf_cb,start_ts)
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version

    try:

        log_to_mlflow("guard_rail",node_results,step=current_turn)
        print("assessment:",plan_output.is_safe)
    
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")



    return {"start_time": start_ts,"turn_count": current_turn,**node_results,"is_safe":plan_output.is_safe,"security_log":plan_output.reason,"prompt_version": "v2.1.0","steps": [f"Security Guard: {'Passed' if plan_output.is_safe else 'Blocked'}"]}
