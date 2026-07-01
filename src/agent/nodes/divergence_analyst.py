from src.agent.state import AgentState,DivergenceAnalyst
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
import time
from src.utils.monitoring import log_to_mlflow
from src.utils.prompt_manager import PromptManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from custom_logging import logger

parser = JsonOutputParser(pydantic_object=DivergenceAnalyst)
promptloader= PromptManager()
perf_cb=PerformanceCallback()

llm_pro = ChatOpenAI(model="gpt-4o",streaming=True,temperature=0, max_retries=3)

    # Snapshot Version (Secondary Backup)
llm_pro_snap = ChatOpenAI(model="gpt-4o-2024-08-06",streaming=True, temperature=0)

# Cheap & Fast (Final Safety Net)
llm_mini = ChatOpenAI(model="gpt-4o-mini",streaming=True, temperature=0, max_retries=5)

# 2. THE CHAIN OF COMMAND
# For heavy tasks (Generator/Auditor): 
# Try GPT-4o -> Try Snapshot -> Try Mini (Fail-safe)
resilient_pro = llm_pro.with_fallbacks([llm_pro_snap, llm_mini])
resilient_mini = llm_mini.with_fallbacks([llm_pro])

def divergence_analyst_node(state:AgentState):
    start_ts=time.time() 
    current_turn = state.get("turn_count", 0) + 1
    raw_context = state.get("final_context","")
    raw_context = [
        {
        "id": c.get("id"),
        "source": c.get("source"),
        "page": c.get("page"),
        "evidence": c.get("evidence")
    } for c in raw_context
    ]

    calculation_result= state.get("calculation_result")

    try:
        node_config = promptloader.prompts.get('divergence_analyst', {})
        raw_template = node_config.get('divergence_analyst_prompt')
        prompt_version = node_config.get('version', '1.0.0')
    
    except Exception as e:
        logger.exception(e)
        return {
             "divergence_analyst_failed": True
       
        }

    # Fix: Cleaned up the double prompt nesting and clarified Type C instruction
    system_prompt = raw_template.format(
        query=state["query"],
        final_context=raw_context,
        calculation_result=calculation_result or "No calculations available"
      
    )

    structured_divergence = resilient_pro.with_structured_output(
    DivergenceAnalyst,
    include_raw=True
     )

    messages= [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
]
    
    try:

        result = structured_divergence.invoke(messages ,config={"callbacks":[perf_cb]})
        print(f"result:{result}")
    except Exception:
        logger.exception(
            f"Divergnec Analyst LLM invocation failed | prompt_version={prompt_version}"
        )
        return {
            "divergence_analyst_failed": True
        }
    
    
    analysis_output = result["parsed"]

    #get raw results for metrics calculation
    raw_res_metrics= {
        "raw":result["raw"],
        "parsed":analysis_output
    }

    metrics_getter= get_node_metrics("divergence_analyst",raw_res_metrics,perf_cb,start_ts)
    
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version

    try:
        log_to_mlflow("divergence_analyst",node_results,step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")

    return{
        "alignment_status":
        analysis_output.alignment_status,

        "narrative_conflict_score":
            analysis_output.narrative_conflict_score,

        "divergence_type":
            analysis_output.divergence_type,

        "divergence_reason":
            analysis_output.divergence_reason,

        "conflict_rationale":
            analysis_output.conflict_rationale,

        "turn_count": current_turn,

        **node_results,
        "steps": [
        f"Divergence Analysis Complete",
        f"Alignment Status: {analysis_output.alignment_status}",
        f"Divergence Type: {analysis_output.divergence_type}",
        f"Conflict Score: {analysis_output.narrative_conflict_score}"
    ]

   }