import pandas
from src.agent.state import AgentState,ForensicArchive
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
import re
from custom_logging import logger
from openai import RateLimitError


promptloader= PromptManager()
perf_cb=PerformanceCallback()

llm_mini = ChatOpenAI(model="gpt-4o-mini",streaming=True,temperature=0, max_retries=3)

    # Snapshot Version (Secondary Backup)
llm_pro_snap = ChatOpenAI(model="gpt-4o-2024-08-06",streaming=True, temperature=0)

# Cheap & Fast (Final Safety Net)
llm_pro = ChatOpenAI(model="gpt-4o",streaming=True, temperature=0, max_retries=5)

# 2. THE CHAIN OF COMMAND
# For heavy tasks (Generator/Auditor): 
# Try GPT-4o -> Try Snapshot -> Try Mini (Fail-safe)
resilient_pro = llm_pro.with_fallbacks([llm_pro_snap, llm_mini])
resilient_mini = llm_mini.with_fallbacks([llm_pro])


def compaction_node(state: AgentState):
    """
    Synthesizes audit_wiki into a high-density 'wiki_archive' and clears the wiki.
    """
    start_ts = time.time()
    audit_wiki = state.get("audit_wiki", [])
    current_turn = state.get("turn_count", 0) + 1
    try:
        node_config = promptloader.prompts.get('compaction_node', {})
        raw_template = node_config.get('compaction_prompt')
        if not raw_template:
            raise ValueError("compaction prompt not found")
        prompt_version = node_config.get('version', '1.0.0')
    except Exception as e:
        logger.exception(e)
        return {
             "compaction_node_failed": True
       
        }

    
    # 2. Define the Structured Compressor
    # Use resilient_mini (gpt-4o-mini) here for cost efficiency
    structured_compressor = resilient_mini.with_structured_output(ForensicArchive, include_raw=True)
    
    compaction_prompt = f"""
    You are a Forensic Ledger Keeper. 
    Synthesize the following raw audit findings into a clean, markdown table.
    
    Raw Findings:
    {audit_wiki}
    
    Format the table with these columns: | Year | Metric | Value | Source | Page | Forensic Fact |
    """
    
    try:
        # Use ainvoke for async flow
        response =  structured_compressor.ainvoke(compaction_prompt, config={"callbacks": [perf_cb]})
        parsed_result = response["parsed"]
        new_table_rows = parsed_result.summary_table
    except Exception as e:
        logger.exception("Compaction LLM invocation failed")
        return {"compaction_failed": True}

    # 3. Update Archive
    current_archive = state.get("wiki_archive", "")
    new_archive = f"{current_archive}\n{new_table_rows}".strip()
    
    # 4. Capture Metrics (Reuse your node_metrics logic)
    metrics_getter = get_node_metrics("compaction_node", response, perf_cb, start_ts)(state)

    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version
    
    try:
        log_to_mlflow("planner",node_results,step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")
    
    return {
        "turn_count": current_turn,
        **node_results,
        "wiki_archive": new_archive,
        "audit_wiki": [], # RESET: Clearing the working paper
        "steps": ["Audit wiki compacted into archive table"]
    }