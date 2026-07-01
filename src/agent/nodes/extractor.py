from langchain_openai import ChatOpenAI # Ensure correct import
from src.agent.state import AgentState, MathPlan
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
import time
from src.utils.monitoring import log_to_mlflow
from src.utils.prompt_manager import PromptManager
from langchain_core.prompts import ChatPromptTemplate
from custom_logging import logger


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

def math_extractor_node(state: AgentState):
    # 1. Access the context from your Retriever (final_contexts)
    # We use .get() to avoid KeyErrors if the retriever failed
    start_ts= time.time()
    contexts = state.get("context", [])
    current_turn = state.get("turn_count", 0) + 1

    try:
        node_config = promptloader.prompts.get('extractor', {})
        raw_template = node_config.get('extractor_prompt')
        prompt_version = node_config.get('version', '1.0.0')

    except Exception as e:
        logger.exception(e)
        return {
             "extractor_failed": True
       
        }
    query=state.get("query")

    if not contexts:
        return {"steps": ["No context found. Skipping extraction."]}

    # 2. Flatten the evidence for the LLM
    kb_contexts=[c for c in contexts if c["source"]=="10K"]
    context_text = "\n\n".join([c['evidence'] for c in kb_contexts])
    
    # 3. Setup the Structured LLM
    # Note: Use 'gpt-4o' (not 'gpt-40') for high-precision extraction
   
    # 4. The Prompt (Grounding the LLM)
    extractor_prompt = raw_template.format(
        query=query,
        context_text=context_text
    )
    
    structured_llm = resilient_brain.with_structured_output(MathPlan,include_raw=True)
    
    try:
    # 5. Execute
        extraction = structured_llm.invoke(
        extractor_prompt, 
        config={"callbacks": [perf_cb]}
    )
    except Exception:
        logger.exception(
            f"Extractor failed | prompt_version={prompt_version}"
        )
        return {
            "extractor_failed": True
        }
    plan_output= extraction["parsed"]
    metrics_getter= get_node_metrics("math_extractor",extraction,perf_cb,start_ts)
    
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version

    try:
        log_to_mlflow("math_extractor",node_results,step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")

     
    metric_summary= ", ".join([f"{m.label}:{m.value}"for m in plan_output.metrics])
    return {
        "math_plan": plan_output,
        "turn_count": current_turn,
        **node_results,
        "math_context": context_text,
        "steps": [
            f" MATH EXTRACTION COMPLETE:",
            f"- Data Source: Analyzed {len(kb_contexts)} 10-K snippets.",
            f"- Metrics Extracted: {len(plan_output.metrics) if plan_output.metrics else 0} values captured.",
            f"- Inventory: [{metric_summary}]",
            f"- Normalization: All units converted to absolute integers for Python REPL compatibility."
        ]
    }
    