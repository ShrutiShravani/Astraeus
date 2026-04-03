from src.agent.state import AgentState,Retriever_feedback
import time
from src.utils.monitoring import log_to_mlflow
from langchain_openai import ChatOpenAI
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
from src.utils.monitoring import log_to_mlflow
import mlflow
from src.utils.prompt_manager import PromptManager
from langchain_core.prompts import ChatPromptTemplate

promptloader= PromptManager()
perf_cb=PerformanceCallback()
llm_pro = ChatOpenAI(model="gpt-4o-mini", 
    temperature=0,streaming=True, max_retries=3)

# Snapshot Version (Secondary Backup)
llm_pro_snap = ChatOpenAI(model="gpt-4o", streaming=True,
    temperature=0)

# Cheap & Fast (Final Safety Net)
llm_mini = ChatOpenAI(model="gpt-4.1-mini",streaming=True,
    temperature=0, max_retries=5)

# 2. THE CHAIN OF COMMAND
# For heavy tasks (Generator/Auditor): 
# Try GPT-4o -> Try Snapshot -> Try Mini (Fail-safe)
resilient_pro = llm_pro.with_fallbacks([llm_pro_snap, llm_mini])

def retrieval_auditor_node(state:AgentState):
    start_ts=time.time()
    current_query = state["query"]
    current_turn = state.get("turn_count", 0) + 1
    structured_llm =  resilient_pro.with_structured_output(Retriever_feedback,include_raw=True)
    context= state.get("context")
    node_config = promptloader.prompts.get('retriever_auditor', {})
    raw_template = node_config.get('retriever_auditor_prompt')
    prompt_version = node_config.get('version', '1.0.0')
    current_attempts = state.get("retriever_audit_attempts", 0)
  
    
    retriever_audit_prompt= raw_template.format(
        current_query=current_query,
        context=context
    )
     
    reflection = structured_llm.invoke(
    retriever_audit_prompt, 
    config={"callbacks": [perf_cb]}
)
    output= reflection["parsed"]
    retriever_critique = output.retriever_critique.lower()
    print(f"retriever_critique:{retriever_critique}")

    # 4. CAPTURE METRICS (The Senior-Level Step)
    metrics_getter = get_node_metrics(
        "retrieval_auditor",
        reflection, 
        perf_cb, 
        start_ts,
    )

    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version
    log_to_mlflow("retrieval_auditor",node_results,step=current_turn)

    needs_revision=output.needs_revision
    print(f"need_revision{needs_revision}")

    if needs_revision is True:
        print("needs_revision")
        current_attempts += 1
  
    
    return{
        "turn_count": current_turn,
        **node_results,
        "retriever_feedback": output,
        "retriever_audit_attempts": current_attempts,
        "critique":   retriever_critique,
        "steps": [
            f"Audit attempt: {current_attempts}",
            f"Feedback: {retriever_critique[:100]}..."
        ]
    }

