from src.agent.state import AgentState, EvidenceFound
import time
from src.utils.monitoring import log_to_mlflow
from custom_logging import logger

def python_repl_node(state:AgentState):
    start_ts=time.time()
    plan= state["math_plan"]
    current_turn = state.get("turn_count", 0) + 1
    node_metrics = {
        "node_benchmarks": {
            "python_repl": {
                "ttft": 0, "latency": 0, "input_tokens": 0, "output_tokens": 0,
                "tokens": 0, "cost": 0.0, "model": "python-3.11-exec", "tps": 0
            }
        }
    }

    #prepare the sanbox environment
    #we create dictionary of variables based on the extrcated metrics
   
    safe_vars= {m.label:m.value for m in plan.metrics}
    safe_vars["result"] = None

    #isolated execution
    try:
        exec(plan.python_formula, {"__builtins__": {}}, safe_vars)
        
        #capture the result
        calc_value= safe_vars.get("result","Calculation error:varibale not set")
        if calc_value is None:
            calc_value = "Calculation error: 'result' variable was not assigned in the formula."

        
        
        print(f"python_formula:{plan.python_formula}")
    
    
        node_latency= time.time()-start_ts
        node_metrics["node_benchmarks"]["python_repl"]["latency"] = round(node_latency, 3)
        node_metrics["prompt_version"] = "v1.1.0"

        try:
            log_to_mlflow("python_repl", node_metrics, step=current_turn)
        except Exception as e:
            logger.warning(f"MLflow node logging failed: {e}")

    
    except Exception as e:
        calc_value=f"Python execution error :str{e}"
        logger.exception(e)
        return {
            "python_repl_failed": True
        }
    

    return {
        "calculation_result":calc_value,
        "turn_count": current_turn,
        **node_metrics,
        "steps": [f"Deterministic Sandbox: Executed formula '{plan.python_formula}'"]

    }
