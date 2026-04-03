import time
from src.agent.state import AgentState

def monitor_node(node_name,start_time,model_name,response):
    def wrapper(state:AgentState):
        #capture latency
        node_latency= time.time()-start_time

        #extract tokens/cost from the result metadata
        usage= response.get("usage_metadata",{"output_tokens":0})
        ttft=response.get("ttft", 0)
        cost= calculate_cost(usage,model_name)
        tps= round(usage['output_tokens']/node_latency,2) if node_latency>0 else 0


        metrics={
            f"node_benchmarks":{
                    node_name :{
                    "ttft": ttft,
                    "latency": round(node_latency, 3),
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "tokens": usage["total_tokens"],
                    "cost": cost,
                    "model": model_name,
                    "tps": tps
                }
                }
            }
        
        return {**metrics}

    return wrapper


def calculate_cost(usage:dict,model_name:str)->float:
    """Calculates the USD cost of a single LLM call."""
    input_t = usage.get("input_tokens", 0) or usage.get("prompt_tokens",0)
    output_t = usage.get("output_tokens", 0) or usage.get9("completion_tokens",0)

    # 2026 Rates Registry
    pricing = {
        "gpt-40-mini": (0.4,1.6),
    }

    input_rate,output_rate= pricing.get(model_name,(2.50,10.00))

    cost = (input_t * input_rate / 1_000_000) + (output_t *output_rate / 1_000_000)
    return cost
