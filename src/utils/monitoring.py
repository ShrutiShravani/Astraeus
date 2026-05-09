from asyncio.tasks import run_coroutine_threadsafe
import os
import psutil
import chromadb
from langchain_core.callbacks import BaseCallbackHandler
import time
import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import Summary, Gauge, Counter
from contextvars import ContextVar


GLOBAL_RUN_ID = None
client = MlflowClient()

audit_run_id_var: ContextVar[str] = ContextVar("audit_run_id", default=None)
# This will hold the run_id for the current query
ACTIVE_AUDIT_RUN_ID = None 

def set_active_run(run_id):
    audit_run_id_var.set(run_id)

def get_active_run()->str:
    val= audit_run_id_var.get()
    if val is not None:
        return val
    
    # Fallback to the Global if ContextVar is lost in async tasks
    return GLOBAL_RUN_ID


class PerformanceCallback(BaseCallbackHandler):
    def __init__(self):
        self.start_time=None
        self.ttft=None
    
    def on_llm_start(self,serialized,prompts,**kwargs):
        self.start_time=time.time()
    
    def on_llm_new_token(self,token,**kwargs):
        if self.ttft is None and self.start_time:
            self.ttft = time.time() - self.start_time
    def on_llm_end(self,response,**kwargs):
        self.latency= time.time()-self.start_time

def log_system_usage():
    process=psutil.Process(os.getpid())
    mem_info=process.memory_info()
    cpu_usage= psutil.cpu_percent(interval=None)

    rss_mb = float(mem_info.rss / 1024 / 1024)

    print(f"--- SERVER STATS ---")
    print(f"RAM Usage: {rss_mb} MB")
    print(f"CPU Usage: {cpu_usage}%")

    return rss_mb,cpu_usage



def check_system_health(host="localhost", port=8000, ram_threshold=90):
    # 1. Check Disk Usage of the Chroma Folder
    client = chromadb.HttpClient(host=host, port=port)
    
    # 1. Check Connectivity instead of Disk
    try:
        heartbeat = client.heartbeat()
        print(f"--- SERVER HEALTH CHECK ---")
        print(f"Chroma Status: ONLINE (Heartbeat: {heartbeat})")
    except Exception as e:
        print(f"CRITICAL: Cannot connect to Chroma at {host}:{port}")
        return False
    

    # 2. Check System RAM Usage (The real 90% risk)
    ram_usage = psutil.virtual_memory().percent
    
    print(f"--- LOCAL HEALTH CHECK ---")
    print(f"System RAM: {ram_usage}%")
    
    if ram_usage > ram_threshold:
        print(f"CRITICAL WARNING: RAM usage is at {ram_usage}%. Exceeds {ram_threshold}% limit!")
        return False # Trigger shutdown or block new audits
        
    return True

def log_to_mlflow(node_name,node_results,step):
    """
    called after node to log metrics per node
    """
    ACTIVE_AUDIT_RUN_ID = get_active_run()
    
    if ACTIVE_AUDIT_RUN_ID is None:
        print(f"DEBUG: MLflow skip - No active run for {node_name}")
        return
    benchmarks = node_results.get("node_benchmarks", {})
    prompt_ver = node_results.get("prompt_version", "v0.0.0")
    client.log_param(ACTIVE_AUDIT_RUN_ID,f"node_{node_name}_prompt_version", prompt_ver)
    # In main.py
    mem_mb, cpu_usage = log_system_usage()
    
    metrics = benchmarks.get(node_name, {})

    if metrics:
        log_to_promtheus(node_name,metrics,mem_mb)
        # Use unique keys for every metric type!
        client.log_metric(ACTIVE_AUDIT_RUN_ID,F"NODE_{node_name}_mem_usage",mem_mb,step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID,F"NODE_{node_name}_cpu_usage",cpu_usage,step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID,F"NODE_{node_name}_ttft",metrics.get("ttft",0),step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID, f"node_{node_name}_input_tokens", metrics.get("input_tokens", 0), step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID, f"node_{node_name}_output_tokens", metrics.get("output_tokens", 0), step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID, f"node_{node_name}_total_tokens", metrics.get("tokens", 0), step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID, f"node_{node_name}_cost", metrics.get("cost", 0), step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID, f"node_{node_name}_latency", metrics.get("latency", 0), step=step)
        client.log_metric(ACTIVE_AUDIT_RUN_ID, f"node_{node_name}_tps", metrics.get("tps", 0), step=step)
        
        # For params (model/version), client uses log_param
        client.log_param(ACTIVE_AUDIT_RUN_ID, f"node_{node_name}_model", metrics.get("model", "unknown"))



NODE_LATENCY= Summary('audit_node_latency_seconds','latency_per_node',['node_name'],buckets=(1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float("inf")))
NODE_TOKENS = Counter('audit_node_tokens_total', 'Total tokens used', ['node_name', 'token_type'])
SYS_MEM = Gauge('audit_sys_memory_mb', 'RAM usage during node execution', ['node_name'])

def log_to_promtheus(node_name,metrics,mem_mb):
    latency=metrics.get("latency",0)
    NODE_LATENCY.labels(node_name=node_name).observe(latency)

    #log memory usage
    SYS_MEM.labels(node_name=node_name).set(mem_mb)
    # Log Tokens
    NODE_TOKENS.labels(node_name=node_name, token_type='input').inc(metrics.get("input_tokens", 0))
    NODE_TOKENS.labels(node_name=node_name, token_type='output').inc(metrics.get("output_tokens", 0))