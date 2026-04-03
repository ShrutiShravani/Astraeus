import os
import psutil
import chromadb
from langchain_core.callbacks import BaseCallbackHandler
import time
import mlflow

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
    mem_info=process.memory_info().rss/(1024*1024)
    cpu_usage= psutil.cpu_percent(interval=1)

    print(f"--- SERVER STATS ---")
    print(f"RAM Usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"CPU Usage: {cpu_usage}%")

    return mem_info,cpu_usage



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
    called after ndoe to log emtrics epr ndoe
    """
    benchmarks = node_results.get("node_benchmarks", {})
    prompt_ver = node_results.get("prompt_version", "v0.0.0")
    mlflow.log_param(f"node_{node_name}_prompt_version", prompt_ver)
    metrics = benchmarks.get(node_name, {})
    if metrics:
        # Use unique keys for every metric type!
        model_used = metrics.get("model", "unknown_model")
        mlflow.log_param(f"node_{node_name}_model", model_used)
        mlflow.log_metric(f"node_{node_name}_ttft", metrics.get("ttft", 0), step=step)
        mlflow.log_metric(f"node_{node_name}_input_tokens", metrics.get("input_tokens", 0), step=step)
        mlflow.log_metric(f"node_{node_name}_output_tokens", metrics.get("output_tokens", 0), step=step)
        mlflow.log_metric(f"node_{node_name}_total_tokens", metrics.get("tokens", 0), step=step)
        mlflow.log_metric(f"node_{node_name}_cost", metrics.get("cost", 0), step=step)
        mlflow.log_metric(f"node_{node_name}_latency", metrics.get("latency", 0), step=step)
        mlflow.log_metric(f"node_{node_name}_tps", metrics.get("tps", 0), step=step)
    