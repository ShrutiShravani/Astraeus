import numpy as np
import mlflow
import pandas as pd


import numpy as np
import mlflow
import pandas as pd

# Define SLAs (Service Level Agreements)
latency_thresholds = {
    "planner": 5.0,    
    "retriever": 12.0, # Given 10-K document sizes, 12s is a fair SLA
    "auditor": 8.0     
}

def production_live_report(experiment_name="Astraeus_Forensic_Audit"):
    # 1. Fetch data from MLflow Tracking Server
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    
    if runs.empty:
        print(f"No runs found for experiment: {experiment_name}")
        return

    nodes = ["planner", "retriever", "generator", "auditor"]
    
    print(f"\n{'--- PRODUCTION PERFORMANCE & SLA AUDIT ---':^60}")
    print(f"{'NODE NAME':<20} | {'AVG':<7} | {'P95':<7} | {'P99':<7} | {'STATUS'}")
    print("-" * 75)

    for node in nodes:
        # NOTE: Verify if your keys are metrics.planner_latency or metrics.node_planner_latency
        col_name = f"metrics.{node}_latency" 
        
        if col_name in runs.columns:
            times = runs[col_name].dropna()
            
            if not times.empty:
                avg = times.mean()
                p95 = np.percentile(times, 95)
                p99 = np.percentile(times, 99)
                
                # SLA Check
                sla_limit = latency_thresholds.get(node, 15.0)
                status = "PASS" if p99 <= sla_limit else f"FAIL (SLA: {sla_limit}s)"
                
                print(f"{node:<20} | {avg:>6.2f}s | {p95:>6.2f}s | {p99:>6.2f}s | {status}")
                
                if p99 > sla_limit:
                    print(f" ALERT: {node.upper()} Tail Latency (P99) is critical!")

    # 2. Overall E2E Logic
    if f"metrics.{node}_latency" in runs.columns:
        e2e = runs["metrics.latency"].dropna()
        print("-" * 75)
        print(f"{'OVERALL E2E':<20} | {e2e.mean():>6.2f}s | {np.percentile(e2e, 95):>6.2f}s | {np.percentile(e2e, 99):>6.2f}s |")

if __name__ == "__main__":
    production_live_report()
  