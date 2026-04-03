from src.utils.observability import monitor_node
import json
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import threading
from deepeval.metrics import ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
import mlflow
import asyncio

def get_node_metrics(node_name,raw_result,perf_cb,start_time):
    response_metadata={
        "usage_metadata":raw_result["raw"].usage_metadata,
        "ttft": perf_cb.ttft if perf_cb.ttft else 0

    }
    metrics_getter=monitor_node(node_name=node_name,
        start_time=start_time,
        model_name="gpt-40-mini",
        response=response_metadata)
    
    return metrics_getter



def log_heavy_metrics(query,context,prompt_version,current_turn):
    try:
       active_run= mlflow.active_run()
       run_id = active_run.info.run_id if active_run else None
    except:
        run_id = None

    def run_eval():
        loop= asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        test_case = LLMTestCase(
            input=query,
            actual_output="N/A", # Not needed for retriever metrics
            retrieval_context=context
        )

        relevancy_metric = ContextualRelevancyMetric(threshold=0.5, model="gpt-4o-mini")
        relevancy_metric.measure(test_case)

        # 2. Are the best chunks at the top? (Precision)
        precision_metric = ContextualPrecisionMetric(threshold=0.5, model="gpt-4o-mini")
        precision_metric.measure(test_case)
        
        
        final_relevancy_score = (relevancy_metric.score * 0.5) + (precision_metric.score * 0.5)
        combined_reason = f"Relevancy: {relevancy_metric.reason} | Precision: {precision_metric.reason}"
        


        scores={
            "deepeval_relevancy": round(relevancy_metric.score, 3),
            "deepeval_precision": round(precision_metric.score, 3),
            "final_relevancy_score": round(final_relevancy_score, 3),
            "reason":combined_reason
        }
        
        if run_id:
            client=MlflowClient()
            for metric,value in scores.items():
                if isinstance(value,(int,float)):
                    client.log_metric(metric,value,step=current_turn)
                    client.set_tag(f"retriever_reason_{current_turn}_{prompt_version}",combined_reason)

    thread = threading.Thread(target=run_eval)
    thread.start()