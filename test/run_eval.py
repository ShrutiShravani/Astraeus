import asyncio
import time
from src.agent.orchestrator import workflow
from src.utils.monitoring import set_active_run
import mlflow
from mlflow.tracking import MlflowClient
import os
from langchain_community.callbacks import get_openai_callback
from ragas.metrics import answer_relevancy,faithfulness
from ragas import evaluate
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
client = MlflowClient()
#print("MLFLOW_TRACKING_URI =", MLFLOW_TRACKING_URI)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Nike_Dual_Source_Audit_new")

# 1. Your 5 Forensic Test Queries
TEST_QUERIES = [
    "Calculate Nike's Gross Margin for 2022 ",
    "Did Nike's management accurately represent DTC shifts in the 2021 transcript?",
    "Management discusses 'digital acceleration' in the 2020 transcript. Cross-reference the digital sales growth claims with the actual 'NIKE Direct' revenue line in the 10-K.",
    "Analyze the 820 basis point drop in 2020 vs 'digital change' claims.",
    "Analyze the divergence between Nike's management’s claim of a 'digital step-function change' in the 2020 Transcript and the reported 820 basis point drop in gross margin in the 2020 10-K"
]

results_for_ragas = []
failed=False
async def run_forensic_eval():
    app = workflow.compile() 
    all_results=[]
    with get_openai_callback() as cb:  
        for idx, q in enumerate(TEST_QUERIES):
            with mlflow.start_run(run_name=f"EVAL_Query_{idx}") as test_run:
                set_active_run(test_run.info.run_id) # Set the ID so monitoring works
                
                print(f"Testing Query {idx+1}/5: {q}")
                
                start_time = time.time()
                # We run the agent end-to-end
                final_state = await app.ainvoke({"query": q, "query_history": [q]})
                
                # Calculate System Latency (Sum of all node benchmarks)
                total_latency = time.time() - start_time
                mlflow.log_metric("end_to_end_latency", total_latency)
                mlflow.log_metric("total_tokens", cb.total_tokens)
                mlflow.log_metric("prompt_tokens", cb.prompt_tokens)
                mlflow.log_metric("completion_tokens", cb.completion_tokens)
                mlflow.log_metric("total_cost_usd", cb.total_cost)
                          
                benchmarks = final_state.get("node_benchmarks", {})
                system_latency = sum(m['latency'] for m in benchmarks.values())
                
                # --- ASSERTIONS (Senior Level Requirement) ---
                print(f"System Latency: {system_latency:.2f}s")
               

                if system_latency > 60:
                    failed=True
                    print(f"FAIL: Latency {system_latency}s exceeds 60s limit!")
                else:
                    print(f"PASS: Latency within limits.")

                # Log to MLflow for the Eval Run
                mlflow.log_metric("eval_system_latency", system_latency)
                
                # RAGAS checks (Faithfulness/Relevancy) would go here next
                print("\nRunning RAGAS Evaluation on all results...")
                dataset = Dataset.from_list(all_results)
                #raw_history = final_state.values.get("query")
                used_snippets =  final_state.values.get("context_history", [])

        
                formatted_contexts = [c.get("evidence", "") for c in used_snippets]

                all_results.append({
                    "question": q,
                    "answer": final_state.get("generation", "No output generated"),
                    "contexts": formatted_contexts if formatted_contexts else ["No context found"],
                    "latency": total_latency
                })

            
                
                
        if all_results:
            # 1. Calculate P95 and P99
            latencies = [r['latency'] for r in all_results]
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)

            # 2. Log suite-level metrics
            mlflow.log_metric("suite_p95_latency", p95)
            mlflow.log_metric("suite_p99_latency", p99)
            print(f"\n--- Suite Statistics ---\nP95: {p95:.2f}s | P99: {p99:.2f}s")

            # 3. P99 Assertion (The Alert)
            if p99 > 60:
                print(f"ALERT: P99 Latency ({p99:.2f}s) exceeds 60s limit!")
                failed = True

            dataset = Dataset.from_dict(all_results)

            print("\nRunning RAGAS Evaluation...")
            # Explicitly set the models to avoid the AttributeError
            eval_llm = ChatOpenAI(model="gpt-4o-mini")
            eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            result = evaluate(
                dataset, 
                metrics=[answer_relevancy,faithfulness],
                llm=eval_llm,
                embeddings=eval_embeddings
            )

            mlflow.log_metric("avg_faithfulness", result["faithfulness"])
            mlflow.log_metric("avg_relevancy", result["answer_relevancy"])
            mlflow.log_metric("total_eval_cost", cb.total_cost)
            

            print(f"Eval Complete. Faithfulness: {result['faithfulness']:.2f}")

            # Final Forensic Assertion
            if result["faithfulness"] < 0.8:
                failed=True
                print("WARNING: Faithfulness below threshold. Potential Hallucinations detected.")
        
    if failed:
        print("\n Regression Sttaus:Failed")
        sys.exit(1)
    
    else:
        print("\nREGRESSION STATUS: SUCCESS")
        sys.exit(0)
        
if __name__ == "__main__":
    asyncio.run(run_forensic_eval())