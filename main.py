import mlflow
from dotenv import load_dotenv
load_dotenv()
from ragas import evaluate
from ragas.metrics import answer_relevancy,context_precision
from datasets import Dataset
import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from langchain_community.callbacks import get_openai_callback
from langgraph.checkpoint.postgres import PostgresSaver
import os
import time
from src.utils.monitoring import check_system_health, log_system_usage
import pandas as pd
import mlflow
from src.agent.orchestrator import workflow
import psutil

DB_URI= os.getenv("DB_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
#print("MLFLOW_TRACKING_URI =", MLFLOW_TRACKING_URI)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Nike_Dual_Source_Audit_new")

def run_nike_audit():
   
    # 1. DEFINE YOUR QUERIES (Until you have a file, define them here)
    nike_queries = [
        #{"type":"C","year":2022,"q": "In the 2022 Earnings Transcripts, management claims that 'consumer demand remains at an all-time high.' Verify this claim by cross-referencing the 10-K 'Inventory' growth and the 'Gross Margin' explanation. Does the 10-K suggest this demand was organic, or was it driven by aggressive promotions and inventory liquidation?"},
        {"q":"Perform an Inventory Turnover ratio calculation for Nike for FY2020.Retrieve RAW DATA: Get 'Cost of Goods Sold' (COGS) from the 2020 Income Statement. Get 'Total Inventory' from the 2020 and 2019 Balance Sheets.Formula logic: Calculate Average Inventory as $(Inventory_2020 + Inventory_2019) / 2$.Final Calculation: Divide COGS by the Average Inventory."}
        #{"type": "B", "year": 2020, "q": "How did Nike's 'Direct-to-Consumer' (DTC) strategy shift in response to global store closures in 2020 according to the 10-K Risk Factors?"}
    ]

    results_for_ragas = []
   
    with mlflow.start_run():
        check_system_health()
        start_time = time.time()

        with get_openai_callback() as cb:
            with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
                checkpointer.setup()
                app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
                for idx,item in enumerate(nike_queries):
                    with mlflow.start_run(run_name=f"Query_{idx+1}", nested=True):
                        # 1. Create a unique session ID for this specific audit task
                        # This is your "waybill id" replacement - unique to this query thread
                        
                        session_id = f"nike_audit_{uuid.uuid4().hex[:8]}"
                        config = {"configurable": {"thread_id": session_id}}
                        
                        print(f"\n Starting Audit Session: {session_id}")
                        
                        # 2. Initial execution
                        # The graph runs and stops at 'human_review'
                        app.invoke({"query": item["q"],"query_history": [item["q"]]}, config)

                        
                    
                        # 3. Interactive Human Loop
                        while True:
                            state = app.get_state(config)
                            total_latency = time.time() - start_time
                            mlflow.log_metric("end_to_end_latency", total_latency)
                            mlflow.log_metric("total_tokens", cb.total_tokens)
                            mlflow.log_metric("prompt_tokens", cb.prompt_tokens)
                            mlflow.log_metric("completion_tokens", cb.completion_tokens)
                            mlflow.log_metric("total_cost_usd", cb.total_cost)
                           
                            mem_info,cpu_usage=log_system_usage()
                        
                            mlflow.log_metric("cpu_usage",cpu_usage)
                            mlflow.log_metric("mem_info",float(mem_info))

                            micro_benchmarks = final_state.values.get("node_benchmarks", {})
                            system_latency = sum(m['latency'] for m in micro_benchmarks.values())
                            mlflow.log_metric("system_latency",system_latency)
                            
                            # If there's no 'next' node, it means the graph finished (Scenario 1 or 2)
                            if not state.next:
                                print("\n" + "User View: FINAL AUDIT REPORT".center(50, "="))
                                print(f"{state.values.get('generation')}")

                                user_cmd= input("\n[Investigate] for New Follow up Query or [End] to Finish Audit: ").strip().lower()
                                if user_cmd == "end":
                                    break

                                new_q = input("\n Enter your forensic follow-up: ")
                                    # We update the query. Because it's the SAME config, 
                                    # the Planner will see all previous snippets in the state!
                                app.update_state(config, {
                                    "is_investigate":True, 
                                    "query": new_q,
                                    "query_history":[new_q],
                                    "is_cached": False
                                })
                                
                                print(f"investing {new_q}...")
                                app.invoke({"query":new_q}, config) 

                            if "human_review" in state.next:
                                print("\n" + "="*50)
                                print("HUMAN REVIEW REQUIRED")
                                print("="*50)
                                print(f"\n--- AI GENERATED REPORT ---\n{state.values.get('generation')}\n")
                                user_choice = input("Action: [1] Pass [2] Reject [3] Manual Override required: ")
                            
                                if user_choice == "1":
                                    app.update_state(config, {"human_decision": "pass","audit_status": "VERIFIED: FINALIZING REPORT"}, as_node="human_review")
                                    app.invoke(None, config)
                                    print("Report Finalized.")
                                    # break loop to move to RAGAS collection
                                    break 
                                    
                                elif user_choice == "2":
                                    print("\nREJECTION PROTOCOL:")
                                    print("[1] Logic/Math/Formatting Error -> (Send to Generator)")
                                    print("[2] Poor Understanding/Missing Data -> (Rewrite Query & Send to Planner)")
                                    
                                    reject_mode=input("Select rejection mode")

                                    if reject_mode==1:
                                        critique = input("\nEnter specific critique for the Generator: ")
                                        app.update_state(config, {
                                            "human_decision": "reject",
                                            "reflection_feedback": {"critique": critique, "needs_revision": True,"target_node":"generator"}
                                        }, as_node="human_review")
                                        print("Re-routing to Generator...")
                                        app.invoke(None, config)
                                    elif reject_mode==2:
                                        refinement = input("\nRewrite query: ")
                                        app.update_state(config, {
                                            "human_decision": "refinement",
                                            "query": f"REFINED MISSION: {refinement}",
                                            "reflection_feedback": None
                                        }, as_node="human_review")
                                    app.invoke(None, config)
                                    print(" Audit Terminated.")
                                    break # break loop
                                    
                                elif user_choice == "3":
                                    manual_text= input("\n Enter correct report: ")
                                    # We update the query. Because it's the SAME config, 
                                    # the Planner will see all previous snippets in the state!
                                    app.update_state(config, {
                                        "generation": manual_text, 
                                        "human_decision": "pass", # Set to pass so it goes straight to the User
                                        "audit_status": "VERIFIED (MANUAL CORRECTION)"
                                    }, as_node="human_review")
                                    
                                    print("State updated. Displaying corrected report to User...")
                                    app.invoke(None, config) 
                                
                                # The loop continues and will hit the human_review breakpoint again
                      

                        # 4. After the loop finishes (Pass or Reject), collect final data for RAGAS
                        final_state = app.get_state(config)
                   
                        
                        #log microbenchmarks to mlflow
                        
                        decision = final_state.values.get("human_decision")
                        mlflow.log_param("final_human_decision", decision)
                        
                        if decision == "pass":
                            mlflow.log_metric("audit_passed", 1)
                        else:
                            mlflow.log_metric("audit_passed", 0)

                        # Only evaluate reports we actually approved
                        used_snippets =  final_state.values.get("context_history", [])
                        formatted_contexts = [c.get("evidence", "") for c in used_snippets]
                    
                        raw_history = final_state.values.get("query_history")
                        # This represents the 'Total Question' the AI had to answer
                        full_question_context = " | ".join(raw_history) if isinstance(raw_history, list) else str(raw_history)
                    
                        results_for_ragas.append({
                            "question": full_question_context,
                            "answer": final_state.values.get("generation"),
                            "contexts": formatted_contexts
                        })

                # 5. RAGAS Evaluation (Reference-Free)
                if decision == "pass":
                    if results_for_ragas:
                        dataset = Dataset.from_dict({
                            "question": [r["question"] for r in results_for_ragas],
                            "answer": [r["answer"] for r in results_for_ragas],
                            "contexts": [r["contexts"] for r in results_for_ragas],
                        })

                        print("\nRunning RAGAS Evaluation...")
                        # Explicitly set the models to avoid the AttributeError
                        eval_llm = ChatOpenAI(model="gpt-4o-mini")
                        eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

                        result = evaluate(
                            dataset, 
                            metrics=[answer_relevancy,context_precision],
                            llm=eval_llm,
                            embeddings=eval_embeddings
                        )
                        
                        scores = result.scores   # usually list[dict]
                        df = pd.DataFrame(scores)

                        print(df.head())
                        print(df.dtypes)

                        # 3) Log only numeric columns as scalar means
                        for col in df.columns:
                            numeric_col = pd.to_numeric(df[col], errors="coerce").dropna()
                            if not numeric_col.empty:
                                mean_value = float(numeric_col.mean())
                                mlflow.log_metric(col, mean_value)
                                print(f"Logged {col} = {mean_value}")
                                        

if __name__ == "__main__":
    run_nike_audit()

