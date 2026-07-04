import mlflow,os,uuid,sys,time,json,asyncio
from dotenv import load_dotenv
load_dotenv()
from ragas import evaluate
from ragas.metrics import answer_relevancy,faithfulness
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from langchain_community.callbacks import get_openai_callback
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.utils.monitoring import check_system_health, log_system_usage
import pandas as pd
from src.agent.orchestrator import workflow
from  src.agent.nodes.auditor import audit_encoder
from prompt_toolkit import prompt
import asyncio
from mlflow.tracking import MlflowClient
from src.utils.monitoring import set_active_run
import sys
import asyncio


DB_URI= os.getenv("DB_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
client = MlflowClient()
#print("MLFLOW_TRACKING_URI =", MLFLOW_TRACKING_URI)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Nike_Dual_Source_Audit_new")

async def async_input(message):
    return await asyncio.to_thread(input,message)

async def async_prompt(message, default_val=""):
    # prompt_toolkit has its own async way, but this is safer for now
    return await asyncio.to_thread(prompt, message, default=str(default_val))

async def run_nike_audit():
    # 1. DEFINE YOUR QUERIES (Until you have a file, define them here)
    nike_queries = [
        #{"type":"C","year":2022,"q": "In the 2022 Earnings Transcripts, management claims that 'consumer demand remains at an all-time high.' Verify this claim by cross-referencing the 10-K 'Inventory' growth and the 'Gross Margin' explanation. Does the 10-K suggest this demand was organic, or was it driven by aggressive promotions and inventory liquidation?"},
        {"q":"Calculate nike gross margin for year 2019"}
        #{"q": "How did Nike's 'Direct-to-Consumer' (DTC) strategy shift in response to global store closures in 2020 according to the 10-K Risk Factors?"}
    ]

    results_for_ragas = []
   
    with mlflow.start_run():
        check_system_health()
        start_time = time.time()

        with get_openai_callback() as cb:
            async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
          
                app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
                for idx,item in enumerate(nike_queries):
                    with mlflow.start_run(run_name=f"Query_{idx+1}", nested=True) as query_run:
                        set_active_run(query_run.info.run_id)
                        # 1. Create a unique session ID for this specific audit task
                        # This is your "waybill id" replacement - unique to this query thread
                        
                        session_id = f"nike_audit_{uuid.uuid4().hex[:8]}"
                        config = {"configurable": {"thread_id": session_id}}
                        
                        print(f"\n Starting Audit Session: {session_id}")
                        
                        # 2. Initial execution
                        # The graph runs and stops at 'human_review'
                        await app.ainvoke({"query": item["q"],"query_history": [item["q"]]}, config)

                        mem_info,cpu_usage=log_system_usage()
                        mlflow.log_metric("cpu_usage",cpu_usage)
                        mlflow.log_metric("mem_info",float(mem_info/ (1024 * 1024)))
                        
                    
                        # 3. Interactive Human Loop
                        while True:
                            state = await app.aget_state(config)
                            total_latency = time.time() - start_time
                            mlflow.log_metric("end_to_end_latency", total_latency)
                            mlflow.log_metric("total_tokens", cb.total_tokens)
                            mlflow.log_metric("prompt_tokens", cb.prompt_tokens)
                            mlflow.log_metric("completion_tokens", cb.completion_tokens)
                            mlflow.log_metric("total_cost_usd", cb.total_cost)
                          

                            micro_benchmarks = state.values.get("node_benchmarks", {})
                            system_latency = sum(m['latency'] for m in micro_benchmarks.values())
                            mlflow.log_metric("system_latency",system_latency)
                            
                            
                            if state.values.get("ask_user"):
                                print("\nClarification Required")
                                print(state.values.get("clarification_question"))
                                clarification= await async_input("")

                                await app.update_state(config,{
                                    "query":clarification,
                                    "query_history":[clarification],
                                    "ask_user":False
                                },as_node="prompt_purifier")
    
                                await app.ainvoke(None,config)
                                continue
                            
                            
                            if "human_review" in state.next:
                                print("\n" + "="*50)
                                print("HUMAN REVIEW REQUIRED")
                                print("="*50)
                                print(f"\n--- AI GENERATED REPORT ---\n{state.values.get('generation')}\n")
                                user_choice = await async_input("Action: [1] Pass [2] Reject")
                            
                                if user_choice == "1":
                                    await app.aupdate_state(config, {"human_decision": "pass","report_history":[state.values.get("generation")],"audit_status": "VERIFIED: FINALIZING REPORT"}, as_node="human_review")
                                    await app.ainvoke(None, config)
                                    print("Report Finalized.")
                                    # break loop to move to RAGAS collection
                                    continue
                                    
                                elif user_choice == "2":
                                    print("\nREJECTION PROTOCOL:")
                                    print("Logic/Math/Formatting Error -> (Correct Manually)")

                                 
                                    current_report = state.values.get("generation", "No report generated yet.")
                                    print(f"\n--- CURRENT REPORT ---\n{current_report}\n----------------------")
                                    manual_text = await async_prompt("Edit Report: ", default_val=current_report)
                                    # We update the query. Because it's the SAME config, 
                                    # the Planner will see all previous snippets in the state!
                                    await app.aupdate_state(config, {
                                        "generation": manual_text, 
                                        "human_decision": "pass", # Set to pass so it goes straight to the User
                                        "audit_status": "VERIFIED (MANUAL CORRECTION)",
                                    }, as_node="human_review")
                                    
                                    print("State updated. Displaying corrected report to User...")
                                
                                    await app.ainvoke(None, config) 
                                    
                                    

                            state = await app.aget_state(config) 
                                
                                # The loop continues and will hit the human_review breakpoint again
                                # If there's no 'next' node, it means the graph finished (Scenario 1 or 2)
                            if not state.next:
                                print("\n" + "User View: FINAL AUDIT REPORT".center(50, "="))
                                print(f"{state.values.get('generation')}")

                                user_cmd= input("\n[Investigate] for New Follow up Query or [End] to Finish Audit: ").strip().lower()
                                if user_cmd == "end":
                                    with open("Final_report.jsonl","a", encoding="utf-8") as f:
                                        f.write(json.dumps(state.values.get("generation"), ensure_ascii=False,default=audit_encoder) + "\n")
                                    break

                                new_q = input("\n Enter your forensic follow-up: ")
                                    # We update the query. Because it's the SAME config, 
                                    # the Planner will see all previous snippets in the state!
                                with mlflow.start_run(run_name="Forensic_Investigation", nested=True) as forensic_run:
                                    set_active_run(forensic_run.info.run_id)
                                    await app.aupdate_state(config, {
                                        "human_decision":"is_investigate", 
                                        "is_investigate":True,
                                        "is_follow_up":True,
                                        "query": new_q,
                                        "query_history":[new_q] ,
                                        "plan": [],
                                                
                                        # 1. KILL THE OUTPUTS: This forces the graph to re-run nodes
                                        "generation": "",            # String (not Annotated, so this overwrites)
                                       
                                     # It's a refinement, not a follow-up
                                        "audit_status": "Investigation",
                                        
                                        # 3. RESET ATTEMPTS: Give the agent a fresh quota
                                        "audit_attempts": 0,
                                        "retriever_audit_attempts": 0,
                                        "total_cost":0,
                                        "output_tokens":0,
                                        
                                        # 4. RESET FEEDBACK: Stop the LLM from reading old critiques
                                        "reflection_feedback": None,
                                        "retriever_feedback": None,
                                        "math_plan": None,
                                    }, as_node="human_review")
                                    
                                    print(f"investing {new_q}...")

                                    with get_openai_callback() as follow_up_cb:
                                        await app.ainvoke(None, config) 
                                        state = await app.aget_state(config)
                                        total_latency = time.time() - start_time
                                        mlflow.log_metric("end_to_end_latency", total_latency)
                                        mlflow.log_metric("total_tokens", follow_up_cb.total_tokens)
                                        mlflow.log_metric("prompt_tokens", follow_up_cb.prompt_tokens)
                                        mlflow.log_metric("completion_tokens", follow_up_cb.completion_tokens)
                                        mlflow.log_metric("total_cost_usd", follow_up_cb.total_cost)
                                    

                                        micro_benchmarks = state.values.get("node_benchmarks", {})
                                        system_latency = sum(m['latency'] for m in micro_benchmarks.values())
                                        mlflow.log_metric("system_latency",system_latency)
                                    
                                    continue

                # 4. After the loop finishes (Pass or Reject), collect final data for RAGAS
                final_state = await app.aget_state(config)
            
                
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
            
                raw_history = final_state.values.get("query")
                # This represents the 'Total Question' the AI had to answer
                full_question_context = " | ".join(raw_history) if isinstance(raw_history, list) else str(raw_history)
            
                results_for_ragas.append({
                    "question": raw_history,
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
                            metrics=[answer_relevancy,faithfulness],
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
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_nike_audit())

