import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks,UploadFile,HTTPException
from pydantic import BaseModel
import os,uuid,time,mlflow,shutil
from src.agent.orchestrator import workflow
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pathlib import Path
from src.data_pipeline.data1_extraction import pipeline_runner_high_throughput
from src.data_pipeline.pii_masking import batch_masking_pipeline_runner
from src.data_pipeline.chunker import process_and_upload
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from src.utils.monitoring import check_system_health, log_system_usage
from prometheus_client import make_asgi_app, Histogram, Counter, Gauge
from src.utils.monitoring import set_active_run, log_to_mlflow
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.utils import monitoring
from langchain_community.callbacks import get_openai_callback
import uvicorn

# 1. Define the Limiter (identifies users by IP address)
limiter = Limiter(key_func=get_remote_address)


load_dotenv()
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    print("DB URI not there")
host= os.getenv("CHROMA_HOST")
port=os.getenv("CHROMA_PORT")

AUDIT_LATENCY = Histogram(
    "audit_op_latency_seconds", 
    "Time spent processing audit request",
    buckets=[1, 5, 10, 30, 60, 120] # Custom buckets to track slow agents
)
ACTIVE_AUDITS = Gauge("audit_active_count", "Current concurrent audits")
AUDIT_ERRORS = Counter("audit_errors_total", "Total failed audit requests")


@asynccontextmanager
async def lifespan(app:FastAPI):
    print("--- ATTEMPTING STARTUP ---")
   
    app.state.audit_engine = None
    
    # 1. Open the connection using the context manager
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # 2. NOW you can call setup because checkpointer is the actual object
            await checkpointer.setup()
            print("--- COMPILING AUDITOR-CRITIC GRAPH ---")
            
            # 3. Compile the graph with the active checkpointer
            app.state.audit_engine = workflow.compile(
                checkpointer=checkpointer, 
                interrupt_before=["human_review"]
            )
            
            print("--- SERVER STARTUP: Postgres Checkpointer Linked ---")
            
            # 4. CRITICAL: yield MUST be inside the 'with' block 
            # so the DB connection stays open while the app is running.
            yield 
        print("--- SERVER SHUTDOWN ---")
    except Exception as e:
        print(f"--- STARTUP CRITICAL ERROR: {e} ---")
        app.state.audit_engine=None
        # Yielding here allows Swagger to load even if DB fails
        yield 

    print("--- SERVER SHUTDOWN: Cleaning up connections ---") 

app = FastAPI(title="Forensic Audit Ledger API",lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)




BASE_DIR= Path(os.getenv("DATA_DIR"))
INPUT_DIR = BASE_DIR/"raw"



class AuditInput(BaseModel):
    query: str

class ReviewInput(BaseModel):
    thread_id: str
    action: str
    report:str=None
    follow_up_query: str | None = None

async def upload_save_file(file: UploadFile, input_directory: Path, category: str = "report"):
    job_dir = input_directory / category
    job_dir.mkdir(parents=True, exist_ok=True)
    dest_path = job_dir / file.filename
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest_path

# --- BACKGROUND WORKER (No @app.post decorator here) ---
async def run_full_pipeline(file_path_str: str):
    file_path = Path(file_path_str) # Convert back to Path for logic
    try:
        processed_files = await pipeline_runner_high_throughput([file_path])
        mask_result = batch_masking_pipeline_runner(processed_files[0])
        masked_md_path = mask_result["output_path"]
        process_and_upload([masked_md_path])
        print(f"Job complete for {file_path.name}")
    except Exception as e:
        print(f"PIPELINE CRITICAL FAILURE: {str(e)}")



# --- THE ACTUAL API ENDPOINT ---
@app.post("/pipeline/upload")
async def upload_file(category: str, file: UploadFile, background_tasks: BackgroundTasks):
    if category not in ["report", "transcripts"]:
        raise HTTPException(status_code=400, detail="Invalid category. Use 'reports' or 'transcripts'")

    # Corrected the arguments passed here:
    local_path = await upload_save_file(file, INPUT_DIR, category)

    # Pass as string to avoid Swagger serialization issues
    background_tasks.add_task(run_full_pipeline, str(local_path))
    
    return {
        "message": f"File uploaded to {category}. Processing started.",
        "filename": file.filename
    }

@app.post("/audit/start")
@limiter.limit("5/minute")
async def start_audit(data: AuditInput,request:Request):
    if not check_system_health(ram_threshold=90):
        AUDIT_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Resource limit reached.")

    ACTIVE_AUDITS.inc() # Track that an audit is running
    start_time = time.time()

    engine = getattr(request.app.state,"audit_engine",None)
    if engine is None:
        raise HTTPException(status_code=503,detail="Engine not intialized")

    thread_id = f"audit_{uuid.uuid4().hex[:8]}"
    config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

    
    with mlflow.start_run(run_name=f"API_audit_{thread_id}") as run:
        with get_openai_callback() as cb:
            run_id = run.info.run_id
            
            #monitoring.GLOBAL_RUN_ID = run_id
            monitoring.set_active_run(run_id)
            #monitoring.ACTIVE_AUDIT_RUN_ID = run_id # Match your node's variable name
            try:

            # FIX 1: You MUST 'await' the invoke so it finishes the first run 
            # and hits the 'human_review' breakpoint before you call get_state.
                await engine.ainvoke(
                    {
                        "query": data.query,
                        "query_history": [data.query],
                        "human_decision": None,
                        "audit_status": "AWAITING_REVIEW"
                    },
                    config
                )

                current_state = await engine.aget_state(config)

                # 3. NOW check if the Purifier asked for clarification
                if current_state.values.get("ask_user"):
                    return {
                        "thread_id": thread_id,
                        "status": "CLARIFICATION_REQUIRED",
                        "question": current_state.values.get("clarification_question"),
                        "report": None
                    }

                return {
                "thread_id": thread_id,
                "report": current_state.values.get("generation"),
                "status": "AWAITING_REVIEW"
                }
        
            finally:
                AUDIT_LATENCY.observe(time.time() - start_time)

                # FIX 2: Use aget_state (async version) to ensure data is pulled from Postgres
                current_state =  await engine.aget_state(config)
                total_latency = time.time() - start_time
                mlflow.log_metric("end_to_end_latency", total_latency)
                mlflow.log_metric("total_tokens", cb.total_tokens)
                mlflow.log_metric("prompt_tokens", cb.prompt_tokens)
                mlflow.log_metric("completion_tokens", cb.completion_tokens)
                mlflow.log_metric("total_cost_usd", cb.total_cost)

                micro_benchmarks = current_state.values.get("node_benchmarks", {})
                system_latency = sum(m['latency'] for m in micro_benchmarks.values())
                mlflow.log_metric("system_latency",system_latency)


                if not current_state or not current_state.values:
                    raise HTTPException(status_code=500, detail="Audit started but state is empty.")
                
                ACTIVE_AUDITS.dec()

                return {
                    "thread_id": thread_id,
                    "report": current_state.values.get("generation"),
                    "status": current_state.values.get("audit_status", "AWAITING_REVIEW"),
                    "menu_options": {
                        "1": "pass",
                        "2": "Correct Manually"
                    }
                }
    

@app.post("/audit/review")
async def review_audit(data: ReviewInput,request:Request):
    engine= request.app.state.audit_engine
    config = {"configurable": {"thread_id": data.thread_id}}
    
    # Use async state retrieval
    state = await engine.aget_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Audit thread not found")

    # Prepare the update based on action
    update_data = {}
    if "human_review" in state.next:
        if data.action == "2":
            await engine.aupdate_state(config,{
            "generation": data.report,
            "human_decision": "pass", # The logic that lets the graph exit the 'Auditor' stage
            "audit_status": "RELEASED_TO_USER" 
            }, as_node="human_review")

        else:
            await engine.aupdate_state(config,{
                "human_decision": "pass",
                "audit_status": "RELEASED_TO_USER"
            }, as_node="human_review")

        await engine.ainvoke(None, config)
        final_state = await engine.aget_state(config)

        # Return the report. The API client should now show this to the END USER.
        return {
        "status": "Finalized by Auditor",
        "report": final_state.values.get("generation"),
        "next_steps": "Awaiting User Decision",
        "options": ["end", "3 (Investigate)"]
    }
        
    else:
        if data.action in ["3"] or data.action in ["Investigate"]:
            if not data.follow_up_query or data.follow_up_query.strip() == "string":
                raise HTTPException(status_code=400, detail="Follow-up query required")
            update_data = {
                "human_decision": "is_investigate",
                "is_investigate":True,
                "is_follow_up":True,
                "query": data.follow_up_query,
                "query_history": [data.follow_up_query],
                "plan": [],
                "generation": "",
                "audit_status": "Investigation",
                "audit_attempts": 0,
                "retriever_audit_attempts": 0,
                "total_cost":0,
                "output_tokens":0,
                
                # 4. RESET FEEDBACK: Stop the LLM from reading old critiques
                "reflection_feedback": None,
                "retriever_feedback": None,
                "math_plan": None,
            }

            # FIX 3: Update state AND THEN await the resume
            await engine.aupdate_state(config, update_data, as_node="human_review")
            
            # Resume the graph (passing None continues from the breakpoint)
            await engine.ainvoke(None, config)
            new_state = await engine.aget_state(config)
            return {"status": "Investigation Started", "new_report": new_state.values.get("generation"),
                "next_step": "Waiting for Auditor Review"}

        elif data.action == "end":
            # Logic to close the audit and log to JSONL
            return {"status": "Audit Closed", "report": state.values.get("generation")}

@app.get("/")
async def root():
    return {"message": "Astraeus Forensic Audit Engine is online"}

print("monitring promtheus endpoint")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    # This starts the server on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)