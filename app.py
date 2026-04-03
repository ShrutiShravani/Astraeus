from fastapi import FastAPI, BackgroundTasks,UploadFile,HTTPException
from pydantic import BaseModel
import os
import uuid
from src.agent.orchestrator import workflow
from langgraph.checkpoint.postgres import PostgresSaver
from pathlib import Path
from src.data_pipeline.data1_extraction import extract_senior_final_v3
from src.data_pipeline.pii_masking import SecureShield
from src.data_pipeline.chunker import process_and_upload
import shutil
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
from src.utils.monitoring import check_system_health, log_system_usage


load_dotenv()


# Connect to Postgres for 100-user thread management
DB_URI = os.getenv("DB_URI")
host= os.getenv("CHROMA_HOST")
port=os.getenv("CHROMA_PORT")


audit_app = None
app = FastAPI(title="Forensic Audit Ledger API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global audit_app
    print("--- ATTEMPTING STARTUP ---")
    DB_URI = os.getenv("DB_URI")
    
    # 1. Open the connection using the context manager
    try:
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # 2. NOW you can call setup because checkpointer is the actual object
            checkpointer.setup()
            
            # 3. Compile the graph with the active checkpointer
            audit_app = workflow.compile(
                checkpointer=checkpointer, 
                interrupt_before=["human_review"]
            )
            
            print("--- SERVER STARTUP: Postgres Checkpointer Linked ---")
            
            # 4. CRITICAL: yield MUST be inside the 'with' block 
            # so the DB connection stays open while the app is running.
            yield 
            
    except Exception as e:
        print(f"--- STARTUP CRITICAL ERROR: {e} ---")
        # Yielding here allows Swagger to load even if DB fails
        yield 

    print("--- SERVER SHUTDOWN: Cleaning up connections ---") 


BASE_DIR= Path(os.getenv("DATA_DIR"))
INPUT_DIR = BASE_DIR/"raw"
shield = SecureShield()
app.router.lifespan_context = lifespan
class AuditInput(BaseModel):
    query: str

class ReviewInput(BaseModel):
    thread_id: str
    action: str
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
        processed_files = await extract_senior_final_v3([file_path])
        mask_result = shield.mask_single_file(processed_files[0])
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
async def start_audit(data: AuditInput):
    if not check_system_health(ram_threshold=90):
        raise HTTPException(status_code=503, detail="Resource limit reached.")

    thread_id = f"audit_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # FIX 1: You MUST 'await' the invoke so it finishes the first run 
    # and hits the 'human_review' breakpoint before you call get_state.
    audit_app.invoke(
        {
            "query": data.query,
            "query_history": [data.query],
            "human_decision": None,
            "audit_status": "AWAITING_REVIEW"
        },
        config
    )

    # FIX 2: Use aget_state (async version) to ensure data is pulled from Postgres
    current_state =  audit_app.get_state(config)

    if not current_state or not current_state.values:
        raise HTTPException(status_code=500, detail="Audit started but state is empty.")

    return {
        "thread_id": thread_id,
        "report": current_state.values.get("generation"),
        "status": current_state.values.get("audit_status", "AWAITING_REVIEW"),
        "menu_options": {
            "1": "pass",
            "2": "reject",
            "3": "investigate"
        }
    }

@app.post("/audit/review")
async def review_audit(data: ReviewInput):
    config = {"configurable": {"thread_id": data.thread_id}}
    
    # Use async state retrieval
    state = audit_app.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Audit thread not found")

    # Prepare the update based on action
    update_data = {}
    if data.action in ["1", "pass"]:
        update_data = {"human_decision": "pass", "audit_status": "VERIFIED"}
    elif data.action in ["2", "reject"]:
        update_data = {"human_decision": "reject", "audit_status": "REJECTED"}
    elif data.action in ["3", "investigate"]:
        if not data.follow_up_query or data.follow_up_query.strip() == "string":
            raise HTTPException(status_code=400, detail="Follow-up query required")
        update_data = {
            "human_decision": "investigate",
            "query": data.follow_up_query,
            "query_history": [data.follow_up_query],
            "audit_status": "AWAITING_REVIEW"
        }

    # FIX 3: Update state AND THEN await the resume
    audit_app.update_state(config, update_data, as_node="human_review")
    
    # Resume the graph (passing None continues from the breakpoint)
    audit_app.invoke(None, config)

    new_state = audit_app.get_state(config)
    return {
        "thread_id": data.thread_id,
        "report": new_state.values.get("generation"),
        "status": new_state.values.get("audit_status")
    }