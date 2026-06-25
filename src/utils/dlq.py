import os
import json
from pathlib import Path  # FIX: Swapped out broken 'path' library for native 'pathlib'
from custom_logging import logger

BASE_DIR = Path(os.getenv("DATA_DIR", "./data"))
DLQ_DIR = BASE_DIR / "dead_letter_queue"
DLQ_DIR.mkdir(parents=True, exist_ok=True)

def send_to_dlq(page_num=None, reason="Unknown Error", document_id="unknown_doc", engine=None, error=None, stage="extraction"):
    """
    Unified Production Dead Letter Queue.
    Defaults to your current extraction layout, but dynamically adjusts for masking and chunking.
    """
    # Clean up stage casing for uniform logs
    stage_upper = stage.upper()
    stage_lower = stage.lower()

    record = {
        "document": document_id,
        "stage": stage_upper,
        "page": page_num,
        "reason": reason,
        "engine": engine,
        "error": str(error) if error else None
    }
    
    # FIX: Dynamic naming so extraction, masking, and chunking logs don't overwrite each other
    if page_num is not None:
        filename = f"{document_id}_{stage_lower}_page_{page_num}.json"
    else:
        filename = f"{document_id}_{stage_lower}_failed.json"
        
    try:
        with open(DLQ_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    except Exception as write_err:
        print(f"Failed to write to DLQ filesystem: {write_err}")
    
    logger.error({
        "document": document_id,
        "stage": stage_upper,
        "page": page_num,
        "status": "DLQ",
        "engine": engine,
        "reason": reason,
        "retry_count": 0,
        "error": str(error) if error else "None"
    })
