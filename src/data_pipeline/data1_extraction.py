import os
import io
import gc
import statistics
import time
import psutil
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from pypdf import PdfReader, PdfWriter
import mlflow
from mlflow.tracking import MlflowClient
import re
import fitz
from threading import Lock
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
from custom_logging import logger
from concurrent.futures.process import BrokenProcessPool
from src.utils.dlq import send_to_dlq

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, PdfBackend
from dotenv import load_dotenv
from docling.datamodel.base_models import DocumentStream
import pdfplumber

load_dotenv()
DB_URI = os.getenv("DB_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
client = MlflowClient()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Nike_Dual_Source_Audit_data_extraction")
stats_lock = Lock()

BASE_DIR = Path(os.getenv("DATA_DIR", "./data"))
INPUT_DIR = BASE_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "extracted_md"
DLQ_DIR = BASE_DIR / "dead_letter_queue"
TEMP_DIR = BASE_DIR / "temp_chunks"

for path in [INPUT_DIR, OUTPUT_DIR, DLQ_DIR, TEMP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Shared volatile worker process engine state tracker
GLOBAL_CONVERTER = None
MAX_RETRIES=2


def run_pypdfium(page_num, pdf_bytes, header_label="Table Structure"):
    """LAZY INITIALIZATION: Boots up Docling safely on-demand within the worker core execution thread."""
    global GLOBAL_CONVERTER
    
    if GLOBAL_CONVERTER is None:
        # Initializing Docling properties securely inside the split process context
        pdf_options = PdfPipelineOptions()  
        pdf_options.do_table_structure = True  
        pdf_options.table_structure_options.mode = TableFormerMode.FAST
        pdf_options.do_ocr = False
        pdf_options.generate_page_images = False
        pdf_options.generate_picture_images = False

        GLOBAL_CONVERTER = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options, pdf_backend="pypdfium2")
            }
        )

    stream = DocumentStream(name=f"page_{page_num}.pdf", stream=io.BytesIO(pdf_bytes))
    result = GLOBAL_CONVERTER.convert(stream)
    parsed_doc = result.document

    full_md = [f"\n\n## PAGE {page_num} ({header_label})\n{'='*20}\n"]
    for element, _ in parsed_doc.iterate_items():
        try:
            full_md.append(element.export_to_markdown(doc=parsed_doc) + "\n")
        except Exception:
            if hasattr(element, "text") and element.text:
                full_md.append(element.text + "\n")
    return "".join(full_md)

def validate_text_output(text, tables):
    """Returns structurally valid parsing states."""
    has_text = len(text.strip()) > 10
    has_tables = bool(tables)
    is_empty = not has_text and not has_tables
    
    return {
        "valid": not is_empty,
        "has_text": has_text,
        "has_tables": has_tables,
        "is_empty": is_empty
    }


def convert_markdown_to_plain(md: str) -> str:
    """
    Converts markdown into plain text for comparison with raw PDF text.
    Keeps table contents while removing markdown syntax.
    """

    # Remove markdown headings
    md = re.sub(r'^#{1,6}\s+', '', md, flags=re.MULTILINE)

    # Convert table separators into spaces
    md = md.replace("|", " ")

    # Remove markdown divider rows
    md = re.sub(r'^\s*:?-{3,}:?\s*$', '', md, flags=re.MULTILINE)

    # Remove emphasis
    md = md.replace("*", "")
    md = md.replace("_", "")

    # Collapse whitespace
    md = re.sub(r'\s+', ' ', md)

    return md.strip()


def evaluate_fidelity(
    markdown_text: str,
    raw_text: str,
    expected_tables: bool
):
    """
    Runtime extraction validation.

    Returns:
        (True, None)

        or

        (False, reason)
    """

    # ---------------------------------------------------
    # Empty output
    # ---------------------------------------------------

    if not markdown_text.strip():
        return False, "empty_output"

    # ---------------------------------------------------
    # Convert markdown into plain text
    # ---------------------------------------------------

    plain_text = convert_markdown_to_plain(markdown_text)

    extracted_words = len(plain_text.split())

    if extracted_words < 10:
        return False, "too_few_words"

    # ---------------------------------------------------
    # Retention ratio
    # ---------------------------------------------------

    raw_words = max(len(raw_text.split()), 1)

    retention = extracted_words / raw_words

    if retention < 0.90:
        return False, f"low_retention_{retention:.2f}"

    # ---------------------------------------------------
    # Symbol ratio
    # ---------------------------------------------------

    special = sum(
        1
        for c in markdown_text
        if not c.isalnum() and not c.isspace()
    )

    ratio = special / max(len(markdown_text), 1)

    if ratio > 0.60:
        return False, "mostly_symbols"

    # ---------------------------------------------------
    # Table sanity
    # ---------------------------------------------------

    if expected_tables:

        if "|" not in markdown_text:
            return False, "table_missing"

    return True, None

def routing_classifier(has_table, has_text):
    # BUG FIX: Re-ordered conditional logic cleanly so complex states prioritize correctly
    if has_table and has_text:
        return {"engine": "pypdfium", "reason": "mixed_layout_detected"}
    if has_table:
        return {"engine": "pypdfium", "reason": "pdfplumber_table_detected"}
    return {"engine": "pypdf_text", "reason": "no_table"}

def process_page_in_process(task, attempt):
    page_num, single_page_bytes, document_id = task
    start_time = time.time()
    
    result = {
        "page_num": page_num,
        "status": "FAILED",
        "content": "",
        "engine": "unknown",
        "latency": 0.0,
        "retry_count": attempt,
        "error": None
    }
    
    doc = None
    try:
        doc = fitz.open(stream=single_page_bytes, filetype="pdf")
        f_page = doc[0]

        with pdfplumber.open(io.BytesIO(single_page_bytes)) as plumber_doc:
            p_page = plumber_doc.pages[0]
            tables = p_page.find_tables() or []

        text = f_page.get_text("text") or ""
        validation = validate_text_output(text, tables)

        if not validation["valid"]:
            send_to_dlq(page_num, reason="Pre-validation check Failed", document_id=document_id)
            result["error"] = "invalid_page"
            return result

        route = routing_classifier(validation["has_tables"], validation["has_text"])
        result["engine"] = route["engine"]
        
        # Content generation stage
        content = run_pypdfium(page_num, single_page_bytes, header_label="Extraction")
        result["content"] = content
        
        # Fidelity evaluations
        is_valid, reason = evaluate_fidelity(
                        markdown_text=content,
                        raw_text=text,
                        expected_tables=validation["has_tables"]
                    )
                            
        if not is_valid:
            send_to_dlq(document_id=document_id,stage="data_extraction",page_num=page_num, reason="Extraction Exception", engine=result["engine"],error=e)
            result["error"] = f"fidelity_failure: {reason}"
            return result
    
        # Success checkpoint
        result["status"] = "SUCCESS"
        result["latency"] = time.time() - start_time
        return result

    except Exception as e:
        result["error"] = str(e)
        send_to_dlq(document_id=document_id,stage="data_extraction",page_num=page_num, reason="Extraction Exception", engine=result["engine"],error="Extraction_Failure")
        return result
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def pipeline_runner_high_throughput(pdf_paths: list[Path], max_workers=4):
    main_process = psutil.Process(os.getpid())

    for pdf_path in pdf_paths:
        print(f"\nInitializing Multi-Process Framework for: {pdf_path.name}")
        start_total = time.time()

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        master_doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(master_doc)
        logger.info({
            "document": pdf_path.stem,
            "status": "PROCESSING_STARTED",
            "total_pages": total_pages
        })

        print(f"--> Pre-slicing single page maps into memory layout components...")
        prepared_tasks = []
        document_id = pdf_path.stem

        for idx in range(total_pages):
            page_num = idx + 1
            page_doc = fitz.open()
            page_doc.insert_pdf(master_doc, from_page=idx, to_page=idx)
            single_page_bytes = page_doc.write()
            page_doc.close()
            
            logger.info({
                "document": document_id,
                "page": page_num,
                "status": "PAGE_STARTED"
            })
            prepared_tasks.append((page_num, single_page_bytes, document_id))

        master_doc.close()
        gc.collect()

        page_results = {}
        routing_stats = {"mixed": 0, "text": 0, "fallback": 0, "table": 0}
        mp_context = multiprocessing.get_context("spawn")

        print(f"--> Spawning {max_workers} independent CPU worker cores instantly...")
        total_retries = 0
        failed_pages = 0
        
        # FIX 1: Explicitly track the active queue dynamically across retry tiers
        current_tasks_to_run = list(prepared_tasks)

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            for attempt in range(MAX_RETRIES):
                if not current_tasks_to_run:
                    break  # Everything succeeded early!
                
                # Submit only what currently needs to be evaluated/retried
                future_to_task = {
                    executor.submit(process_page_in_process, task, attempt): task 
                    for task in current_tasks_to_run
                }
                
                # Dynamic container for holding fresh transient failures inside this tier
                next_failed_tasks = []

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=45)
                        total_retries += result.get("retry_count", 0)
                        
                        # 1. SUCCESS: Record perfectly.
                        if result["status"] == "SUCCESS":
                            page_results[result["page_num"]] = result
                            
                        # 2. PERMANENT FAILURE: Engine rejected page structure. Do not retry.
                        else:
                            failed_pages += 1
                            page_results[result["page_num"]] = result

                    except Exception as e:
                        # 3. TRANSIENT FAILURE: Worker process crash / timeout / system error.
                        failed_task = future_to_task[future]
                        doc_id = failed_task[2]
                        p_num = failed_task[0]
                       
                        if attempt < MAX_RETRIES - 1:
                            next_failed_tasks.append(task)
                            total_retries += 1
                            logger.warning(f"Transient failure (Retry {attempt+1}): {e} for {doc_id} Page {p_num}")
                        else:
                            # Max retries completely exhausted across all execution limits
                            failed_pages += 1
                            
                            # FIX 2: Swapped loose leaky variables for exact task targets (doc_id, p_num)
                            logger.error(f"Transient failure exhausted after {MAX_RETRIES} attempts: {e} for {doc_id} Page {p_num}")
                           
                            page_results[p_num] = {
                                "document_id": doc_id,
                                "page_num": p_num,
                                "status": "FAILED",
                                "error": f"Max retries exceeded: {str(e)}"
                            }

                            send_to_dlq(page_num=p_num, reason="Max retries exceeded", document_id=doc_id, engine="Crashed_Worker", error="System_failure")

                # Swap the active execution state cleanly to prevent infinite lists
                current_tasks_to_run = next_failed_tasks

        total_duration = time.time() - start_total
        peak_ram_mb = main_process.memory_info().rss / 1024 / 1024

        # Compile and generate standard markdown documentation layout outputs
        with open(OUTPUT_DIR / f"{pdf_path.stem}.md", "w", encoding="utf-8") as outfile:
            for p_num in sorted(page_results.keys()):
                result = page_results[p_num]
                if result["status"] == "SUCCESS":
                    logger.info(f"Page {p_num} processed with {result.get('engine')} in {result.get('latency', 0):.2f}s")
                    
                    engine = result.get("engine")
                    # Dynamically check engines against structural metrics safely
                    if engine == "pypdfium" and "mixed" in result.get("reason", ""): 
                        routing_stats["mixed"] += 1
                    elif engine == "pypdf_text": 
                        routing_stats["text"] += 1
                    elif engine == "pypdfium": 
                        routing_stats["table"] += 1
                    
                    outfile.write(result["content"])
                else:
                    outfile.write(f"\n\n--- PAGE {p_num} FAILED: {result.get('error', 'Unknown Error')} ---\n\n")

        if os.getenv("MLFLOW_TRACKING_URI"):
            with mlflow.start_run(run_name=f"HighThroughput_Extract_{pdf_path.stem}"):
                mlflow.log_param("file_name", pdf_path.name)
                mlflow.log_metric("total_duration_sec", total_duration)
                mlflow.log_metric("peak_process_ram_mb", peak_ram_mb)
                
                # FIX 3: Reassigned unique log keys to prevent data overwrites or tracker crashes
                mlflow.log_metric("total_retry_attempts", total_retries)
                mlflow.log_metric("final_failed_pages_count", failed_pages)

                total_parsed = sum(routing_stats.values())

                if total_parsed > 0:
                    mlflow.log_metric("ratio_mixed", routing_stats['mixed'] / total_parsed)
                    mlflow.log_metric("ratio_text", routing_stats['text'] / total_parsed)
                    mlflow.log_metric("ratio_table", routing_stats['table'] / total_parsed)
                    mlflow.log_metric("ratio_fallback", routing_stats['fallback'] / total_parsed)
                  
        print(f"\n--> SUCCESS: {pdf_path.name} converted in {total_duration:.2f} seconds. Peak RAM: {peak_ram_mb:.2f} MB")
        print("="*40)
        print("ROUTING OBSERVABILITY MATRIX SUMMARY:")
        print(f"  Mixed Layout Pages Parsed   : {routing_stats['mixed']}")
        print(f"  Pure Text Pages Parsed      : {routing_stats['text']}")
        print(f"  Pure Table Pages Parsed     : {routing_stats['table']}")
        print(f"  Fallback Pages Parsed       : {routing_stats['fallback']}")
        print("="*40 + "\n")

    
        

if __name__ == "__main__":
    
    files = list(INPUT_DIR.rglob("*.pdf"))
    if files:
        pipeline_runner_high_throughput(files,max_workers=4)
        
    else:
        print("No pdf files found in raw directory")
    
    """

    files = list(INPUT_DIR.rglob("*.pdf"))
    
    debug_file= files[0]

    if not debug_file.exists():
        print(f"Debug target file missing: {debug_file}")
    else:
        print(f"!!! STARTING ISOLATED 4-PAGE LAYOUT DEBUG FOR: {debug_file.name} !!!")


        debug_page_slices= [(str(debug_file),idx)for idx in range(15)]

        page_results={}
        for task_args in debug_page_slices:
            p_num, markdown_text, engine_used, metrics = process_hybrid_page(task_args)
            page_results[p_num] = markdown_text
            print(f"-> Page {p_num} successfully parsed with engine: [{engine_used}]")

        debug_output_path = OUTPUT_DIR / f"DEBUG_LAYOUT_{debug_file.stem}.md"
        with open(debug_output_path,"w",encoding="utf-8")as outfile:
            for p_num in sorted(page_results.keys()):
                outfile.write(page_results[p_num])
                outfile.write(f"\n\n\n")
    """