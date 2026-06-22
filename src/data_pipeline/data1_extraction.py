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

total_retries = 0
failed_pages = 0

def run_pypdfium(page_num, pdf_bytes, header_label="Table Structure"):
    """LAZY INITIALIZATION: Boots up Docling safely on-demand within the worker core execution thread."""
    global GLOBAL_CONVERTER
    
    if GLOBAL_CONVERTER is None:
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
            if hasattr(element, "text"):
                full_md.append(element.text + "\n")
    return "".join(full_md)

def validate_text_output(text,tables):
    """
    Returns:
        "valid" | "invalid" | "fallback_required"
    """
    has_text = len(text.strip())>10
    has_tables = bool(tables)
    is_empty = not has_text and not has_tables
    

    return {
        "valid": not is_empty ,
        "has_text": has_text,
        "has_tables": has_tables,
        "is_empty":is_empty
    }

def evaluate_fidelity(markdown_text,engine_choice):
    text= markdown_text.strip()

    if not text:
        return False, "empty_output"
    
    #text to symbol ratio

    total_len= len(text)
    special_chars= sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    if (special_chars/total_len)>0.8:
        return False,"high_symbol_ratio"

    #detcetign repeatign alphabets or numebrs
    unique_chars =len(set(text))

    if total_len>50 and unique_chars<5:
        return False, "low_entropy_garbage"
    
    return True, None

def send_to_dlq(page_num,reason,document_id,error=None):

    record={
        "document": document_id,
        "page": page_num,
        "reason": reason,
        "error": str(error) if error else None
    }

    filename = f"{document_id}_page_{page_num}.json"

    with open (DLQ_DIR/filename,"w") as f:
        json.dump(record,f,indent=2)
    
    logger.error({
    "document": document_id,
    "page": page_num,
    "status": "DLQ",
    "reason": reason,
    "retry_count": 0,
    "error": str(error)
})

def routing_classifier(has_table, has_text):
    if has_table:
        return {"engine": "pypdfium", "reason": "pdfplumber_table_detected"}
    if has_table and has_text:
        return {"engine": "pypdfium", "reason": "mixed_layout_detected"}
    """
    if structured_signal > 15:
        return {"engine": "pypdfium", "reason": "weak_structure"}
    """
    return {"engine": "pypdf_text", "reason": "no_table"}

def process_page_in_process(task,attempt):
    page_num, single_page_bytes, document_id = task
    start_time = time.time()
    doc = None
    
    try:
        doc = fitz.open(stream=single_page_bytes, filetype="pdf")
        f_page = doc[0]

        with pdfplumber.open(io.BytesIO(single_page_bytes)) as plumber_doc:
            p_page = plumber_doc.pages[0]
            tables = p_page.find_tables()

        text = f_page.get_text("text") or ""
        validation = validate_text_output(text, tables)

        if not validation["valid"]:
            send_to_dlq(page_num, reason=f"Pre valdiation check Failed", document_id=document_id)
            return {
                "page_num": page_num, "status": "FAILED", 
                "content": "", "engine": "fallback", "error": "invalid_page"
            }

        route = routing_classifier(validation["has_tables"], validation["has_text"])
        engine_choice = route["engine"]

        # PATHWAY: PURE TEXT
        if engine_choice == "pypdf_text":
            text_markdown = run_pypdfium(page_num, single_page_bytes, header_label="Fallback Structure")
            
            is_valid,reason=evaluate_fidelity(text_markdown,engine_choice)

            if not is_valid:
                send_to_dlq(page_num, reason=f"Fidelity Check Failed: {reason}", document_id=document_id)
                return {
                    "page_num": page_num, "status": "failed", "content": text_markdown,
                    "engine": "pypdf_text", "latency": latency

                }
            latency = time.time() - start_time
            return {
                "page_num": page_num, "status": "SUCCESS", "content": text_markdown,
                "engine": "pypdf_text", "latency": latency
            }

        # PATHWAY: MIXED/TABLE
        table_markdown = run_pypdfium(page_num, single_page_bytes, header_label="Table Extract")
        latency = time.time() - start_time
        is_valid,reason=evaluate_fidelity(text_markdown,engine_choice)

        if not is_valid:
            send_to_dlq(page_num,reason=f"Fidelity Check Failed: {reason}", document_id=document_id)
            return {
                "page_num": page_num, "status": "failed", "content": table_markdown,
                "engine": "pypdfium", "latency": latency

            }
        return {
            "page_num": page_num, "status": "SUCCESS", "retry_count": attempt,"content": table_markdown,
            "engine": engine_choice, "latency": latency
        }

    except Exception as e:
        send_to_dlq(page_num, reason="Extraction Exception", document_id=document_id, error=e)
        return {
            "page_num": page_num, "status": "FAILED", "content": "",
            "engine": None, "error": str(e), "latency": time.time() - start_time
        }
    finally:
        if doc is not None:
            doc.close()

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

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            failed_tasks = []
            for attempt in range(MAX_RETRIES):
                future_to_task = {executor.submit(process_page_in_process, task,attempt): task for task in (failed_tasks or prepared_tasks)}
                failed_tasks = []

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=45)
                        total_retries+= result.get("retry_count",0)
                        if result["status"]=="FAILED":
                            failed_pages+=1

                        page_results[result["page_num"]] = result
                    except Exception as e:
                        failed_tasks.append(task)
                        logger.warning(f"Task failed: {e}")

                if not failed_tasks:
                    break
                elif attempt == MAX_RETRIES - 1:
                    for page_num, _, doc_id in failed_tasks:
                        page_results[page_num] = {
                            "document_id": doc_id,
                            "page_num": page_num,
                            "status": "FAILED",
                            "error": "Max retries exceeded"
                        }
                        send_to_dlq(page_num, reason="Max retries exceeded", document_id=doc_id, error=None)

        total_duration = time.time() - start_total
        peak_ram_mb = main_process.memory_info().rss / 1024 / 1024

        with open(OUTPUT_DIR / f"{pdf_path.stem}.md", "w", encoding="utf-8") as outfile:
            for p_num in sorted(page_results.keys()):
                result = page_results[p_num]
                if result["status"] == "SUCCESS":
                    engine = result.get("engine")
                    if engine == "mixed": routing_stats["mixed"] += 1
                    elif engine == "pypdf_text": routing_stats["text"] += 1
                    elif engine == "pypdfium": routing_stats["table"] += 1
                    outfile.write(result["content"])
                else:
                    # This makes the error visible in the final doc
                    outfile.write(f"\n\n--- PAGE {p_num} FAILED: {result.get('error', 'Unknown Error')} ---\n\n")

        if os.getenv("MLFLOW_TRACKING_URI"):
            with mlflow.start_run(run_name=f"HighThroughput_Extract_{pdf_path.stem}"):
                mlflow.log_param("file_name", pdf_path.name)
                mlflow.log_metric("total_duration_sec", total_duration)
                mlflow.log_metric("peak_process_ram_mb", peak_ram_mb)
                mlflow.log_metric("total_failed_pages",total_retries)
                mlflow.log_metric("total_failed_pages", failed_pages)

                total_parsed = sum(routing_stats.values())

                # Add to your MLflow block:
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