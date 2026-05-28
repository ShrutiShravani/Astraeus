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

def validate_text_output(text):
    if not text or len(text.strip()) < 5:
        return True
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    aligned_lines = sum(1 for l in lines if len(re.findall(r"\s{3,}", l)) >= 2)
    return (aligned_lines / len(lines)) <= 0.25

def routing_classifier(has_table, blocks_count, structured_signal):
    if has_table:
        return {"engine": "pypdfium", "reason": "pdfplumber_table_detected"}
    if blocks_count < 20:
        return {"engine": "pypdf_text", "reason": "too_small"}
    if structured_signal > 15:
        return {"engine": "pypdfium", "reason": "weak_structure"}
    return {"engine": "pypdf_text", "reason": "no_table"}

def process_page_in_process(task):
    """Executes layout routing completely free of startup initialization locks."""
    page_num, single_page_bytes = task
    start_time = time.time()
    

    doc = fitz.open(stream=single_page_bytes, filetype="pdf")
    f_page = doc[0] 

    with pdfplumber.open(io.BytesIO(single_page_bytes)) as plumber_doc:
        p_page = plumber_doc.pages[0] 
        tables = p_page.find_tables()
        table_bboxes = [table.bbox for table in tables]

    text = f_page.get_text("text") or ""
    blocks = f_page.get_text("blocks")
    blocks_count = len(blocks)

 
    structured_signal = sum(1 for b in blocks if len(b[4]) > 10)
    blocks_summary = (blocks_count, structured_signal)

    has_table = len(table_bboxes) > 0
    has_text = len(text.strip()) > 0
    
    route = routing_classifier(has_table, blocks_count, structured_signal)
    engine_choice = route["engine"]

    # PATHWAY 1: MIXED PAGE PATHWAY SPLITTER
    if has_table and has_text:
        doc.close()
        table_markdown = run_pypdfium(page_num, single_page_bytes, header_label="Table Extract")
        return page_num, table_markdown, "mixed", time.time() - start_time

    # PATHWAY 2: PURE TEXT PATHWAY ONLY
    if engine_choice == "pypdf_text":
        if not validate_text_output(text):
            fallback_md = run_pypdfium(page_num, single_page_bytes, header_label="Fallback Structure")
            doc.close()
            return page_num, fallback_md, "fallback", time.time() - start_time
            
        formatted_text = f"\n\n## PAGE {page_num} (Text Layer Only)\n{'='*20}\n{text}"
        doc.close()
        return page_num, formatted_text, "text", time.time() - start_time

    # PATHWAY 3: PURE TABLE PATHWAY ONLY
    doc.close()
    pure_table_md = run_pypdfium(page_num, single_page_bytes, header_label="Table Structure")
    return page_num, pure_table_md, "table", time.time() - start_time


def pipeline_runner_high_throughput(pdf_paths: list[Path], max_workers=4):
    main_process = psutil.Process(os.getpid())
    
    for pdf_path in pdf_paths:
        print(f"\nInitializing Multi-Process Framework for: {pdf_path.name}")
        start_total = time.time()

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
        
        # Open full document master handle once to slice tasks natively
        master_doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(master_doc)
        
        print(f"--> Pre-slicing single page maps into memory layout components...")
        prepared_tasks = []
        
        for idx in range(total_pages):
            page_num = idx + 1
            page_doc = fitz.open()
            page_doc.insert_pdf(master_doc, from_page=idx, to_page=idx)
            single_page_bytes = page_doc.write()
            page_doc.close()
            
            # Pass only the isolated layout bytes to avoid data deadlocks
            prepared_tasks.append((page_num, single_page_bytes))
        
        master_doc.close()
        gc.collect()
        
        page_results = {}
        routing_stats = {"mixed": 0, "text": 0, "fallback": 0, "table": 0}
        mp_context = multiprocessing.get_context("spawn")
        
        print(f"--> Spawning {max_workers} independent CPU worker cores instantly...")
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            future_to_page = {executor.submit(process_page_in_process, task): task for task in prepared_tasks}
            
            for future in as_completed(future_to_page):
                p_num, text_content, engine_used, latency = future.result()
                page_results[p_num] = text_content
                routing_stats[engine_used] = routing_stats.get(engine_used, 0) + 1
                
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_pages:
                    print(f"    [COMPLETED] {processed_count}/{total_pages} Pages parsed. (Latest: Page {p_num} via {engine_used.upper()})")

        total_duration = time.time() - start_total
        peak_ram_mb = main_process.memory_info().rss / 1024 / 1024

        target_dir = OUTPUT_DIR / pdf_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / f"{pdf_path.stem}.md", "w", encoding="utf-8") as outfile:
            for p_num in sorted(page_results.keys()):
                outfile.write(page_results[p_num])
        
        if os.getenv("MLFLOW_TRACKING_URI"):
            with mlflow.start_run(run_name=f"HighThroughput_Extract_{pdf_path.stem}"):
                mlflow.log_param("file_name", pdf_path.name)
                mlflow.log_metric("total_duration_sec", total_duration)
                mlflow.log_metric("peak_process_ram_mb", peak_ram_mb)
        
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