import os
import io
import gc
from pydoc import text
import time
from unicodedata import numeric
import psutil
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from pypdf import PdfReader, PdfWriter
import mlflow
from mlflow.tracking import MlflowClient
import time
import re
from collections import defaultdict, Counter
import fitz

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import re
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode,PdfBackend
from dotenv import load_dotenv
from dotenv import load_dotenv
from docling.datamodel.base_models import DocumentStream
import re

load_dotenv()
DB_URI= os.getenv("DB_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
client = MlflowClient()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Nike_Dual_Source_Audit_data_extraction")


BASE_DIR = Path(os.getenv("DATA_DIR"))

INPUT_DIR = BASE_DIR / "raw"

OUTPUT_DIR = BASE_DIR / "extracted_md"

DLQ_DIR = BASE_DIR / "dead_letter_queue"

TEMP_DIR = BASE_DIR / "temp_chunks"


for path in [INPUT_DIR, OUTPUT_DIR, DLQ_DIR, TEMP_DIR]:

    path.mkdir(parents=True, exist_ok=True)


pdf_options = PdfPipelineOptions()  # 1. Hyper-fast C++ backend parser
pdf_options.do_table_structure = True  
pdf_options.table_structure_options.mode = TableFormerMode.FAST
pdf_options.do_ocr = False
pdf_options.generate_page_images = False
pdf_options.generate_picture_images = False

# 2. Single Global Instance shared safely by all executing threads
GLOBAL_CONVERTER = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options,pdf_backend="pypdfium2")
    }
)


def extract_blocks(doc, page_num):

    page = doc[page_num-1]

    raw_blocks = page.get_text("words")

    blocks = []

    # LIMIT BLOCKS FOR LOW LATENCY
    for b in raw_blocks[:300]:

        x0, y0, x1, y1, text, *_ = b

        text = text.strip()

        if not text:
            continue

        blocks.append({
            "text": text,
            "x0": x0,
            "y0": y0
        })

   
    print(blocks[:10])

    return blocks

    

def routing_classifier(blocks):
    # Use a generator to be memory efficient and fast
    if len(blocks)<6:
        return "pypdf_text"
    
    rows= defaultdict(list)

    for b in blocks[:50]:
        y_bucket=round(b["y0"]/5)*5

        rows[y_bucket].append(b)
    
    structured_rows=0
    numeric_rows=0
    patterns=[]

    for row in rows.values():

        if len(row) < 3:
            continue
        row=sorted(row,key=lambda r:r["x0"])

        x_pattern= tuple(
            round(r["x0"],-1)
            for r in row
        )

        patterns.append(x_pattern)

        if len(set(x_pattern))>=3:
            structured_rows+=1
        
        row_text = " ".join(r["text"] for r in row)

        digit_ratio= (
            sum(c.isdigit() for c in row_text)/max(len(row_text),1)
        )

        if digit_ratio>0.1:
            numeric_rows+=1

    pattern_repeat = 0

    if patterns:
        pattern_repeat= max(
            Counter(patterns).values()
        )

    score = (
        structured_rows * 2 +
        numeric_rows * 2 +
        pattern_repeat * 3
    )

    print(
    f"structured={structured_rows}, "
    f"numeric={numeric_rows}, "
    f"repeat={pattern_repeat}, "
    f"score={score}"
)

    if score >= 10:
        return "pypdfium"

    return "pypdf_text"
        
    
def process_page_in_thread(doc,page_num,text,pdf_bytes):
    """Zero serialization overhead threaded processor."""
    
    start_time = time.time()
    

    blocks = extract_blocks(doc,page_num)
    engine_choice = routing_classifier(blocks)

    if engine_choice == "pypdf_text":
       return page_num, text, "pypdf_text", time.time()-start_time
    
    else:
        # Stream bytes directly out of shared process memory space
        stream = DocumentStream(name=f"page_{page_num}.pdf", stream=io.BytesIO(pdf_bytes))
        result = GLOBAL_CONVERTER.convert(stream)
        doc = result.document

        full_md = [f"\n\n## PAGE {page_num} (Table Structure)\n{'='*20}\n"]
        for element, _ in doc.iterate_items():
            try: 
                full_md.append(element.export_to_markdown(doc=doc) + "\n")
            except Exception:
                if hasattr(element, "text"):
                    full_md.append(element.text + "\n")
        
        latency = time.time() - start_time
        print(f"[Thread Optimized] Page {page_num} compiled in {latency:.2f}s")
        return page_num, "".join(full_md), "pypdfium", latency

def pipeline_runner_high_throughput(pdf_paths: list[Path], max_threads=4):
    """High-throughput execution pool engineered for your 32GB laptop."""
    main_process = psutil.Process(os.getpid())
    
    for pdf_path in pdf_paths:
        print(f"\nInitializing High-Throughput Thread-Framework for: {pdf_path.name}")
        start_total = time.time()


        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        doc = fitz.open(pdf_path)
        
        # Pre-allocate layout extractions and in-memory streams natively on the main thread
        print(f"--> Isolating and preparing layout streams for {total_pages} pages...")
        prepared_tasks = []
        for idx in range(total_pages):
            page_num = idx + 1
            page = reader.pages[idx]
            text = page.extract_text() or ""
            
            writer = PdfWriter()
            writer.add_page(page)
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()
            buf.close()
            
            prepared_tasks.append((page_num,text,pdf_bytes))
        
        del reader
        gc.collect()

        page_results = {}
        latencies = []
        engine_metrics = {"pypdf_text": 0, "pypdfium": 0}
        
        # Run execution pipeline utilizing concurrent worker threads
        print(f"--> Spawning execution threads (Capacity: {max_threads} concurrent workers)...")
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_page = {
                executor.submit(process_page_in_thread,doc, p_num,text,p_bytes): p_num 
                for p_num, text, p_bytes in prepared_tasks
            }
            
            for future in as_completed(future_to_page):
                p_num, text_content, engine_used, latency = future.result()
                page_results[p_num] = text_content
                engine_metrics[engine_used] += 1
                latencies.append(latency)
            doc.close()

        total_duration = time.time() - start_total
        peak_ram_mb = main_process.memory_info().rss / 1024 / 1024

        # Save files to disk out of compute loop scope
        target_dir = OUTPUT_DIR / pdf_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / f"{pdf_path.stem}.md", "w", encoding="utf-8") as outfile:
            for p_num in sorted(page_results.keys()):
                outfile.write(page_results[p_num])
        
        # Log optimized metrics to MLflow
        with mlflow.start_run(run_name=f"HighThroughput_Extract_{pdf_path.stem}"):
            mlflow.log_param("file_name", pdf_path.name)
            mlflow.log_param("max_threads", max_threads)
            mlflow.log_metric("total_duration_sec", total_duration)
            mlflow.log_metric("peak_process_ram_mb", peak_ram_mb)
            if latencies:
                mlflow.log_metric("avg_page_latency_sec", sum(latencies) / len(latencies))

        print("\n" + "="*40)
        print(f"HIGH-THROUGHPUT SUMMARY FOR {pdf_path.name}:")
        print(f"Total Execution Time: {total_duration:.2f} seconds")
        print(f"Peak Process RAM Footprint: {peak_ram_mb:.2f} MB")
        print(f"Pages routed to PyPDF: {engine_metrics['pypdf_text']} | Docling: {engine_metrics['pypdfium']}")
        print("="*40 + "\n")

    


if __name__ == "__main__":
    
    files = list(INPUT_DIR.rglob("*.pdf"))
    if files:
        pipeline_runner_high_throughput(files,max_threads=4)
    else:
        print("No pdf files found in raw directory")
    
    """
    debug
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