import os
import re
import gc
import time
import psutil
import logging
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Assuming these are imported correctly from your codebase structure
# from presidio_analyzer import AnalyzerEngine
# from presidio_anonymizer import AnonymizerEngine
# from presidio_anonymizer.entities import OperatorConfig
from src.utils.dlq import send_to_dlq  

load_dotenv()

BASE_DIR = Path(os.getenv("DATA_DIR", "./data"))
MASKED_DIR = BASE_DIR / "masked"
MAX_ATTEMPTS = 3
EXTRACTED_DIR = BASE_DIR/"extracted_md"
MASKED_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("SecureShieldPipeline")

GLOBAL_ANALYZER = None
GLOBAL_ANONYMIZER = None

def process_single_markdown_worker(task, attempt):
    """
    WORKER EXECUTION LAYER: Runs independently inside child CPU processor cores.
    Processes, masks, and self-validates an isolated single-page text segment.
    """
    global GLOBAL_ANALYZER, GLOBAL_ANONYMIZER
    
    # Unpack the exact page segment text block directly from the task payload matrix
    document_id, page_num, raw_page_text, md_path_str = task
    md_path = Path(md_path_str)
    start_time = time.time()
    
    result = {
        "document_id": document_id,
        "page_num": page_num,
        "status": "FAILED",
        "content": "",
        "entities_masked": 0,
        "latency": 0.0,
        "retry_count": attempt,
        "error": None
    }
    
    try:
        if GLOBAL_ANALYZER is None or GLOBAL_ANONYMIZER is None:
            GLOBAL_ANALYZER = AnalyzerEngine()
            GLOBAL_ANONYMIZER = AnonymizerEngine()
            
        patterns = {
            "SEC_ID": r"\d{10}-\d{2}-\d{6}",
            "ZIP_CODE": r"\b\d{5}(?:-\d{4})?\b"
        }
        
        # FIX 1: Swapped raw file read with our isolated single-page 'raw_page_text' payload
        working_content = raw_page_text
        
        # 1. Stage 1 Protection: NLP Named Entity Recognition
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[ENTITY_REDACTED]"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"chars_to_mask": 12, "masking_char": "*", "from_end": True})
        }
        
        analyzer_results = GLOBAL_ANALYZER.analyze(
            text=working_content,
            language='en',
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]
        )
        ner_detected = len(analyzer_results)
        
        anonymized_res = GLOBAL_ANONYMIZER.anonymize(
            text=working_content, 
            analyzer_results=analyzer_results, 
            operators=operators
        )
        safe_content = anonymized_res.text 
        
        # 2. Stage 2 Protection: Rule-Based Match Subscriptions
        reg_entity_detected = 0
        for label, pattern in patterns.items():
            safe_content, match_count = re.subn(pattern, f"[{label}_REDACTED]", safe_content)
            reg_entity_detected += match_count
            
        total_page_entities = ner_detected + reg_entity_detected
        
        # 3. SELF-VALIDATION LAYER: Run post-masking scan to catch leaks
        post_validation_nlp = GLOBAL_ANALYZER.analyze(
            text=safe_content,
            language='en',
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]
        )

        leaked_nlp_entities = [ent for ent in post_validation_nlp if "REDACTED" not in safe_content[ent.start:ent.end]]
        leaked_regex_labels = []

        for label, pattern in patterns.items():
            matches = re.findall(pattern, safe_content)
            if matches:
                leaked_regex_labels.append(f"{label} ({len(matches)} missed)")
        
        if len(leaked_nlp_entities) > 0 or len(leaked_regex_labels) > 0:
            error_details = []
            if leaked_nlp_entities:
                error_details.append(f"NLP Leaks detected: {len(leaked_nlp_entities)} items")
            if leaked_regex_labels:
                error_details.append(f"Regex Leaks detected: {', '.join(leaked_regex_labels)}")
                
            raise ValueError(f"PII Leakage Validation Failure: {'. '.join(error_details)}")

        # Operation succeeded for this isolated page block string context
        result["status"] = "SUCCESS"
        result["content"] = safe_content
        result["entities_masked"] = total_page_entities
        result["latency"] = time.time() - start_time
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def batch_masking_pipeline_runner(md_paths: list[Path], max_workers=4):
    """
    ORCHESTRATION LAYER: Splits Markdown files into page tasks, coordinates execution pools,
    aggregates results, writes individual page failures to the DLQ, and builds clean outputs.
    """
    main_process = psutil.Process(os.getpid())
    mp_context = multiprocessing.get_context("spawn")
    
    for md_path in md_paths:
        if not md_path.exists():
            continue
            
        print(f"\nInitializing Page-Level Masking Framework for: {md_path.name}")
        start_total = time.time()
        document_id = md_path.stem
        
        full_content = md_path.read_text(encoding='utf-8')
        
        # Precise regex lookahead mapping to slice content by page headers
        page_splits = re.split(r'(?=\n\n## PAGE \d+ \(Extraction\))', full_content)
        
        if len(page_splits) <= 1:
            page_chunks = [(1, full_content)]
        else:
            page_chunks = []
            for block in page_splits:
                if not block.strip():
                    continue
                match = re.search(r'## PAGE (\d+)', block)
                p_num = int(match.group(1)) if match else len(page_chunks) + 1
                page_chunks.append((p_num, block))
                
        prepared_tasks = [(document_id, p_num, text_block, str(md_path)) for p_num, text_block in page_chunks]
        
        page_results = {}
        current_tasks_to_run = list(prepared_tasks)
        
        doc_processed = 0
        failures = 0
        total_retries = 0
        entities_detected = 0

        print(f"--> Slicing document into {len(prepared_tasks)} discrete page tasks...")

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            for attempt in range(MAX_ATTEMPTS):
                if not current_tasks_to_run:
                    break
                    
                # FIX 2: Explicitly matching the exact worker function name
                future_to_task = {
                    executor.submit(process_single_markdown_worker, task, attempt): task 
                    for task in current_tasks_to_run
                }
                
                next_failed_tasks = []
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    doc_id, p_num, _, _ = task
                    
                    try:
                        result = future.result(timeout=60)
                        total_retries += result.get("retry_count", 0)
                        
                        if result["status"] == "SUCCESS":
                            doc_processed += 1
                            entities_detected += result["entities_masked"]
                            page_results[p_num] = result
                            
                            logger.info({
                                "document": doc_id,
                                "page": p_num,
                                "stage": "MASKING",
                                "status": "PROCESSING_FINISHED",
                                "entities_in_page": result["entities_masked"],
                                "latency": f"{result['latency']:.2f}s"
                            })
                        else:
                            failures += 1
                            page_results[p_num] = result
                            
                            send_to_dlq(
                                page_num=p_num,
                                document_id=doc_id, 
                                reason="Engine Masking Validation Rejected", 
                                error=result.get("error"), 
                                stage="masking"
                            )
                            
                    except Exception as e:
                        if attempt < MAX_ATTEMPTS - 1:
                            next_failed_tasks.append(task)
                            total_retries += 1
                            logger.warning(f"Transient masking crash (Retry {attempt+1}): {e} for {doc_id} Page {p_num}")
                        else:
                            failures += 1
                            page_results[p_num] = {"status": "FAILED", "error": "Max process retries exceeded"}
                            
                            send_to_dlq(
                                page_num=p_num,
                                document_id=doc_id, 
                                reason="Max Retries Exhausted on Process Pool Crash", 
                                error=e, 
                                stage="masking"
                            )
                
                current_tasks_to_run = next_failed_tasks

        # Assemble individual page outputs back into full matching masked structures
        subfolder = md_path.parent.name
        target_dir = MASKED_DIR / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = md_path.name.replace(".md", "_masked.md")
        output_path = target_dir / output_filename
        
        with open(output_path, "w", encoding="utf-8") as outfile:
            for p_num in sorted(page_results.keys()):
                res = page_results[p_num]
                if res["status"] == "SUCCESS":
                    outfile.write(res["content"])
                else:
                    outfile.write(f"\n\n--- MASKING FAILED ON PAGE {p_num}: {res.get('error', 'Unknown Error')} ---\n\n")

        total_duration = time.time() - start_total
        print(f"--> Assembled: {output_path.name}. Masked: {entities_detected} entities. Broken pages: {failures}")


if __name__ == "__main__":
    
    files = list(EXTRACTED_DIR.rglob("*.md"))
    if files:
        batch_masking_pipeline_runner(files,max_workers=4)
        
    else:
        print("No pdf files found in raw directory")
    