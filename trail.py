def process_single_markdown_worker(task, attempt):
    """
    WORKER EXECUTION LAYER: Runs independently inside child CPU processor cores.
    Processes a single file, masks it, and validates that NO entities remain.
    """
    global GLOBAL_ANALYZER, GLOBAL_ANONYMIZER
    
    md_path_str, document_id = task
    md_path = Path(md_path_str)
    start_time = time.time()
    
    result = {
        "document_id": document_id,
        "status": "FAILED",
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
        
        raw_content = md_path.read_text(encoding='utf-8')
        
        # 1. Stage 1 Protection: NLP Named Entity Recognition
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[ENTITY_REDACTED]"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"chars_to_mask": 12, "masking_char": "*", "from_end": True})
        }
        
        analyzer_results = GLOBAL_ANALYZER.analyze(
            text=raw_content,
            language='en',
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]
        )
        ner_detected = len(analyzer_results)
        
        anonymized_res = GLOBAL_ANONYMIZER.anonymize(
            text=raw_content, 
            analyzer_results=analyzer_results, 
            operators=operators
        )
        safe_content = anonymized_res.text 
        
        # 2. Stage 2 Protection: Rule-Based Match Subscriptions
        reg_entity_detected = 0
        for label, pattern in patterns.items():
            safe_content, match_count = re.subn(pattern, f"[{label}_REDACTED]", safe_content)
            reg_entity_detected += match_count
            
        total_file_entities = ner_detected + reg_entity_detected

        # =========================================================================
        # 3. 🚨 SELF-VALIDATION LAYER: CATCH MISSES & ENSURE OUTPUT IS SAFE
        # =========================================================================
        
        # Validation Pass A: Re-analyze with Presidio to catch NLP leaks
        post_validation_nlp = GLOBAL_ANALYZER.analyze(
            text=safe_content,
            language='en',
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]
        )
        
        # Filter out false positives caused by Presidio flag-matching your own "[ENTITY_REDACTED]" string labels
        leaked_nlp_entities = [ent for ent in post_validation_nlp if "REDACTED" not in safe_content[ent.start:ent.end]]

        # Validation Pass B: Scan text with Regex to catch structural pattern leaks
        leaked_regex_labels = []
        for label, pattern in patterns.items():
            matches = re.findall(pattern, safe_content)
            if matches:
                leaked_regex_labels.append(f"{label} ({len(matches)} missed)")

        # Evaluate Validation Results
        if len(leaked_nlp_entities) > 0 or len(leaked_regex_labels) > 0:
            error_details = []
            if leaked_nlp_entities:
                error_details.append(f"NLP Leaks detected: {len(leaked_nlp_entities)} items")
            if leaked_regex_labels:
                error_details.append(f"Regex Leaks detected: {', '.join(leaked_regex_labels)}")
                
            # Force validation failure exception to trigger retry logic or send straight to DLQ
            raise ValueError(f"PII Leakage Validation Failure: {'. '.join(error_details)}")

        # =========================================================================
        
        # 4. Save transformed content structural outputs if validation passes
        subfolder = md_path.parent.name
        target_dir = MASKED_DIR / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = md_path.name.replace(".md", "_masked.md")
        output_path = target_dir / output_filename
        output_path.write_text(safe_content, encoding="utf-8")
        
        result["status"] = "SUCCESS"
        result["entities_masked"] = total_file_entities
        result["latency"] = time.time() - start_time
        return result

    except Exception as e:
        result["error"] = str(e)
        return result
