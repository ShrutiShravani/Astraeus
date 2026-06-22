def is_markdown_high_quality(markdown_text, engine_choice):
    # Rule 1: Minimum content threshold
    if len(markdown_text.strip()) < 50: return False
    
    # Rule 2: Table fidelity (If we used table engine, we expect table markers)
    if "table" in engine_choice and "|" not in markdown_text: return False
    
    # Rule 3: Noise threshold (Simple heuristic: percentage of non-alphanumeric)
    noise = sum(1 for c in markdown_text if not c.isalnum() and not c.isspace())
    if noise / max(len(markdown_text), 1) > 0.3: return False
    
    return True

# USE IT INSIDE process_page_in_process:
# After generating text_markdown or table_markdown:
if not is_markdown_high_quality(table_markdown, engine_choice):
    return {"page_num": page_num, "status": "FAILED", "error": "low_fidelity_output"}