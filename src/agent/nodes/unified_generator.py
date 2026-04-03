from src.agent.state import AgentState,FinalGeneration
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
import time
from src.utils.monitoring import log_to_mlflow
from src.utils.prompt_manager import PromptManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

promptloader= PromptManager()
perf_cb=PerformanceCallback()

llm_pro = ChatOpenAI(model="gpt-4o-mini",streaming=True,temperature=0, max_retries=3)

    # Snapshot Version (Secondary Backup)
llm_pro_snap = ChatOpenAI(model="gpt-4o-2024-08-06",streaming=True, temperature=0)

# Cheap & Fast (Final Safety Net)
llm_mini = ChatOpenAI(model="gpt-4o",streaming=True, temperature=0, max_retries=5)

# 2. THE CHAIN OF COMMAND
# For heavy tasks (Generator/Auditor): 
# Try GPT-4o -> Try Snapshot -> Try Mini (Fail-safe)
resilient_pro = llm_pro.with_fallbacks([llm_pro_snap, llm_mini])
resilient_mini = llm_mini.with_fallbacks([llm_pro])

def unified_generator_node(state: AgentState):
    start_ts=time.time() 
    current_turn = state.get("turn_count", 0) + 1
    previous_draft = state.get("generation", "")
    query_type = state.get("type") 
    math_result = state.get("calculation_result") 
    feedback = state.get("reflection_feedback")
    context = state.get("context")
    saved_context = state.get("context_history", [])
    planner_tasks = state.get("plan")
    node_config = promptloader.prompts.get('unified_generator', {})
    raw_template = node_config.get('unified_generator_prompt')
    prompt_version = node_config.get('version', '1.0.0')
    final_report_with_evidence = ""
    final_context = []
    parser = JsonOutputParser(pydantic_object=FinalGeneration)
    
    # 1. Coordinate-Based Context Construction
    context_str = ""
    for c in context:
        context_str += f"--- [COORD: {c['source']} | PAGE: {c['page']}] ---\n{c['evidence']}\n\n"

    is_correction = (feedback is not None and feedback.needs_revision and feedback.target_node == "generator")
    
    revision_instruction = ""
    if is_correction:
        print("correction required")
        revision_instruction = f"""
        ### REVISION REQUIRED
        Audit Critique: {feedback.critique}
        Error Type: {feedback.err_type or "N/A"}
        Exact Trace: {feedback.exact_trace or "N/A"}
        Reason: {feedback.reason}
        Expected Fix: {feedback.expected or "N/A"}
        Action: {feedback.action or "N/A"}

        Revise the report strictly using the audit feedback above.
        Do not introduce new facts outside the provided evidence.
        Preserve all correct facts, calculations, and valid citations from the previous draft.
        """
            

    # 2. Refined Rule Set
    filter_rule = """
    ### CONSISTENCY & COORDINATE RULES:
    1. Do not re-write sections that are already accurate.YOU MUST FILTER THE CORRCET EVIDENCES THAT DIRECTLY SUPPORTS THE QUERY AND MATH CALCULaTION.
    2. COORDINATE-FIRST: Every row in the 'EVIDENCE TABLE' must be linked to a [COORD] header.
    3. NO GUESSING: 'Source' and 'Page' columns MUST match the [COORD] header exactly.
    4. SURGICAL EXTRACTION: Only extract sentences that directly answer the query. Discard unrelated content on the same page.
    5. MULTI-YEAR CALCULATION: If the query spans multiple years, you MUST perform separate math calculations and results for EACH year mentioned. Do not aggregate them into a single value unless specifically asked. Each year's calculation must show the raw numbers used from the evidence.
    6. TYPE B/C REQUIREMENT: You must include evidence from BOTH the 10-K and the Transcript if asked.
    7. Type B: Synthesize the narrative strategy. If multiple sources are provided, compare them. If only one source is provided, perform a deep-dive analysis of that source's explanation.
    8. ZERO-ROUNDING POLICY: You are STRICTLY FORBIDDEN from rounding any numerical values. 
       - If the evidence says '$4,520,311.42', you MUST write '$4,520,311.42'. 
       - Do NOT use abbreviations like 'M' for Million or 'B' for Billion unless the raw evidence uses them. 
       - Every decimal point must be preserved. Failure to do so will result in an Audit Rejection.
    """
    
   
    mode_instruction = f"""
    ### FINAL PUBLICATION MODE:
    {filter_rule}
    - Remove all internal critiques or drafting notes.
    - **STRICT PROFESSIONALISM**: Refine the report to be accurate, senior-level, and professional.
    - **TOKEN EFFICIENCY**: Be extremely precise and on-point. 
    - **PRECISION**: Eliminate all unnecessary 'fluff', introductory filler, or redundant explanations. 
    - **INTEGRITY**: Do not shorten the EVIDENCE table, but ensure the EXECUTIVE SUMMARY is a high-density 'Flash Report'.
    - Ensure 'EVIDENCE TABLE' columns: Fiscal Year | Source | Page | Evidence Description | Relevance.
    - Use clean Markdown; no 'Step 1' or 'Chain of Thought' headers.
    - Give correct and complete used_evdience_texts with source,page num and doc_type
    """
    structure = f"""
    1. EXECUTIVE SUMMARY
    2. ANALYSIS 
    3. EVIDENCE TABLE
    4. Math Calculation and Result
    5. **used_evidence_texts**
    """
   
    is_investigation = (state.get("is_investigate"),False)
    merge_instruction=""
    if previous_draft:
        if is_investigation:
            merge_instruction = f"""
            ### THE AUDIT LEDGER PROTOCOL (FOLLOW-UP MODE):
            You are appending a new mission to an existing Forensic Audit.
            Do NOT provide two separate reports. You must merge the Previous Audit Findings with the New Forensic Discovery into a single, seamless narrative.Use a 'Cumulative Audit Result' section.Do not repeat anything.
            CRITICAL: YOU ARE OPERATING ON AN APPEND-ONLY LEDGER. 
            DELETING PREVIOUS DATA IS A BREACH OF AUDIT INTEGRITY.
            STRICTLY DO NOT REMOVE PREVIOUS MATH CALCULATION AND RESULT.Maintain a 'Calculations & Variance' table that includes both old and new data points.
    
            
            1. **NARRATIVE SYNTHESIS (Sections 1 & 2)**: 
            - DO NOT delete the existing Summary or Analysis.
            - MERGE and INTEGRATE the new findings for "{state['query']}" into these sections. 
            
            
            2. **IMMUTABLE LOGS (Sections 3, 4, & 5)**:
            CRITICAL:
            - **Evidence Table**: e the PREVIOUS DRAFT's table as your starting point. Copy every row exactly, then append new rows.
            - **Math Ledger**: DO NOT OVERWRITE. If the previous draft has  Math calculation,KEEP IT DO NOT REMOVE DRAFT MATH CALCUALTION. Add your new calculation for '{state['query']}' as a separate bullet point below it as 'Calculation [Iteration 2]'.
            - **used_evidence_texts (Section 5)**: This must be a cumulative list. If the previous draft had 4 snippets, and you found 2 new ones, Section 5 MUST contain 6 snippets total.ALSO SUMAMRIZE ALL EVDIENCES HERE
            
            3. **CONFLICT RESOLUTION**:
            - If new evidence contradicts the PREVIOUS DRAFT, do not delete the old text. Instead, add a 'CONTRADICTION ALERT' note  highlight the variance, but keep both sets of calculations visible to maintain the audit trail
            """
        elif is_correction:
            # RULE: REWRITE (Fix the mistake in the existing report)
            merge_instruction = f"""
            ### THE REVISION PROTOCOL (CORRECTION MODE):
            The Auditor rejected your previous draft. 
            FIX the errors mentioned in the REVISION REQUIRED section.
            DO NOT APPEND. REWRITE the report so it is clean and accurate.
            """
    # Fix: Cleaned up the double prompt nesting and clarified Type C instruction
    system_prompt = raw_template.format(
        planner_tasks=planner_tasks,
        merge_instruction=merge_instruction,
        revision_instruction=revision_instruction,
        mode_instruction=mode_instruction,
        previous_draft=previous_draft,  # Logic handled!
        query_type= query_type,
        math_result=math_result,
        structure=structure
    )
    
    system_prompt_format= system_prompt +"\n\n" + parser.get_format_instructions()
    
    user_content = f"QUERY: {state['query']}\n\nCONTEXT:\n{context_str}"
    
    generator_chain= (ChatPromptTemplate.from_messages([("system", "{system_prompt_format}"), # Match this...
        ("human", "{user_content}")
    ])| resilient_pro)
    
    full_response = None
    
    input_map={
        "system_prompt_format": system_prompt_format,
        "user_content": user_content
    }
    for chunk in generator_chain.stream(
        input_map,
         config={"callbacks": [perf_cb]}
    ):

     full_response=chunk

    raw_text= full_response.content
    #clean tetx
    cleaned_text= raw_text.replace("```json", "").replace("```", "").strip()

    plan_output= parser.parse(cleaned_text)

    #get raw results for metrics calculation
    raw_res_metrics= {
        "raw":full_response,
        "parsed":plan_output
    }
    metrics_getter= get_node_metrics("unified_generator",raw_res_metrics,perf_cb,start_ts)
    
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version
    log_to_mlflow("unified_generator",node_results,step=current_turn)
    final_report = plan_output.get("report")

    def clean_coords(val):
        if val is None: return ""
        return str(val).replace("-", "").replace(" ", "")
    if is_correction:
        # In a correction, the Retriever didn't run. Use the history we already have.
        final_context = saved_context
    else:
        cited_coords= {(clean_coords(coord.source),clean_coords(coord.page)) for coord in plan_output.used_coordinates}
        print(cited_coords)
        for coord in plan_output.used_coordinates:
                print(f"Source: {coord.source}, Page: {coord.page}")
        for c in context:
            c_source = clean_coords(c.get('source'))
            c_page = clean_coords(c.get('page'))
            print(f"c_source:{c_source}")
            print(f"c_page:{c_page}")
            # Check 1: Does this chunk's Source and Page exist in the LLM's used_coordinates?
            if (c_source,c_page) in cited_coords:
                print("match_data_found")
                
                # Check 2: Does this specific chunk contain one of the snippets?
                # Using a partial match (first 50 chars) makes it robust against minor formatting changes
            
                final_context.append(c)
                print("appended")

    #APPEND CONTETX HISTORY
    # 1. Initialize a clean string for the NEW evidence found in this run
    current_evidence_str = "\n\n### AUDIT EVIDENCE & SOURCE CITATIONS (NEW):\n"
    if not final_context:
        current_evidence_str += "NO MATCHING EVIDENCE FOUND IN CONTEXT.\n"
    else:
        for idx,c in enumerate(final_context):
            # Use += to append, not = to overwrite!
            src = c.get('source') or c.get('SOURCE') or "Unknown"
            pg = c.get('page') or c.get('PAGE') or "?"
            txt = c.get('evidence') or c.get('content') or "No text."
            current_evidence_str += f"[{idx+1}] Source: {src} | Page: {pg}\nSnippet: {txt[:2000]}...\n\n"

    # 2. Handle the "Joined" report logic
    if previous_draft and is_investigation:
        # We need to turn the list of old context dictionaries into a readable string
        old_evidence_text = "\n### PREVIOUS AUDIT CONTEXT:\n"
        for old_c in saved_context:
            osrc = old_c.get('source') or old_c.get('SOURCE')
            opg = old_c.get('page') or old_c.get('PAGE')
            old_evidence_text += f"- {osrc} (p.{opg}): {old_c.get('evidence')[:500]}...\n"
        
        # Combine: New Report + New Evidence + Old Context History
        final_report_with_evidence = f"{final_report}\n{current_evidence_str}\n{old_evidence_text}"
    else:
        # Just the new report and its evidence
        final_report_with_evidence = f"{final_report}\n{current_evidence_str}"
     
    decision=state.get("human_decision")
    if  decision=="pass":
        with open("final_report.jsonl", "a") as f:
            print(f"final_report:{final_report_with_evidence}")
            f.write(json.dumps({"report": final_report_with_evidence}))
    

    return {
        "generation": final_report_with_evidence,
        "turn_count": current_turn,
        "context_history": list(saved_context + final_context) if is_investigation else final_context,
        **node_results,
        "steps": [
            f"- Mode: {query_type} Analysis.",
            f"- Evidence Integration: Synthesized {len(context_str)} source snippets.",
            f"- Refinement: {'Applied auditor feedback' if is_correction else 'Initial generation'}.",
            f"- Audit Status: Marked as {state.get('audit_status', 'Pending')}."
        ]
    }