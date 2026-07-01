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
import re
from custom_logging import logger


promptloader= PromptManager()
perf_cb=PerformanceCallback()

llm_pro = ChatOpenAI(model="gpt-4o",streaming=True,temperature=0, max_retries=3)

    # Snapshot Version (Secondary Backup)
llm_pro_snap = ChatOpenAI(model="gpt-4o-2024-08-06",streaming=True, temperature=0)

# Cheap & Fast (Final Safety Net)
llm_mini = ChatOpenAI(model="gpt-4o",streaming=True, temperature=0, max_retries=5)

# 2. THE CHAIN OF COMMAND
# For heavy tasks (Generator/Auditor): 
# Try GPT-4o -> Try Snapshot -> Try Mini (Fail-safe)
resilient_pro = llm_pro.with_fallbacks([llm_pro_snap, llm_mini])
resilient_mini = llm_mini.with_fallbacks([llm_pro])

def extract_from_verified_evidence(context):
    coordinates = []
    for c in context:
        coordinates.append({"source": c.get('source'),
        "page": c.get('page'),
        "company": c.get('company'),
        "year": c.get('year'),
        "evidence": c.get('evidence')}
        )
    
    return  coordinates
    

def unified_generator_node(state: AgentState):
    start_ts=time.time() 
    current_turn = state.get("turn_count", 0) + 1
    previous_draft = state.get("generation","")
    query_type = state.get("type") 
    math_result = state.get("calculation_result") 
    feedback = state.get("reflection_feedback")
    raw_context = state.get("final_context","")
    is_follow_up = state.get("is_follow_up", False)
    saved_context = state.get("context_history", [])
    audit_wiki=state.get("audit_wiki","")
    alignment_status=state.get("alignment_status")
    narrative_conlfict_score=state.get("narrative_conflict_score")
    divergence_type =state.get("divergence_type")
    divergence_reason =state.get("divergence_reason")
    conflict_rationale = state.get("conflict_rationale")
    
    consolidated_block = ""
    if audit_wiki and is_follow_up:
        wiki_str = "\n".join([f"---[COORD:  Company: {item.company} | Year: {item.year} | Source: {item.source} | PAGE: {item.page}] ---\nVERIFIED METRIC: {item.evidence}: {item.quote})" for item in audit_wiki])
        consolidated_block += f"### [1] VERIFIED FINANCIAL DATA (Wiki):\n{wiki_str}\n"
    
    if is_follow_up and saved_context:
        consolidated_block += f"\n### [2] PREVIOUS NARRATIVE EVIDENCE:\n{saved_context[-1]}\n"
    

    planner_tasks = state.get("plan")
    active_task_list = [f"[NEW RESEARCH]: {t.title}" for t in planner_tasks]
    if is_follow_up:
        if not active_task_list:
        # Scenario: Total Wiki Match
         active_task_list = [f"[WIKI SYNTHESIS]: Answer '{state['query']}' ONLY using Verified FINANCIAL Data and PREVIOUS NARRTAIVE EVIDENCE"]
        else:
        # Scenario: Hybrid (Some New, Some Wiki)
        # We add a virtual task to remind the LLM to integrate existing facts
         active_task_list.append(f"[INTEGRATION]: Reconcile the new findings with existing metrics in the VERIFIED FINANCIAL DATA and and PREVIOUS NARRTAIVE EVIDENCE")
    
    try:
        node_config = promptloader.prompts.get('unified_generator', {})
        raw_template = node_config.get('unified_generator_prompt')
        prompt_version = node_config.get('version', '1.0.0')
    except Exception as e:
        logger.exception(e)
        return {
             "generator_failed": True
       
        }

    final_report_with_evidence = ""
 
    parser = JsonOutputParser(pydantic_object=FinalGeneration)
    
    # 1. Coordinate-Based Context Construction
    
    evidence = extract_from_verified_evidence(raw_context)
    print(f"evdience:{evidence}")
 
    if is_follow_up:
        # In a follow-up, we show the hierarchy: Wiki [1], History [2], New [3]
        final_context_for_llm = f"{consolidated_block}\n### [3] NEW INVESTIGATION FINDINGS:\n{evidence}"
    else:
        # In Turn 1, there is no [1] or [2], so we just give it [3]
        final_context_for_llm = f"### [3] NEW INVESTIGATION FINDINGS:\n{evidence}"

    follow_up_instruction=" "
    if is_follow_up:
        follow_up_instruction = f"""

            ### FOLLOW-UP INVESTIGATION PROTOCOL (TURN 2+):
            You are now in a multi-turn audit. The context includes previous findings.
            
            1. **SKEPTICAL MATCHING**: If the query refers to a metric in [VERIFIED FINANCIAL DATA], PRIORITIZE and use that value. 
            2. **EVIDENCE LINKING**: For any verified fact, scan [PREVIOUS NARRATIVE EVIDENCE] for justification.
            3. **TASK DIFFERENTIATION**: 
            - If an 'Active Task' exists in the plan for a metric, prioritize [NEW INVESTIGATION FINDINGS].
            - If NO 'Active Task' exists in the plan for a metric  but the user asks about the metric, use [VERIFIED FINANCIAL DATA][PREVIOUS NARRATIVE EVIDENCE].
            
        """

    is_correction = (feedback is not None and feedback.needs_revision and state.get("target_node") == "generator")
    
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
    filter_rule = f"""
    You are operating in a ZERO-TRUST environment. Your output must be a mirror of the provided [final_context_str].

    ### THE 9 GOLDEN RULES OF EXTRACTION:
    1. NO REWRITES: Do not re-write sections that are already accurate. Focus only on fulfilling the current [planner_tasks].
    2. CLOSED-WORLD ENFORCEMENT: If a fact, number, or claim is NOT in the provided context, it DOES NOT EXIST. You are strictly forbidden from adding external knowledge.
    3. COORDINATE-FIRST: Every row in your 'EVIDENCE TABLE' must be preceded by a [COORD] header (e.g., [Nike_2019_10K | Page 42]).
    4. NO GUESSING: 'Source' and 'Page' columns in your JSON/Table MUST match the [COORD] header exactly.
    5. SURGICAL EXTRACTION: Extract ONLY the sentences that directly answer the query. Discard "fluff" or unrelated content from the same page.
    6. MULTI-YEAR ISOLATION: For multi-year queries, perform SEPARATE math for EACH year. Do not aggregate. Show raw numbers for each specific year's calculation.
    7. SOURCE DUALITY (Type B/C): You MUST include evidence from BOTH the 10-K and the Transcript if both are provided.
    8. NARRATIVE SYNTHESIS (Type B): If multiple sources exist, compare them. If one source exists, perform a deep-dive on that specific explanation.
    9. ZERO-ROUNDING POLICY: Preserve every decimal point. If the evidence says '$4,520,311.42', you write '$4,520,311.42'. Do not use 'M' or 'B' unless the raw text does.
    10."SYSTEM ALERT: The final_context_str provided is the ONLY existence of reality. If final_context_str is empty or missing a specific metric, you MUST output 'DATA_NOT_FOUND' for that metric. DO NOT generate a table based on previous knowledge. If you mention a number not in the context, the audit will fail and the system will shut down."
    """
    
   
    mode_instruction = f"""
    ### FINAL PUBLICATION MODE:
    {filter_rule}
    - Remove all internal critiques or drafting notes.
    - **STRICT PROFESSIONALISM**: Refine the report to be accurate, senior-level, and professional.
    - **TOKEN EFFICIENCY**: Be extremely precise and on-point. 
    - **PRECISION**: Eliminate all unnecessary 'fluff', introductory filler, or redundant explanations. 
    - **INTEGRITY**: Do not shorten the EVIDENCE table, but ensure the EXECUTIVE SUMMARY is a high-density 'Flash Report'.
    - Use only information in final_context_str to create evidence table
    - Ensure 'EVIDENCE TABLE' columns: Fiscal Year | Source | Page | Year| Evidence Description | Relevance.
    - Use clean Markdown; no 'Step 1' or 'Chain of Thought' headers.
    - Give correct and complete used_evdience_texts with source,page num and doc_type
    - Cleary state year against each row in evidence table
    """


    if previous_draft and  is_correction:
            # RULE: REWRITE (Fix the mistake in the existing report)
            merge_instruction = f"""
            ### THE REVISION PROTOCOL (CORRECTION MODE):
            The Auditor rejected your previous draft. 
            FIX the errors mentioned in the REVISION REQUIRED section.
            DO NOT APPEND. REWRITE the report so it is clean and accurate.
            """
    else:
        merge_instruction=""

    # Fix: Cleaned up the double prompt nesting and clarified Type C instruction
    system_prompt = raw_template.format(
        planner_tasks=planner_tasks,
        merge_instruction=merge_instruction,
        revision_instruction=revision_instruction,
        follow_up_instruction=follow_up_instruction,
        mode_instruction=mode_instruction,
        previous_draft=previous_draft,  # Logic handled!
        query_type= query_type,
        math_result=math_result,
        final_context_str=final_context_for_llm,
        alignment_status = alignment_status,
        narrative_conflict_score =narrative_conlfict_score,
        divergence_type= divergence_type,
        divergence_reason = divergence_reason,
        conflict_rationale = conflict_rationale

      
    )
    
    system_prompt_format= system_prompt +"\n\n" + parser.get_format_instructions()
    
    user_content = f"QUERY: {state['query']}\n\nCONTEXT:\n{raw_context}"
    
    generator_chain= (ChatPromptTemplate.from_messages([("system", "{system_prompt_format}"), # Match this...
        ("human", "{user_content}")
    ])| resilient_pro)
    
    try:
        full_response = None
        
        input_map={
            "system_prompt_format": system_prompt_format,
            "user_content": user_content
        }
        for chunk in generator_chain.stream(
            input_map,
            config={"callbacks": [perf_cb]}
        ):
            if full_response is None:
                full_response=chunk
            else:
                full_response+=chunk

      
    except Exception as e:
        logger.exception(
            f"Generator LLM invocation failed | prompt_version={prompt_version}"
        )
        return {
            "generator_failed": True
        }

    raw_text= full_response.content
    if not raw_text or len(raw_text.strip())==0:
        logger.error(
        f"Generator returned empty response | prompt_version={prompt_version}"
    )
        return {
            "generator_failed": True
        }

    #clean tetx
    cleaned_text= raw_text.replace("```json", "").replace("```", "").strip()

    plan_output= parser.parse(cleaned_text)
    
    try:
    #get raw results for metrics calculation
        raw_res_metrics= {
            "raw":full_response,
            "parsed":plan_output
        }
    except Exception as e:
        logger.exception(
            f"Generator JSON parsing failed | prompt_version={prompt_version}"
        )
        return {
            "generator_failed": True
        }
        
    metrics_getter= get_node_metrics("unified_generator",raw_res_metrics,perf_cb,start_ts)
    
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version
    try:
        log_to_mlflow("unified_generator",node_results,step=current_turn)
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")

    final_report = plan_output.get("report")
    print(f"final_report:{final_report}")
    

    def clean_coords(val):
        if val is None: return ""
        return str(val).replace("-", "").replace(" ", "")

    final_context=[]

    if is_correction:
        # In a correction, the Retriever didn't run. Use the history we already have.
        final_context = saved_context
    else:
    
        final_context=raw_context
    
    """
    #APPEND CONTETX HISTORY
    # 1. Initialize a clean string for the NEW evidence found in this run
    current_evidence_str = "\n\n### AUDIT EVIDENCE & SOURCE CITATIONS (NEW):\n"
    if not final_context:
        current_evidence_str += "NO MATCHING EVIDENCE FOUND IN CONTEXT.\n"
    else:
        sorted_context= sorted(final_context,key=lambda x:str(x.get('year')))
        for idx,c in enumerate(sorted_context):
            # Use += to append, not = to overwrite!
            company= c.get('company') or c.get('company')
            year= c.get('year') or c.get('year')
            src = c.get('source') or c.get('SOURCE') or "Unknown"
            pg = c.get('page') or c.get('PAGE') or "?"
            txt = c.get('evidence') or c.get('content') or "No text."
            current_evidence_str += f"[{idx+1}] Company: {company} | Year: {year} | Source: {src} | Page: {pg}\nSnippet: {txt[:2000]}...\n\n"
            print(current_evidence_str)
 
        # Just the new report and its evidence
        final_report_with_evidence = f"{final_report}\n{current_evidence_str}"
        print("final_report appended with evidence")
    """
    
    final_report_with_evidence = f"{final_report}\n{raw_context}"
    print("final_report appended with evidence")
    score= state.get("narrative_conflict_score")
    if score>1:
        print("Alert!!! divergence detected")
    else:
        print("No divergence detected")
    

    return {
        "generation": final_report_with_evidence,
        "turn_count": current_turn,
        "context_history":final_context,
        **node_results,
        "steps": [
            f"- Mode: {query_type} Analysis.",
            f"- Evidence Integration: Synthesized {len(raw_context)} source snippets.",
            f"- Refinement: {'Applied auditor feedback' if is_correction else 'Initial generation'}.",
            f"- Audit Status: Marked as {state.get('audit_status', 'Pending')}."
        ]
    }