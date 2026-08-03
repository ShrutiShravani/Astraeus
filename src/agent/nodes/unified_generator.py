import pandas
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
from openai import RateLimitError


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

    

def unified_generator_node(state: AgentState):
    start_ts=time.time() 
    current_turn = state.get("turn_count", 0) + 1
    previous_draft = state.get("generation","")
    query_type = state.get("type") 
    query_history= state.get("query_history")
    math_result = state.get("calculation_result") 
    feedback = state.get("reflection_feedback")
    is_follow_up = state.get("is_follow_up", False)
    audit_wiki=state.get("audit_wiki","")
    alignment_status=state.get("alignment_status")
    narrative_conlfict_score=state.get("narrative_conflict_score")
    divergence_type =state.get("divergence_type")
    divergence_reason =state.get("divergence_reason")
    conflict_rationale = state.get("conflict_rationale")
    wiki_archive = state.get("wiki_archive", "")
    max_retries=2
    attempts=0

    missing_entries=[]
    missing_tasks = []
    found_entries=[]
    # -----------------------
    for item in audit_wiki:
        entry = (f"---[COORD:  Company: {item.company} | Year: {item.year} | Source: {item.source} | PAGE: {item.page}] ---\nVERIFIED METRIC: {item.evidence})")
        if item.status.lower() == "missing":
           if item.status.lower() == "missing":
             missing_entries.append(entry)
             missing_tasks.append(item.task_name)
        else:
             found_entries.append(entry)

    wiki_str = "\n".join(found_entries)

    final_context_for_llm = f""" ### HISTORICAL LEDGER (ARCHIVE)
    {wiki_archive if wiki_archive else "No historical archive yet."}
    ### ACTIVE INVESTIGATION (WIKI)
    {wiki_str if wiki_str else "No new findings this turn."}
    """
    
    missing_warning = ""
    if missing_tasks:
        missing_warning = f"""
        ⚠️ DATA GAPS — THE FOLLOWING task from [planner_task] COULD NOT BE RETRIEVED:
        {chr(10).join(f'- {t}' for t in missing_tasks)}
        
        STRICT RULE: Do NOT fabricate or estimate values 
        for these items. Instead write:
        "DATA NOT AVAILABLE: [task name] could not be 
        verified from source documents."
        """
    planner_tasks = state.get("plan")
    
    try:
        node_config = promptloader.prompts.get('unified_generator', {})
        raw_template = node_config.get('unified_generator_prompt')
        prompt_version = node_config.get('version', '1.0.0')
    except Exception as e:
        logger.exception(e)
        return {
             "generator_failed": True
       
        }

 
    parser = JsonOutputParser(pydantic_object=FinalGeneration)
    
    # 1. Coordinate-Based Context Construction
    

    follow_up_instruction=" "
    if is_follow_up:
        follow_up_instruction = f"""

            ### FOLLOW-UP INVESTIGATION PROTOCOL (TURN 2+):
            You are now in a multi-turn audit. The context includes previous findings.
            
            **critical**: 
             Use both [query_history] and [planner_tasks] to understand user query and then answer using [final_context_str] .
            
        """

    is_correction = (feedback is not None and feedback.needs_revision and state.get("target_node") == "generator")
    
    revision_instruction = ""
    if previous_draft and is_correction:
        print("correction required")
        revision_instruction = f"""

         ### THE REVISION PROTOCOL (CORRECTION MODE):
            The Auditor rejected your previous draft {previous_draft}. 
            FIX the errors mentioned in the REVISION REQUIRED section.
            DO NOT APPEND. REWRITE the report so it is clean and accurate.
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
    8. ZERO-ROUNDING POLICY: Preserve every decimal point. If the evidence says '$4,520,311.42', you write '$4,520,311.42'. Do not use 'M' or 'B' unless the raw text does.
    9."SYSTEM ALERT: The [final_context_str] provided is the ONLY existence of reality. If final_context_str is empty or missing a specific metric, you MUST output 'DATA_NOT_FOUND' for that metric. DO NOT generate a table based on previous knowledge. If you mention a number not in the context, the audit will fail and the system will shut down."
    """
    
   
    mode_instruction = f"""
    ### FINAL PUBLICATION MODE:
    {filter_rule}
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



    # Fix: Cleaned up the double prompt nesting and clarified Type C instruction
    system_prompt = raw_template.format(
        planner_tasks=planner_tasks,
        query_history= query_history if query_history else "No prior context",
        revision_instruction=revision_instruction,
        follow_up_instruction=follow_up_instruction,
        mode_instruction=mode_instruction,
        query_type= query_type,
        math_result=math_result,
        final_context_str= final_context_for_llm,
        alignment_status = alignment_status,
        narrative_conflict_score =narrative_conlfict_score,
        divergence_type= divergence_type,
        divergence_reason = divergence_reason,
        conflict_rationale = conflict_rationale,
        missing_warning= missing_warning if missing_warning else "N/A"

      
    )
    
    system_prompt_format= system_prompt +"\n\n" + parser.get_format_instructions()
    
    user_content = f"QUERY: {state['query']}"

    if len(system_prompt_format)+len(user_content)>25000:
        logger.warning("Context window bloat: Truncating evidence to prevent 429 error.")
        # Logic: Either truncate or return a failure to trigger an Audit Summary node
        return {
            "generator_failed": True,
            "error": "Context length exceeded. Please run a 'Summarize Evidence' task.",
            "force_compact": True
        }

    
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

    except RateLimitError:
        logger.error("TPM Limit Exceeded: Context window too large.")
        return {
            "generator_failed": True, 
            "error": "The audit is getting too complex. Please request a summary of the findings so far."
        }

      
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

    plan_output = None
    
    while attempts<max_retries and plan_output is None:
        try:
            cleaned_text= raw_text.replace("```json", "").replace("```", "").strip()

            plan_output= parser.parse(cleaned_text)
            if plan_output:
                print("--- SUCCESS: Parsed JSON ---")
                break # Exit the loop if successful
            else:
                raise ValueError("Parser returned None")
            
        except Exception as e:
            attempts += 1
            logger.error(f"Attempt {attempts} failed: {e}")
            # DO NOT RETURN HERE. Let the loop continue to the next attempt.
    
   
    #get raw results for metrics calculation
    raw_res_metrics= {
            "raw":full_response,
            "parsed":plan_output
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
    

    print("final_report appended with evidence")
    score= state.get("narrative_conflict_score")
    if type == "C" and score>1:
        print("Alert!!! divergence detected")
    else:
        print("No divergence detected")
    

    return {
        "generation": final_report,
        "turn_count": current_turn,
        **node_results,
        "steps": [
            f"- Mode: {query_type} Analysis.",
            f"- Refinement: {'Applied auditor feedback' if is_correction else 'Initial generation'}.",
            f"- Audit Status: Marked as {state.get('audit_status', 'Pending')}."
        ]
    }