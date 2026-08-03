from torch import monitor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import time
from src.agent.state import AgentState, Planner
from dotenv import load_dotenv
import os
from src.utils.monitoring import PerformanceCallback
from src.utils.get_metrics import get_node_metrics
from src.utils.monitoring import log_to_mlflow
from src.utils.prompt_manager import PromptManager
from custom_logging import logger
from src.agent.state import AgentState, FollowUpOutput

promptloader= PromptManager()

perf_cb=PerformanceCallback()


# Primary: Cheap & Fast 
llm_mini = ChatOpenAI(model="gpt-4o-mini",streaming=True,temperature=0, max_retries=5)

# Backup: Smart & High-Limit (The Fallback)
llm_gpt4o = ChatOpenAI(model="gpt-4o",streaming=True, temperature=0)

# Resilient Brain for everyone
resilient_brain = llm_mini.with_fallbacks([llm_gpt4o])


def follow_up_node(state:AgentState):
    start_ts=time.time()
    plan = state.get("plan", [])

    
    planner_tasks= "\n".join([f"TASK_TITLE:{t.title}" for i, t in enumerate(plan)]) 
    current_turn = state.get("turn_count", 0) + 1
    prev_company = state.get("target_company",[])
    print(prev_company)
    prev_year = state.get("target_year",[])
    print(prev_year)
    
    MAX_ATTEMPTS = 2
    
    audit_wiki=state.get("audit_wiki",[])
    
 
    try:
        node_config = promptloader.prompts.get('follow_up', {})
        raw_template = node_config.get('follow_up_prompt')
        if not raw_template:
            raise ValueError("gap_analysis prompt not found")
        prompt_version = node_config.get('version', '1.0.0')
    except Exception as e:
        logger.exception(e)
        # FAIL SAFE: if the prompt itself can't load, don't silently drop
        # tasks -- pass everything through to retriever rather than lose
        # a task that genuinely needed retrieval.
        return {"gap_analysis_failed": True}
 
    already_verified_facts = ",".join([
        f"Metric: {item.task_name} | Year: {item.year} | Company: {item.company}"
        for item in audit_wiki
    ])
    task_list_text ="\n".join([f"TASK_TITLE:{t.title} | Source: {t.doc_source}" for i, t in enumerate(plan)]) 
 
    structured_gap_checker = resilient_brain.with_structured_output(FollowUpOutput, include_raw=True)
    gap_prompt = raw_template.format(
        audit_wiki=already_verified_facts,
        planner_tasks=task_list_text,
    )
 
    response = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = structured_gap_checker.invoke(gap_prompt, config={"callbacks": [perf_cb]})
            break
        except Exception:
            logger.exception(f"Gap analysis LLM invocation failed (attempt {attempt}/{MAX_ATTEMPTS})")
         
            if attempt == MAX_ATTEMPTS:
                # FAIL SAFE: same principle -- if the check itself can't
                # run, don't silently drop tasks. Pass everything to
                # retriever; worst case is a redundant retrieval, which
                # is far safer than silently skipping a needed one.
                logger.warning("Gap analysis failed after retries -- passing all tasks through unfiltered.")
                return {"plan": plan, "answered_by_wiki": [], "gap_analysis_failed": True}
 
    result = response["parsed"]
 
        
    tasks_to_retrieve = [c for c in result.classifications if c.status == "needs_retrieval"]
    tasks_already_answered = [c for c in result.classifications if c.status == "answered"]

    verified_answered = []
    
    follow_up_sample_log=[]
    for classification in tasks_already_answered:
        real_task = plan_by_title.get(classification.matched_task_title)
        if real_task is None:
            continue 
        matching_wiki_entries = [
        item for item in audit_wiki
        if item.company == real_task.extracted_company and item.year == real_task.extracted_year
    ]

        follow_up_sample_log.append({
        "node": "follow_up",
        "turn": current_turn,
        "task_title": real_task.title,
        "task_company": real_task.extracted_company,
        "task_year": real_task.extracted_year,
        "matched_wiki_fact_text": classification.matched_wiki_fact,
        "wiki_entries_at_company_year": [
            {"metric": e.task_name, "evidence": e.evidence, "source": e.source, "page": e.page}
            for e in matching_wiki_entries
        ],
        "llm_said": "answered",
    })

        if matching_wiki_entries:
            verified_answered.append(classification)
        else:
            # FAIL SAFE: the LLM said "answered" but no wiki entry actually
            # exists for this task's company/year. Don't just log it and
            # move on -- demote it to needs_retrieval so it still reaches
            # Retriever, instead of silently being trusted as answered
            # when it clearly isn't.
            logger.warning(
                f"Gap analysis wrongly marked '{classification.matched_task_title}' as answered -- "
                f"no wiki entry found for company={real_task.extracted_company}, year={real_task.extracted_year}. Demoting to needs_retrieval."
            )
            tasks_to_retrieve.append(real_task)


    matching_wiki_entries = [
        item for item in audit_wiki
        if item.company == task.extracted_company and item.year == task.extracted_year
        for task in tasks_already_answered 
    ]
    
    # Build an exact-title lookup ONCE -- O(1) matching instead of a nested
    # loop, and exact equality instead of fragile substring ("in") matching.
    plan_by_title = {task.title: task for task in plan}
    
    follow_up_tasks = []
    for classification in tasks_to_retrieve:
        matched_task = plan_by_title.get(classification.matched_task_title)
        if matched_task is not None:
            # The FULL original Task object goes to Retriever -- doc_source,
            # zone, extracted_company, extracted_year all intact, exactly as
            # the Planner decomposed it. We are NOT reconstructing a task
            # from the classification -- we're looking up the real one.
            follow_up_tasks.append(matched_task)
        else:
            # The LLM didn't echo the title exactly (paraphrased it, or
            # invented one). FAIL SAFE: log and count it -- do NOT silently
            # drop it, since that could mean a real task never reaches
            # Retriever. Worth alerting on if this ever fires in production.
            logger.warning(
                f"Gap analysis returned unmatched title: '{classification.matched_task_title}'"
                f"-- no exact match found in planner tasks."
            )
            return {"gap_analysis_failed": True}
 
    logger.info(
        f"Gap analysis: {len(tasks_already_answered)} task(s) already answered, "
        f"{len(tasks_to_retrieve)} task(s) need retrieval."
    )

    metrics_getter = get_node_metrics("gap_analysis", response, perf_cb, start_ts)
    node_results = metrics_getter(state)
    node_results["prompt_version"] = prompt_version

    try:
        log_to_mlflow("gap_analysis", node_results, step=state.get("turn_count", 0))
    except Exception as e:
        logger.warning(f"MLflow node logging failed: {e}")

    return {
        "follow_up_sample_log": follow_up_sample_log,
        "task_to_retrieve": tasks_to_retrieve,
        "turn_count": current_turn,
        "answered_by_wiki": tasks_already_answered,
        "follow_up_tasks":follow_up_tasks,
        **node_results,
        "steps": [
            f"Follow up: {len(tasks_to_retrieve)} task(s) to retrieve, "
            f"{len(tasks_already_answered)} matched to existing wiki facts."
        ],
    }