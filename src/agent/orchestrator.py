from langgraph.graph import StateGraph,END,START
from typing import TypedDict, List, Annotated
import operator
from src.agent.nodes.auditor import auditor_node
from src.agent.nodes.semantic_cache_nodes import semantic_cache_check_node,finalize_audit
from src.agent.nodes.retriever import hybrid_retriever_node
from src.agent.nodes.python_repl import python_repl_node
from src.agent.nodes.purifier_node import prompt_purifier_node
from src.agent.nodes.audit_engine import audit_engine
from src.agent.nodes.system_guradrail import system_1_guard # The actual function
from src.agent.nodes.planner import planner_node
from src.agent.nodes.compaction_node import compaction_node
from src.agent.nodes.human_review import human_review_node
from src.agent.nodes.unified_generator import unified_generator_node
from src.agent.nodes.extractor import math_extractor_node
from typing import List
from src.agent.state import AgentState
from src.agent.nodes.retrieval_auditor import retrieval_auditor_node
from src.agent.nodes.divergence_analyst import divergence_analyst_node

def route_after_auditor(state:AgentState):
    if state.get("audit_engine_failed"):
        return "end"
    return state.get("target_node")


def route_after_extractor(state:AgentState):
    if state.get("extractor_failed"):
        return "end"
    
    return "python_repl"

def should_compact(state: AgentState):
    if len(state.get("audit_wiki", [])) >= 5:
        return "compact"
    return "generate"

def route_after_generator(state:AgentState):
    if state.get("force_compact") and state.get("generator_failed"):
        return "force_compact"
    if state.get("generator_failed"):
        return "human_review"
    
    return "user_auditor"


def route_after_planner(state: AgentState):
    if state.get("is_cached"):
        return "human_review"
    if state.get("target_node")=="END":
        return "end"
    if  len(state["plan"])==0:
        print("--- ROUTE: Direct to Generator (Wiki Match) ---")
        return "generator"
    if state.get("planner_failed"):
        return "end"      # or planner_failure_node
    \
    return "retriever"

def route_after_prompt_purifier(state):
    if state.get("prompt_purifier_failed"):
        return "end"
    if state.get("ask_user"):
        return "prompt_purifier"  # This halts the graph and returns control to main.py
    
    return "planner"

def route_after_retriever_auditor(state: AgentState):
    # This is where the decision happens!
    feedback= state.get("retriever_feedback")
    attempts = state.get("retriever_audit_attempts", 0)
    needs_revision = feedback.needs_revision

    if state.get("retriever_auditor_failed"):
        return "end"
  
    elif needs_revision:
        if attempts>3 and feedback.no_evidence_found:
            return "human_review"
        print(f"--- RETRIEVER REJECTED (Attempt {attempts}): LOOPING TO PLANNER ---")
        return "planner"
        
    return "extractor"

def route_after_compaction(state):
    if state.get("compaction_failed"):
        return "end"
    
    return "generator"

def route_cache(state):
    val = state.get("is_cached", False)
    print(val)

    if val is True:
        print("--- CACHE HIT: Proceeding to Review ---")
        return  route_after_planner
       
    else:
        print("continue to prompt purifier")
        return route_after_planner

def route_after_python_repl(state):
    if state.get("type")=="C":
        return "divergence_analyst"
    elif len(state.get("audit_wiki", [])) >= 5:
            return "compact"
    else:
        return "generator"

def route_after_retriever(state):
    if state.get("retriever_failed"):
        return "end"
    else:
        return "retriever_auditor"

def route_after_divergence_analyst(state):
    if state.get("divergence_analyst_failed"):
        return "end"
    elif len(state.get("audit_wiki", [])) >= 5:
            return "compact"
    else:
        return "generator"


def route_after_human(state:AgentState):
    decision=state.get("human_decision")
    #status=state.get("audit_status")
    if decision=="is_investigate" or decision == "refinement":
        return "guard"
        
    if decision == "pass":
        return "cache_add"

    return "end"
        

#define the workflow
workflow=StateGraph(AgentState)

workflow.add_node("cache_check", semantic_cache_check_node)
workflow.add_node("guard",system_1_guard)
workflow.add_node("prompt_purifier",prompt_purifier_node)
workflow.add_node("planner",planner_node)
workflow.add_node("retriever_auditor",retrieval_auditor_node)
workflow.add_node("compaction_node",compaction_node)
workflow.add_node("extractor",math_extractor_node)
workflow.add_node("divergence_analyst",divergence_analyst_node)
workflow.add_node("generator",unified_generator_node)
workflow.add_node("auditor",audit_engine)
workflow.add_node("retriever",hybrid_retriever_node)
workflow.add_node("python_repl",python_repl_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("cache_add",finalize_audit)
workflow.add_node("user_auditor",auditor_node)

workflow.add_edge("planner", "cache_check")

# 2. Security Conditional Edge
workflow.add_conditional_edges(
    "guard",
    # We use a lambda to check the boolean and return the routing string
    lambda state: "secure" if state["is_safe"] else "unsafe",
    {
        "secure": "prompt_purifier",
        "unsafe": END
    }
)

workflow.add_conditional_edges(
    "divergence_analyst",
    route_after_divergence_analyst,
    {
        "generator": "generator",
        "end": END
    }
)


workflow.set_entry_point("guard")



workflow.add_edge("user_auditor","auditor")
workflow.add_edge("cache_add", END)


# Conditional Edge 2: Auditor Feedback (Self-Correction Loop)
workflow.add_conditional_edges(
    "auditor",
    route_after_auditor,
    {    
        "generator": "generator",
        "human_review": "human_review",
        "end":END
    }
)

workflow.add_conditional_edges(
    "prompt_purifier",
    route_after_prompt_purifier,
    {    
        "prompt_purifier": "prompt_purifier",
        "planner":"planner",
        "end":END
    }
)

workflow.add_conditional_edges(
    "generator",
    route_after_generator,
    {    
        "user_auditor": "user_auditor",
        "end": END
    }
)

workflow.add_conditional_edges(
    "extractor",
    route_after_extractor,
    {    
        "python_repl": "python_repl", # NEW: Links 'investigate' to the cache node
        "end":END
      
    }
)


workflow.add_conditional_edges(
    "python_repl",
    route_after_python_repl,
    {    
        "generator": "generator", # NEW: Links 'investigate' to the cache node
        "divergence_analyst": "divergence_analyst",
        "end":END
      
    }
)


workflow.add_conditional_edges(
    "human_review",
    route_after_human,
    {    
        "guard": "guard", # NEW: Links 'investigate' to the cache node
        "cache_add": "cache_add",
        "end": END
    }
)

workflow.add_conditional_edges(
    "cache_check",
    route_after_planner,
    {    
        "retriever": "retriever",
        "generator":"generator",
        "human_review":"human_review",
        "end": END
    }
)

workflow.add_conditional_edges(
    "retriever",
    route_after_retriever,
    {    
        "retriever_auditor": "retriever_auditor",
        "end": END
    }
)


# In your graph definition:
workflow.add_conditional_edges(
    "retriever_auditor", 
    route_after_retriever_auditor,
    {
        "generator": "generator",
        "human_review": "human_review",
        "extractor": "extractor",
        "planner":"planner",
        "end": END
    }
)


workflow.add_conditional_edges(
    "compaction_node",
    route_after_compaction,
    {    
        "extractor": "extractor",
        "end": END
    }
)
