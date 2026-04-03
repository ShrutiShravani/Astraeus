from langgraph.graph import StateGraph,END,START
from typing import TypedDict, List, Annotated
import operator
from src.agent.nodes.auditor import auditor_node
from src.agent.nodes.semantic_cache_nodes import semantic_cache_check_node,finalize_audit
from src.agent.nodes.retriever import hybrid_retriever_node
from src.agent.nodes.python_repl import python_repl_node
from src.agent.nodes.audit_engine import audit_engine
from src.agent.nodes.system_guradrail import system_1_guard # The actual function
from src.agent.nodes.planner import planner_node
from src.agent.nodes.human_review import human_review_node
from src.agent.nodes.unified_generator import unified_generator_node
from src.agent.nodes.extractor import math_extractor_node
from typing import List
from src.agent.state import AgentState
from src.agent.nodes.retrieval_auditor import retrieval_auditor_node


def route_after_auditor(state:AgentState):
    return state.get("target_node")


def route_after_planner(state: AgentState):
    if state.get("target_node")=="END":
        return "end"
    
    return "retriever"

def route_after_retriever_auditor(state: AgentState):
    # This is where the decision happens!
    feedback= state.get("retriever_feedback")
    attempts = state.get("retriever_audit_attempts", 0)
    needs_revision = feedback.needs_revision
    query_type = state["type"]

    if needs_revision:
        if attempts>=2:
            return "human_review"
        print(f"--- RETRIEVER REJECTED (Attempt {attempts}): LOOPING TO PLANNER ---")
        return "planner"
    else:
        if query_type == "B":
        # Skip Math! Go straight to the Generator
          return "generator" 
        
        
    return "extractor"
    



def route_after_human(state:AgentState):
    decision=state.get("human_decision")
    #status=state.get("audit_status")
    if decision=="pass":
        if state.get("is_cached") is True:
            return "end"
        return "cache_add"
    elif decision=="reject":
        return "generator"
    elif decision=="refinement":
        return "planner"
    elif decision=="investigate":
        return "guard"
    return "end"

#define the workflow
workflow=StateGraph(AgentState)

workflow.add_node("cache_check", semantic_cache_check_node)
workflow.add_node("guard",system_1_guard)
workflow.add_node("planner",planner_node)
workflow.add_node("retriever_auditor",retrieval_auditor_node)
workflow.add_node("extractor",math_extractor_node)
workflow.add_node("generator",unified_generator_node)
workflow.add_node("auditor",audit_engine)
workflow.add_node("retriever",hybrid_retriever_node)
workflow.add_node("python_repl",python_repl_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("cache_add",finalize_audit)
workflow.add_node("user_auditor",auditor_node)

# 2. Security Conditional Edge
workflow.add_conditional_edges(
    "guard",
    # We use a lambda to check the boolean and return the routing string
    lambda state: "secure" if state["is_safe"] else "unsafe",
    {
        "secure": "cache_check",
        "unsafe": END
    }
)

workflow.add_conditional_edges(
    "cache_check",
    lambda state: "hit" if state.get("is_cached") else "continue",
    {
        "hit": "human_review",
        "continue": "planner"
    }
)

workflow.set_entry_point("guard")
workflow.add_edge("retriever","retriever_auditor")
workflow.add_edge("extractor","python_repl")
workflow.add_edge("python_repl","generator")
workflow.add_edge("generator","user_auditor")
workflow.add_edge("user_auditor","auditor")
workflow.add_edge("cache_add", END)


# Conditional Edge 2: Auditor Feedback (Self-Correction Loop)
workflow.add_conditional_edges(
    "auditor",
    route_after_auditor,
    {    
        "generator": "generator",
        "human_review": "human_review"
    }
)

workflow.add_conditional_edges(
    "human_review",
    route_after_human,
    {    
        "generator": "generator",
        "cache_check": "guard", # NEW: Links 'investigate' to the cache node
        "cache_add": "cache_add",
        "end": END
    }
)

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {    
        "retriever": "retriever",
        "end": END
    }
)



# In your graph definition:
workflow.add_conditional_edges(
    "retriever_auditor", 
    route_after_retriever_auditor,
    {
        "generator": "generator",
        "extractor": "extractor",
        "planner":"planner",
        "human_review": "human_review"
    }
)

