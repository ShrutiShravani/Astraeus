from collections import defaultdict
import json
import datetime
from src.agent.state import AgentState


def audit_encoder(obj):
    if isinstance(obj,(datetime.datetime,datetime.date)):
        return obj.isoformat() # Converts to "2026-04-02T..."
    if hasattr(obj, 'dict'): # Handles Pydantic models automatically
        return obj.dict()
    if hasattr(obj, '__dict__'): # Handles generic classes
        return obj.__dict__
    return str(obj)

def auditor_node(state:AgentState):
    steps_taken = state.get("steps", [])
    audit_status=state.get("audit_status")
    
    # 2. Inspect for Divergence (Type C logic)
    math_val = state.get("calculation_result")
    benchmarks = state.get("node_benchmarks", {})
    audit_wiki=state.get("audit_wiki","")


    total_cost=sum(v.get('cost',0)for v in benchmarks.values())
    total_latency= sum(v.get('latency',0)for v in benchmarks.values())
    
    # 3. Construct a Lean Audit Entry
    audit_entry = {
        "audit_id": f"AUDIT-{datetime.datetime.now().strftime('%Y%H%M%S')}",
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "run_stats":{
            "total_cost":round(total_cost,2),
            "total_latency":round(total_latency,2)
        },
        "metadata": {
            "company": state.get("target_company"),
            "year": state.get("target_year"),
            "query_type": state.get("type")
        },
        "query": state.get("query"),
        "planner_tasks": [getattr(t,'title',str(t)) for t in state.get("plan", [])],
        "forensic_evidence": {
            "math_verified": math_val,
            "sources_used":[getattr(t, 'doc_source', 'N/A') for t in state.get("plan", [])]
        
        },
        "agent_logic_trace": steps_taken, # This shows the node-by-node history
        "status":audit_status,
        "performance_breakdown": benchmarks ,
        "final_evidence":audit_wiki,
        
        "final_report_preview": state.get("generation", "")[:2000] + "..." # Keep logs lean
    }
   
    # 4. Save to Disk (JSONL is best for 4-Waybill analysis)
    with open("audit_trail.jsonl","a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry, ensure_ascii=False,default=audit_encoder) + "\n")
    
    print(f"Compliance Audit Logged for {state.get('target_company')}")
    
    return {"steps": ["Audit trail entry created for compliance"]}