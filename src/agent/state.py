from typing import List,TypedDict,Annotated,Optional,Dict,Literal
from pydantic import BaseModel, Field,ConfigDict
import operator
import operator
from typing import Annotated, TypedDict, Dict, Any, Union


def merge_node_metrics(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    return {**old, **new}

class Task(BaseModel):
    title: str = Field(description="A list of specific search queries or data extraction tasks needed to answer the user request.")
    doc_source: Literal["10K", "Transcript", "Both"] = Field(
        description="The target document source for this specific task."
    )
    zone: str = Field(description="The specific section to target (e.g., Item 8, Q&A, MD&A).")
    rationale: str = Field(description="Why this task is necessary for the audit.")
    extracted_company: str = Field(description="Company identifed for the task")
    extracted_year: int = Field(description="Year identifed for the task")
   
class SecurityRating(BaseModel):
    is_safe:bool=Field(description="True if the query is a legittimate financial audit question,False if it is an injection or jail break attempt")
    reason:str= Field(description="Brief reason for the safety rating.")

class EvidenceFound(BaseModel):
    task_name: str = Field(description="The specific task or metric from the Planner (e.g., '2019 COGS')")
    evidence: str = Field(description="The exact value found (e.g. $5,622M)")
    status: str = Field(description="FOUND, MISSING, or MISMATCHED")
    company:str= Field(description="Company")
    year:int= Field(description="Year")
    source: str = Field(description="Filename/Snippet ID of the source")
    page: str = Field(description="Page number")
    quote: str = Field(description="Verbatim 5-7 word quote for verification")

class Retriever_feedback(BaseModel):
    needs_revision:bool= Field(description="STRICT: Set to False if all raw numbers (e.g. COGS, Inventory) are present. Set to True ONLY if a document or year is missing.")
    retriever_critique:str= Field(description="Detailed explanation of hallucinations or missing info")
    found_evidence: List[EvidenceFound] = Field(description="List of all verified data points found in context")

class Reflection(BaseModel):
    decision:str=Field(description="Accept or reject the report")
    critique:str= Field(description="Detailed explanation of hallucinations or missing info")
    needs_revision:bool= Field(description="True if we need to go back to retrieval.")
    #target_node: Literal["generator","human"]=Field(description="Who needs to fix the error?Retriever for missing data,Generator for writing errors, Human for Red Flags.")
    hallucination_score: int = Field(
        ge=1, le=5, 
        description="Score 1-5: 5 if perfectly grounded, 1 if hallucinated. Pass threshold is 4."
    )
    divergence_score: int = Field(
        ge=1, le=5, 
        description="Score 1-5: 5 if conflicts handled, 1 if ignored. Default 5 for Type A/B."
    )
    traceability_score: int = Field(
        ge=1, le=5, 
        description="Score 1-5: 5 if all coords valid, 1 if missing/wrong. Pass threshold is 3."
    )
    math_score: int = Field(
        ge=1, le=5, 
        description="Score 1-5: 5 if math matches {math_info} perfectly, 1 if any error."
    )
    
    
    err_type: Optional[Literal["Math","Hallucination","Traceability","Incomplete","Divergence","Contradiction","Plan"]] = Field(default=None,
        description="Type of failure detected."
    )

    exact_trace: Optional[str] = Field(
        default=None,
        description="Exact offending text OR missing claim."
    )

    reason: str = Field(
        description="Why the report passed or failed."
    )

    expected: Optional[str] = Field(
        default=None,
        description="What should be corrected."
    )

    action: Optional[str] = Field(
        default=None,
        description="Instruction to generator/retriever."
    )

class Planner(BaseModel):
    tasks:List[Task]
    type:Literal["A","B","C"]=Field(description="A:Quant(10K),B:Qual(Both),C: Forensic (Both + Math)")
    reasoning:str =Field(description="Brief explanation of why these specific tasks were chosen")
    target_year:List[int]= Field(description="List of all year required for evidence")
    target_company:List[str] = Field(description="List of all company about which query is asked")

class Coordinate(BaseModel):
    # This sub-model also needs the strict config
    source: str = Field(description="The source document")
    # Use Union to catch cases where the LLM sends the page as a string
    page: Union[int, str] = Field(description="The page number")

class DivergenceAnalyst(BaseModel):
    alignment_status: str = Field(
    description="""
    Financial narrative alignment classification between earnings call transcripts and 10-K filings.

    Values:
    - ALIGNED
    - PARTIALLY_ALIGNED
    - DIVERGENT

    Indicates whether management statements are supported by reported disclosures.
    """
)
    #used_page_numbers: list[int] = Field(description="List of unique page numbers cited in the EVIDENCE TABLE")
    #used_evidence_texts: List[str] = Field(description="The list of raw evidence strings used in the report.Do not include instructions or metadata headers.")
    #used_coordinates:List[Coordinate] = Field(description="List of Company,year,Source and Page coordinates used")
    # --- THE NEW ARCHITECT FIELDS ---
    narrative_conflict_score: int=Field(
        ge=1, le=5, 
        description= "A qualitative measure (1-5) representing the degree of discrepancy between corporate narrative (Transcripts) and empirical reporting (10-K). High scores indicate a high-risk divergence requiring immediate human investigation."
    ) # Business Metric: Is the CEO lying? (1-5)
    divergence_type:str=Field(description="""
    For TYPE_C financial forensic analysis only.
    Classification of the detected divergence between management statements
    in earnings call transcripts and disclosures in 10-K filings.

    Allowed values:
    - Revenue
    - Guidance
    - Risk
    - CapEx
    - Narrative
    - None

    Return 'None' if no divergence is detected.
    """)
    divergence_reason:str=Field(decsription="""
    Short explanation of why the divergence was classified.

    Must summarize the conflict between transcript evidence and filing evidence.
    Include the key business issue being contradicted or misaligned.

    Examples:
    - Management described demand as accelerating while the filing disclosed declining orders.
    - Earnings call emphasized strong growth but the filing reported revenue contraction.
    - No material divergence detected; transcript narrative aligns with reported metrics.
    """)

    conflict_rationale:str  =Field(description="A concise justification explaining the forensic discrepancy. This field serves as the Proactive Summary for the human auditor, distilling why the system flagged the specific statement as contradictory.")


class FinalGeneration(BaseModel):
    report: str = Field(description="The full markdown audit report")

    
class FinancialMetric(BaseModel):
    label: str = Field(description="The name of the metric (e.g., Net_Income, Total_Assets).")
    value: float = Field(description="The numeric value extracted.")
    unit: str = Field(description="The unit (e.g., Millions, Billions, Absolute).")

class MathPlan(BaseModel):
    reasoning: str = Field(description="Why these specific numbers are needed for the formula.")
    metrics: List[FinancialMetric] = Field(description="List of extracted numbers from the context.")
    python_formula: str = Field(description="The Python code to execute (e.g., 'result = net_income / total_revenue').")

class AgentState(TypedDict):
    query:str
    type: str
    # The PATH of the investigation (Appends automatically)
    query_history: Annotated[list[str], operator.add] 
    report_history:Annotated[list[str],operator.add]

    # We use operator.merge so new facts are added to the old ones.
    audit_wiki: Annotated[list, operator.add]
    # The KNOWLEDGE BASE (Appends automatically)
    context_history: Annotated[list, operator.add]
    is_follow_up:str
    is_investigate:bool
    start_time:float
    is_cached:bool
    force_refresh: bool 
    plan: List[Task]
    context:List[Dict]
    math_plan: Optional[MathPlan]=None
    generation:str
    response:str
    is_safe:bool
    audit_status:str
    audit_attempts:int
    retriever_audit_attempts:int
    security_log:str
    reflection_feedback: Optional[Reflection]
    retriever_feedback:Optional[Retriever_feedback]
    ragas_scores:str
    alignment_status: Optional[str]
    narrative_conflict_score: Optional[int]
    divergence_type: Optional[str]
    divergence_reason: Optional[str]
    conflict_rationale: Optional[str]
    turn_count: int
    human_decision:str
    steps: Annotated[List[str], operator.add]
    target_company: str  # Must be here
    target_year: int
    final_context:List[Dict]
    calculation_result: float
    target_node:str
    total_cost: Annotated[float, operator.add]
    output_tokens: Annotated[int, operator.add]
    node_benchmarks: Annotated[dict, merge_node_metrics] # Set at the very beginning of the graph
    memory_usage: float # Captured at the end of the graph
    end_time_node:float