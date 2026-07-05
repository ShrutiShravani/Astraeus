import pytest
from unittest.mock import patch,MagicMock,AsyncMock
from src.agent.nodes.retrieval_auditor import retrieval_auditor_node
from src.agent.state import EvidenceFound, Task,Retriever_feedback


mock_llm_output = MagicMock()
mock_llm_output.usage_metadata = {
    "input_tokens":100,
    "output_tokens":50,
    "total_tokens":200
}

mock_plan_test= [Task(title="Retrieve Nike revenue from 2020 financial report",
                doc_source="10K",
                zone="Item 8",
                rationale="Need revenue data for answer",
                extracted_company="Nike",
                extracted_year=2020,
)
]


@pytest.fixture
def base_state():
    return  {
    "query": "what was nike's revenue in 2020",
    "turn_count":0,
    "context": [
    {
        "page": "67",
        "source": "10k",
        "company": "Nike",
        "year": "2020",
        "evidence": "Nike's revenue reported 540 million for year 2020"
    }
],
    "audit_wiki":[],
    "evidence_found":[],
    "combined_critique":[],
    "final_context":[],
    "plan":mock_plan_test
    }

mock_planner_tasks= "Task 1 Nike's revenue for 2020 | Source: 10k" 

@patch("src.agent.nodes.retrieval_auditor.query_aware_compression", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieval_auditor.resilient_pro")
@patch("src.agent.nodes.retrieval_auditor.promptloader")
@patch("src.agent.nodes.retrieval_auditor.llm_pro")
@patch("src.agent.nodes.retrieval_auditor.log_to_mlflow")
@pytest.mark.asyncio
async def test_retriever_output(
    mock_log, 
    mock_llm_pro, 
    mock_prompt_loader, 
    mock_resilient, 
    mock_compress, 
    base_state
):
    # 1. Setup Mocks
    mock_compress.return_value = "Compressed evidence block"
    mock_prompt_loader.prompts.get.return_value = {
        "retriever_auditor_prompt": "Planner Tasks: {planner_tasks} Context: {context}",
        "version": "1.0.0"
    }
    
    # 2. Setup Structured Output Mock
    mock_auditor_output = Retriever_feedback(
        needs_revision=False,
        retriever_critique="All evidence found",
        found_evidence=[EvidenceFound(
            task_name="Retrieve Nike revenue", 
            status="found", 
            evidence="50B", 
            source="10k", 
            page="67", 
            company="Nike", 
            year="2020", 
            quote="..."
        )],
        no_evidence_found=False, 
        no_evidence_found_reason="None"
    )
    
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value={
        "parsed": mock_auditor_output,
        "raw": mock_llm_output
    })
    
    # This is the critical line: we are patching resilient_pro (the object used in your node)
    mock_resilient.with_structured_output.return_value = mock_structured

    # 3. Execution
    result = await retrieval_auditor_node(base_state)
    
    # 4. Assertions
    assert result["turn_count"] == 1
    # Check context from base_state
    assert base_state["context"][0]["evidence"] == "Nike's revenue reported 540 million for year 2020"
    # Check output from mock_auditor_output
    assert result["audit_wiki"][0].evidence == "50B"
    assert result['retriever_feedback'].retriever_critique == "All evidence found"