import pytest
from unittest.mock import patch,MagicMock
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

@patch("src.agent.nodes.retrieval_auditor.resilient_pro")
@patch("src.agent.nodes.retrieval_auditor.promptloader")
@patch("src.agent.nodes.retrieval_auditor.llm_pro")
@patch("src.agent.nodes.retriever.log_to_mlflow")

def test_retriever_output(mock_log_to_mlflow,mock_llm_pro,mock_prompt_loader,mock_llm,base_state):
    
    mock_llm_pro.invoke.return_value.content="Nike's revenue was 240 billion for year 2020"
   
    mock_prompt_loader.prompts.get.return_value={
        "retriever_auditor_prompt": """ Planner Tasks: {planner_tasks} Context: {context}""",
        "version": "1.0.0"
}

   
    mock_auditor_output= Retriever_feedback(
        needs_revision = "False",
        retriever_critique="All evdience found",
        found_evidence=[EvidenceFound(
             task_name="What is nike's revenue in 2020?",
             evidence="Nike's revenue is 50 billion as of 2020",
             status= "Found",
             company="Nike",
             year="2020",
             source="10k",
             page="67",
             quote="Nike revenue report end of financial year for 2020 as 50 billion"
    )
    ]
    )

    mock_structured = MagicMock()
    mock_structured.ainvoke.return_value = {
        "parsed": mock_auditor_output,
        "raw": mock_llm_output
    }

    mock_llm.with_structured_output.return_value = mock_structured

    result= retrieval_auditor_node(base_state)
    
    assert result["turn_count"]==1
    assert result["context"][0]["evidence"]=="Nike's revenue reported 540 million for year 2020"
    
    assert result["audit_wiki"][0].evidence =="Nike's revenue is 50 billion as of 2020"
    assert result['retriever_feedback'].retriever_critique =="All evdience found"