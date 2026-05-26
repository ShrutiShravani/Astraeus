import pytest
from unittest.mock import patch, MagicMock

from src.agent.nodes.audit_engine import audit_engine
from src.agent.state import Reflection, Task


mock_raw_message = MagicMock()
mock_raw_message.usage_metadata = {
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150
}


mock_plan = [
    Task(
        title="Retrieve Nike revenue from 2020 financial report",
        doc_source="10K",
        zone="Item 8",
        rationale="Need revenue data",
        extracted_company="Nike",
        extracted_year=2020,
    )
]


@pytest.fixture
def base_state():
    return {
        "query": "What was Nike revenue in 2020?",
        "turn_count": 0,
        "type": "B",
        "audit_attempts": 0,
        "generation": "Nike revenue in 2020 was 50 billion.",
        "plan": mock_plan,
        "calculation_result": None,

        "context_history": [
            {
                "source": "10K",
                "page": "67",
                "evidence": "Nike revenue was 50 billion in 2020."
            }
        ]
    }


@patch("src.agent.nodes.audit_engine.log_to_mlflow")
@patch("src.agent.nodes.audit_engine.get_node_metrics")
@patch("src.agent.nodes.audit_engine.monitoring")
@patch("src.agent.nodes.audit_engine.promptloader")
@patch("src.agent.nodes.audit_engine.resilient_pro")

def test_audit_engine_pass(
    mock_resilient_pro,
    mock_promptloader,
    mock_monitoring,
    mock_get_metrics,
    mock_log_mlflow,
    base_state
):


    mock_promptloader.prompts.get.return_value = {
        "audit_engine_prompt": """
        Query: {current_query}
        Context: {context_str}
        Report: {generated_report}
        """,
        "version": "1.0.0"
    }

 
    mock_reflection = Reflection(
        needs_revision=False,
        decision="PASS",
        critique="All evidence properly grounded.",
        hallucination_score=5,
        math_score=5,
        traceability_score=5,
        divergence_score=4,
        reason="All checks passed.",
        err_type=None,
        expected=None,
        action=None,
        exact_trace=None
    )

  
    mock_structured_llm = MagicMock()

    mock_structured_llm.invoke.return_value = {
        "parsed": mock_reflection,
        "raw": mock_raw_message
    }

    mock_resilient_pro.with_structured_output.return_value = mock_structured_llm

    mock_metrics_fn = MagicMock(return_value={
        "latency": 0.45,
        "tokens": 150
    })

    mock_get_metrics.return_value = mock_metrics_fn

   
    result = audit_engine(base_state)


    assert result["turn_count"] == 1

    assert result["audit_status"] == "VERIFIED_BY_AUDITOR"

    assert result["target_node"] == "human_review"

    assert result["audit_attempts"] == 0

    assert result["reflection_feedback"].critique == "All evidence properly grounded."

    assert result["reflection_feedback"].hallucination_score == 5

    assert result["reflection_feedback"].math_score == 5

    assert result["critique"] == "All evidence properly grounded."