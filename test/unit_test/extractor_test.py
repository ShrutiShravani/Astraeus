import pytest
from unittest.mock import patch, MagicMock

from src.agent.nodes.extractor import math_extractor_node
from src.agent.state import MathPlan



@pytest.fixture
def base_state():
    return {
        "query": "what was revenue in 2020",
        "turn_count": 0,
        "context": [
            {
                "source": "10K",
                "evidence": "Revenue was 540 million in 2020 page 67"
            }
        ]
    }


# -----------------------------
# TEST
# -----------------------------
@patch("src.agent.nodes.extractor.resilient_brain")
@patch("src.agent.nodes.extractor.promptloader")
@patch("src.agent.nodes.extractor.get_node_metrics")
@patch("src.agent.nodes.extractor.log_to_mlflow")

def test_math_extractor_node(
    mock_log_mlflow,
    mock_metrics,
    mock_promptloader,
    mock_resilient,
    base_state
):
  
    mock_promptloader.prompts.get.return_value = {
        "extractor_prompt": "Query: {query} Context: {context_text}",
        "version": "1.0.0"
    }

    mock_plan = MathPlan(
    metrics=[],
    reasoning="No computation needed for test case",
    python_formula="result = revenue"
)

    mock_llm = MagicMock()

    mock_llm.invoke.return_value = {
        "parsed": mock_plan,
        "raw": MagicMock(usage_metadata={"tokens": 10})
    }

    mock_resilient.with_structured_output.return_value = mock_llm

   
    mock_metrics.return_value = lambda state: {
        "dummy_metric": 1
    }

   
    result = math_extractor_node(base_state)


    assert result["turn_count"] == 1
    assert "math_context" in result
    assert result["math_plan"] == mock_plan
    assert result["math_context"] == "Revenue was 540 million in 2020 page 67"