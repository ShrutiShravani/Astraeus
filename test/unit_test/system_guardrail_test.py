#external dependencies:llm/callback/resilient brain/prompt loader
import pytest
from unittest.mock import patch, MagicMock
from src.agent.nodes.system_guradrail import system_1_guard
from src.agent.state import SecurityRating


mock_raw_message =MagicMock()

usage_metadata= {
    "input_tokens":100,
    "output_tokens":50,
    "total_tokens":200
}


@pytest.fixture
def base_state():
    return {
        "query": "wht was Nike's revenue in 2023?",
        "current_turn": 0
    }


@patch("src.agent.nodes.system_guradrail.resilient_brain")
@patch("src.agent.nodes.system_guradrail.promptloader")
@patch("src.agent.nodes.system_guradrail.llm_mini")

def test_system_guardrail(mock_llm,mock_prompt_loader,mock_resilient_brain,base_state):
    mock_prompt_loader.prompts.get.return_value={
        "system_guardrail_prompt":"{user_input}",
        "version": "1.0.0"
    }
   
    mock_system_guardrail_obj=SecurityRating(
        is_safe="True",
        reason="the document is safe"
    )
   
    mock_resilient_brain.with_structured_output.return_value.invoke.return_value={
        "parsed":mock_system_guardrail_obj,
        "raw":mock_raw_message
    }

    result= system_1_guard(base_state)

    assert result["is_safe"]==True
    assert result["security_log"]== "the document is safe"