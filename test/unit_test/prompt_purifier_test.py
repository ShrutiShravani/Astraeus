import pytest
from unittest.mock import MagicMock, patch
from src.agent.nodes.purifier_node import prompt_purifier_node # Adjust import path

# Define a dummy state for testing
@pytest.fixture
def mock_state():
    return {
        "query": "Is the revenue growth sustainable?",
        "query_history": ["What is the revenue?"],
        "audit_wiki": [],
        "turn_count": 0
    }

@patch("src.agent.nodes.purifier_node.resilient_brain")
@patch("src.agent.nodes.purifier_node.promptloader")
@patch("src.agent.nodes.purifier_node.log_to_mlflow")
def test_prompt_purifier_node_success(mock_log, mock_loader, mock_llm, mock_state):
    # 1. Setup mocks
    mock_loader.prompts.get.return_value = {
        'query_purifier_prompt': "History: {history}, Query: {current_query}, Facts: {already_verified_facts}",
        'version': '1.0.0'
    }
    
    # Mock the structured output return
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": MagicMock(action="rewrite", rewritten_query="Refined query about growth"),
        "raw": MagicMock()
    }

    # 2. Call the function
    result = prompt_purifier_node(mock_state)

    # 3. Assertions
    assert result["query"] == "Refined query about growth"
    assert result["turn_count"] == 1
    assert "Rewriting" in result["steps"][0]

@patch("src.agent.nodes.purifier_node.resilient_brain")
@patch("src.agent.nodes.purifier_node.promptloader")
def test_prompt_purifier_node_clarification(mock_loader, mock_llm, mock_state):
    # Simulate the LLM asking for clarification
    mock_loader.prompts.get.return_value = {
        'query_purifier_prompt': "...",
        'version': '1.0.0'
    }
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": MagicMock(action="clarify", clarification_question="Which company?"),
        "raw":MagicMock()
    }

    result = prompt_purifier_node(mock_state)

    assert result["ask_user"] is True
    assert result["clarification_question"] == "Which company?"