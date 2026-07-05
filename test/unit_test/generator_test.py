import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from src.agent.nodes.unified_generator import unified_generator_node
from src.agent.state import EvidenceFound # Ensure this import is correct

@pytest.fixture
def base_state():
    return {
        "query": "What was Nike revenue in 2020?",
        "turn_count": 0,
        "type": "financial_analysis",
        "audit_wiki": [
            EvidenceFound(
                task_name="Revenue Check",
                evidence="$50B",
                status="FOUND",
                company="Nike",
                year=2020,
                source="10K",
                page="67",
                quote="Nike revenue was 50 billion"
            )
        ],
        "plan": [],
        "wiki_archive": ""
    }

@patch("src.agent.nodes.unified_generator.promptloader")
@patch("src.agent.nodes.unified_generator.JsonOutputParser")
@patch("src.agent.nodes.unified_generator.get_node_metrics")
@patch("src.agent.nodes.unified_generator.log_to_mlflow")
def test_unified_generator_node(mock_log, mock_metrics, mock_parser, mock_loader, base_state):
    # 1. Mock Prompt Loader
    mock_loader.prompts.get.return_value = {
        "unified_generator_prompt": "{final_context_str}",
        "version": "1.0.0"
    }

    # 2. Mock Parser
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse.return_value = {"report": "Nike revenue analysis result"}
    mock_parser.return_value = mock_parser_instance

    # 3. Mock Metrics
    mock_metrics.return_value = lambda state: {"latency": 0.1, "tokens": 50}

    # 4. Patching the Chain (The critical part)
    # We mock the chain.stream so it returns an AIMessage that supports +=
    with patch("src.agent.nodes.unified_generator.ChatPromptTemplate"), \
         patch("src.agent.nodes.unified_generator.resilient_pro") as mock_resilient:
        
        mock_chain = MagicMock()
        mock_chain.stream.return_value = [AIMessage(content='{"report": "Nike revenue analysis result"}')]
        mock_resilient.__or__.return_value = mock_chain
        
        # Execute
        result = unified_generator_node(base_state)

        # Assertions
        assert result["generation"] == "Nike revenue analysis result"
        assert result["turn_count"] == 1
        assert "latency" in result