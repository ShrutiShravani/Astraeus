import pytest
from unittest.mock import patch, MagicMock
from src.agent.nodes.unified_generator import unified_generator_node
from src.agent.state import Task


mock_plan = [
    Task(
        title="Retrieve Nike revenue for 2020",
        doc_source="10K",
        zone="Item 8",
        rationale="Need revenue data",
        extracted_company="Nike",
        extracted_year=2020
    )
]


@pytest.fixture
def base_state():
    return {
        "query": "What was Nike revenue in 2020?",
        "turn_count": 0,
        "type": "financial_analysis",
        "generation": "",
        "calculation_result": None,
        "reflection_feedback": None,
        "is_follow_up": False,
        "context_history": [],
        "audit_wiki": [],
        "plan": mock_plan,

        "final_context": [
            {
                "id": "nike_67",
                "company": "Nike",
                "year": "2020",
                "source": "10K",
                "page": "67",
                "evidence": "Nike revenue was 50 billion in 2020"
            }
        ]
    }

@patch("src.agent.nodes.unified_generator.ChatPromptTemplate")
@patch("src.agent.nodes.unified_generator.get_node_metrics")
@patch("src.agent.nodes.unified_generator.log_to_mlflow")
@patch("src.agent.nodes.unified_generator.JsonOutputParser")
@patch("src.agent.nodes.unified_generator.promptloader")

def test_unified_generator_node(
    mock_promptloader,
    mock_parser,
    mock_log,
    mock_metrics,
    mock_chat_prompt,
    base_state
):


    mock_promptloader.prompts.get.return_value = {
        "unified_generator_prompt": """
        Planner: {planner_tasks}
        Context: {final_context_str}
        """,
        "version": "1.0.0"
    }

    mock_parser_instance = MagicMock()

    mock_parser.return_value = mock_parser_instance

    mock_parser_instance.get_format_instructions.return_value = "Return valid JSON"

    mock_parsed_output = {
        "report": """
        Nike revenue analysis.

        ### USED_COORDINATES
        {"source":"10K","page":"67"}

        Revenue found successfully.
        """,

        "narrative_conflict_score": 0
    }

    mock_parser_instance.parse.return_value = mock_parsed_output


    mock_chunk = MagicMock()
    mock_chunk.content = """
    {
      "report":"Nike revenue analysis. ### USED_COORDINATES {'source':'10K','page':'67'}",
      "narrative_conflict_score":0
    }
    """
    mock_chain= MagicMock()

    mock_chain.stream.return_value = [mock_chunk]

    mock_prompt_template = MagicMock()

    mock_prompt_template.__or__.return_value = mock_chain

    mock_chat_prompt.from_messages.return_value= mock_prompt_template

    mock_metrics.return_value = lambda state: {
        "latency": 0.2
    }


    result = unified_generator_node(base_state)


    assert result["turn_count"] == 1

    assert "generation" in result

    assert "Nike revenue analysis" in result["generation"]

    assert result["context_history"][0]["company"] == "Nike"

    assert result["context_history"][0]["page"] == "67"