import pytest
from unittest.mock import patch, MagicMock
from src.agent.nodes.planner import planner_node
from src.agent.state import Planner, Task

#MOCK AUDIT WIKI STRCUTURE
mock_raw_message = MagicMock()

usage_metadata={
        "input_tokens":100,
        "output_tokens":50,
        "total_tokens":200
    }

class MockWikiEntry:
    def __init__(self,year,company,task_name,evidence,source,page,quote):
        self.year=year
        self.company=company
        self.task_name=task_name
        self.evidence=evidence
        self.source=source
        self.page=page
        self.quote=quote


@pytest.fixture
def base_state():
    return {
        "query":"wht was Nike's revenue in 2023?",
        "turn_count":0,
        "query_history":[],
        "type":"A",
        "reasoning":"",
        "target_company":["Nike"],
        "target_year":["2023"],
        "is_follow_up":False,
        "audit_wiki":[]
    }

@patch("src.agent.nodes.planner.resilient_brain")
@patch("src.agent.nodes.planner.promptloader")
def test_planner_fresh_audit(mock_prompt_loader,mock_llm,base_state):
    mock_prompt_loader.prompts.get.return_value={
        "planner_prompt": "Instruction: {planner_instruction} Query: {current_query}",
        "version": "1.0.0"
    }

    #mock structured_output= Planner(
    mock_planner_obj=Planner(
        type="A",
        target_year=["2023"],
        target_company=["Nike"],
        reasoning="Need to find revenue line item",
        tasks=[Task(title="fetch_revenue",doc_source="10K",zone="Item 8",rationale="Required for query",extracted_company="Nike", extracted_year="2023")]
    )

    #configure the mock to return the parsed object in th e"include_raw' format
    mock_llm.with_structured_output.return_value.invoke.return_value={
        "parsed":mock_planner_obj,
        "raw":mock_raw_message
    }

    #x=exceute node
    result= planner_node(base_state)

    #assertions
    assert result["turn_count"]==1
    assert result["type"]=="A"
    assert "Nike" in result["target_company"]
    assert len(result["plan"])==1

@patch("src.agent.nodes.planner.llm_mini")
@patch("src.agent.nodes.planner.resilient_brain")

def test_planner_follow_up_purification(mock_llm_main,mock_llm_mini,mock_prefiltering,base_state):
    base_state["is_follow_up"]=True
    base_state["audit_wiki"]=[
        MockWikiEntry("2022", "Nike", "Revenue", "46B", "10-K", "5", "Revenue was 46B")
    ]
    base_state["query"] = "Compare it to 2023."

    #mock the purifier
    mock_llm_mini.invoke.return_value.content= "Compare Nike 2022 Revenue (46B) to REQUIRED_FROM_SOURCE 2023 Revenue."

    #mock main palnner response
    mock_planner_obj= Planner(
        type='A',
        target_year=["2023"],
        target_company=["Nike"],
        reasoning="Fetching 2023 to compare with verified 2022.",
        tasks=[Task(title="fetch_2023",doc_source="10K",zone="Item 8",rationale="Required for query",extracted_company="Nike", extracted_year="2023")]
    )

    mock_llm_main.with_structured_output.return_value.invoke.return_value= {"parsed": mock_planner_obj,"raw":mock_raw_message}

    # Execute
    result = planner_node(base_state)

    # Assert logic: Check if the "Purified" query is what we expected
    # In your code, you print(plan_output.tasks), we check if the plan came back
    assert result["target_year"] == [2023]
