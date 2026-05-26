import pytest 
from unittest.mock import patch, MagicMock
from src.agent.nodes.retriever import hybrid_retriever_node
from src.agent.state import AgentState,Task

mock_plan_task= [Task(title="Retrieve Nike revenue from 2020 financial report",
                doc_source="10K",
                zone="Item 8",
                rationale="Need revenue data for answer",
                extracted_company="Nike",
                extracted_year=2020,
)
]


@pytest.fixture
def base_state():
    return{
       "query":"what was nike's revenue in 2020",
       "plan": mock_plan_task,
       "current_turn": 0,
       'target_company':"Nike",
       "target_year":"2020"
    }

@patch("src.agent.nodes.retriever.pre_filtering")
@patch("src.agent.nodes.retriever.client")
@patch("src.agent.nodes.retriever.ranker")
@patch("src.agent.nodes.retriever.sparse_encoder")
@patch("src.agent.nodes.retriever.dense_encoder")
@patch("src.agent.nodes.retriever.log_to_mlflow")

def test_retriever_node(mock_log_to_mlflow,mock_dense_encoder,mock_sparse_encoder,mock_client,mock_ranker,mock_prefiltering,base_state):

    mock_dense_vector = MagicMock()

    mock_dense_vector.tolist.return_value = [0.1, 0.2, 0.3]

    mock_dense_encoder.embed.return_value = [mock_dense_vector]

    mock_sparse_result = MagicMock()

    mock_sparse_result.indices.tolist.return_value = [1, 2, 3]
    mock_sparse_result.values.tolist.return_value = [0.5, 0.8, 0.9]

    mock_sparse_encoder.embed.return_value = [mock_sparse_result]
    

    mock_point = MagicMock()

    mock_point = MagicMock()

    mock_point.id = "point-1"

    mock_point.payload = {
        "page_content": "Nike revenue was 217 million in 2020",
        "metadata": {
            "company": "Nike",
            "year": 2020,
            "source": "NIKE_10K_2020.pdf",
            "doc_type": "10K",
            "page": 12,
        },
    }

    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]

    mock_client.query_points.return_value = mock_search_result

    # -------------------------------------------------------
    # MOCK RERANKER
    # retriever expects list of chunks
    # -------------------------------------------------------

    mock_ranker.rerank.return_value = [
        {
            "company": "Nike",
            "year": 2020,
            "source": "NIKE_10K_2020.pdf",
            "page": 12,
            "text": "Nike revenue was 217 million in 2020",
        }
    ]

    # -------------------------------------------------------
    # MOCK PRE FILTERING
    # final retriever output uses this
    # -------------------------------------------------------

    mock_prefiltering.return_value = [
        {
            "company": "Nike",
            "year": 2020,
            "source": "NIKE_10K_2020.pdf",
            "page": 12,
            "evidence": "Nike revenue was 217 million in 2020",
        }
    ]

    
    result = hybrid_retriever_node(base_state)
    

    assert result["turn_count"] == 1

    assert len(result["context"]) == 1

    assert (
        result["context"][0]["evidence"]
        == "Nike revenue was 217 million in 2020"
    )
