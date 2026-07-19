import pytest 
from unittest.mock import patch, MagicMock
from src.agent.nodes.retriever import hybrid_retriever_node
from src.agent.state import Task



@pytest.fixture
def base_state():
    return {
        "query": "What was Nike revenue in 2020?",
        "turn_count": 0,
        "target_company": "Nike",
        "target_year": 2020,
        "plan": [
            Task(
                title="Retrieve Nike revenue from 2020 financial report",
                doc_source="10K",
                zone="Item 8",
                rationale="Need revenue",
                extracted_company="Nike",
                extracted_year=2020,
            )
        ],
    }


@patch("src.agent.nodes.retriever.log_to_mlflow")
@patch("src.agent.nodes.retriever.ranker")
@patch("src.agent.nodes.retriever.client")
@patch("src.agent.nodes.retriever.sparse_encoder")
@patch("src.agent.nodes.retriever.dense_encoder")
def test_hybrid_retriever_success(
    mock_dense,
    mock_sparse,
    mock_client,
    mock_ranker,
    mock_mlflow,
    base_state,
):

    # ---------- Dense ----------
    dense_vec = MagicMock()
    dense_vec.tolist.return_value = [0.1, 0.2]
    mock_dense.embed.return_value = [dense_vec]

    # ---------- Sparse ----------
    sparse = MagicMock()
    sparse.indices.tolist.return_value = [1, 2]
    sparse.values.tolist.return_value = [0.5, 0.8]
    mock_sparse.embed.return_value = [sparse]

    # ---------- Qdrant ----------
    point = MagicMock()
    point.id = "1"
    point.payload = {
        "page_content": "Nike revenue was $37.4B.",
        "metadata": {
            "company": "Nike",
            "year": 2020,
            "source": "NIKE_10K_2020.pdf",
            "doc_type": "10K",
            "page": 15,
        },
    }

    search_result = MagicMock()
    search_result.points = [point]

    mock_client.query_points.return_value = search_result

    # ---------- FlashRank ----------
    mock_ranker.rerank.return_value = [
        {
            "company": "Nike",
            "year": 2020,
            "source": "NIKE_10K_2020.pdf",
            "page": 15,
            "text": "Nike revenue was $37.4B.",
        }
    ]

    result = hybrid_retriever_node(base_state)

    assert result["turn_count"] == 1
    assert len(result["context"]) == 1

    assert result["context"][0]["company"] == "Nike"
    assert result["context"][0]["year"] == 2020
    assert result["context"][0]["source"] == "NIKE_10K_2020.pdf"
    assert result["context"][0]["page"] == 15
    assert result["context"][0]["evidence"] == "Nike revenue was $37.4B."

    mock_client.query_points.assert_called_once()
    mock_ranker.rerank.assert_called_once()
    mock_mlflow.assert_called_once()