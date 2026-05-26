import pytest
from unittest.mock import patch
from src.agent.nodes.python_repl import python_repl_node
from src.agent.state import MathPlan, FinancialMetric

@pytest.fixture
def base_state():
    math_plan=MathPlan(reasoning="Calculate operating margin",
        python_formula="result = revenue - expenses",
        metrics=[
            FinancialMetric(label="revenue",
    value=1000,
    unit="USD"),
            FinancialMetric(
            label="expenses",
            value=400,
            unit="USD"
        )
    ]
        
    )

    return {
        "turn_count":0,
        "math_plan":math_plan
    }

@patch("src.agent.nodes.python_repl.log_to_mlflow")

def test_python_repl_success(mock_log_mlflow, base_state):
    result = python_repl_node(base_state)


    assert result["calculation_result"] == 600

    assert result["turn_count"] == 1

    assert "python_repl" in result["node_benchmarks"]

    assert "Executed formula" in result["steps"][0]