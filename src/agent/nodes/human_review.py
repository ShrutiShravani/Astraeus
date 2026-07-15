from src.agent.state import AgentState


def human_review_node(state: AgentState):
    """
    Breakpoint node. The graph stops here.
    The human will update 'human_decision' and 'query' via graph.update_state.
    """
    return state

