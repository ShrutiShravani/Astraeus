from src.agent.state import AgentState

async def user_node(state: AgentState):
    # This node does nothing but wait for the user to update the state
    return state