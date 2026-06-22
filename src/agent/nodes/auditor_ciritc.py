import os
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Double check that the key loaded properly (optional safety check)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not found! Check your .env file.")


# 1. State Definition (Tracks the conversation history between agents)
class DebateState(BaseModel):
    job_description: str
    dialogue: Annotated[list, add_messages] = []
    iterations: int = 0
    final_report: str = ""

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# 2. Node: Lead Auditor (Extracts & Defends)
# 2. Node: Lead Auditor (Extracts & Defends)
def lead_auditor(state: DebateState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Lead Forensic Auditor. Your job is to analyze the provided Job Description and extract key technical constraints.\n"
            "CRITICAL CONTEXT:\n"
            "- The current year is 2026.\n"
            "- LangGraph and AutoGen were released in early 2024 (roughly 2 to 2.5 years ago).\n"
            "- Model Context Protocol (MCP) was released by Anthropic in late 2024.\n\n"
            "Evaluate experience timelines mathematically. If a JD asks for 3+ years of LangGraph/AutoGen/MCP, it is a chronological impossibility.\n"
            "If the Critic Agent challenges your findings, review their critique, defend your stance with hard logic if they are wrong, "
            "or adjust your finding if their logic holds true. Keep your responses concise and adversarial."
        )),
        ("placeholder", "{dialogue}"),
        ("user", "Initial Task: Audit this job description for technical risks:\n\n{job_description}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"job_description": state.job_description, "dialogue": state.dialogue})
    response.name = "Lead_Auditor"
    return {"dialogue": [response], "iterations": state.iterations + 1}

# 3. Node: Critic Agent (Cross-Examiner)
def critic_agent(state: DebateState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Elite Technical Critic. Your sole purpose is to pick apart the Lead Auditor's findings and cross-examine the original JD.\n"
            "CRITICAL ENGINE RULES:\n"
            "1. Current Temporal Anchor: The year is 2026. Calculate all tech tool lifetimes relative to this year.\n"
            "2. Tool Timeline Check: LangGraph (2024) = Max ~2.5 years old. AutoGen (late 2023/2024) = Max ~2.5 years old. MCP (late 2024) = Max 1.5 years old.\n"
            "3. Fact-Check the Auditor: If the Auditor claims a tool is a 'future release' or mathematically miscalculates the talent pool timeline based on 2026 reality, call them out immediately.\n"
            "4. Infrastructure & Architecture Auditing: Look for architectural friction. Look for scope inflation (e.g., expecting one engineer to build data pipelines, serve raw infra sandboxes, and build UI frameworks simultaneously) or architectural mismatching.\n\n"
            "Challenge the Auditor's points aggressively. If the Auditor successfully addresses all your critiques and provides a clean, accurate summary, "
            "explicitly state 'CRITIQUE_CONCLUDED' at the very end of your response."
        )),
        ("placeholder", "{dialogue}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"dialogue": state.dialogue})
    response.name = "Critic_Agent"
    return {"dialogue": [response]}

# 4. Conditional Router (Controls the Loop)
def route_debate(state: DebateState):
    # Stop condition: Critic is satisfied, or we hit a circuit breaker loop limit
    last_message = state.dialogue[-1].content if state.dialogue else ""
    if "CRITIQUE_CONCLUDED" in last_message or state.iterations >= 3:
        return "compile_report"
    return "lead_auditor"

# 5. Node: Final Report Compiler
def compile_report(state: DebateState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert technical forensic report compiler. Your job is to process the debate history "
            "and format the verified technical risks into an anonymous, high-density ASCII terminal screen output.\n\n"
            
            "STRICT CONTENT & TIMELINE RULES:\n"
            "1. ANONYMITY: Completely REMOVE all references to specific company names like 'Moring AI' or 'Moring'. "
            "Use '[STEALTH_AI_PLATFORM]' as the placeholder.\n"
            "2. TEMPORAL ACCURACY: Ensure all timeline statements reflect the current year 2026. "
            "(e.g., LangGraph/AutoGen = ~2.5 years old, MCP = ~1.5 years old). Do NOT output outdated or inaccurate temporal calculations.\n"
            "3. TONALITY: Condense the text so it reads aggressively sharp, engineering-focused, clean, and professional.\n\n"
            
            "STRICT LAYOUT & FORMATTING RULES:\n"
            "- Do not use generic markdown code blocks or conversational chat logs.\n"
            "- Use clean ASCII character borders (┌, ─, ┐, │, ├, ┤, └, ┘) to build a clear console window grid.\n"
            "- Set a strict maximum width of 80 characters per line so it is instantly scannable and doesn't wrap awkwardly when cropped for a LinkedIn slide image.\n"
            "- Structure the console layout exactly as follows:\n\n"
            
            "┌──────────────────────────────────────────────────────────────────────────────┐\n"
            "│ SYSTEM RUNTIME: FORENSIC AUDIT DISCOVERY ENGINE LOG                          │\n"
            "├──────────────────────────────────────────────────────────────────────────────┤\n"
            "│ TARGET: [STEALTH_AI_PLATFORM] JOB SPECIFICATION                             │\n"
            "│ TIMESTAMP: 2026-Q2 | STATUS: AUDIT_COMPLETE                                  │\n"
            "├──────────────────────────────────────────────────────────────────────────────┤\n"
            "│ IDENTIFIED TECHNICAL RISKS & ARCHITECTURAL ANOMALIES:                        │\n"
            "│                                                                              │\n"
            "│ 1. [CRITICAL] TIMELINE CONTRADICTION: [Core punchy description of age math]  │\n"
            "│ 2. [WARNING] SCOPE INFLATION: [Core description of architecture mismatch]    │\n"
            "│ ...                                                                          │\n"
            "└──────────────────────────────────────────────────────────────────────────────┘\n"
        )),
        ("placeholder", "{dialogue}"),
        ("user", "Compile the final report from the dialogue history now. Ensure complete anonymity and rigid ASCII layout execution.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"dialogue": state.dialogue})
    return {"final_report": response.content}

# 6. Construct the Cyclic Graph
workflow = StateGraph(DebateState)

workflow.add_node("lead_auditor", lead_auditor)
workflow.add_node("critic_agent", critic_agent)
workflow.add_node("compile_report", compile_report)

workflow.add_edge(START, "lead_auditor")
workflow.add_edge("lead_auditor", "critic_agent")
# The Loop: Critic either loops back to Auditor or sends to Compiler
workflow.add_conditional_edges("critic_agent", route_debate, {
    "lead_auditor": "lead_auditor",
    "compile_report": "compile_report"
})
workflow.add_edge("compile_report", END)

app = workflow.compile()


if __name__ == "__main__":
    sample_jd = """
    Position: AI Engineer - Agentic AI Platform (Moring AI)
Location: In-Office (Chennai, India / Atlanta, GA)
Salary: ₹15 – 25 LPA

Core Requirements Summary:
- Build multi-agent systems using LangGraph, AutoGen, multi-agent orchestration, state management, and memory.
- Contribute to the Moring Agentic AI Platform: the AI runtime, AI gateway, agent execution sandbox, and evaluation suite.
- Deploy and operate AI infrastructure on cloud platforms (AWS, Azure, GCP) using infrastructure-as-code, Kubernetes, and CI/CD pipelines.
- Build and maintain the agent observability stack: tracing, latency instrumentation, hallucination detection, and automated eval pipelines (LangSmith, DeepEval, Langfuse, Braintrust).
- Familiarity with Model Context Protocol (MCP) for standardized agent-to-tool connectivity.
    
    """

    print("🤖 STARTING AGENTIC CROSS-EXAMINATION LOOP...\n")
    
    # Run the graph and stream the actual argument live
    state = {"job_description": sample_jd}
    for output in app.stream(state, stream_mode="updates"):
        for node_name, node_state in output.items():
            if node_name in ["lead_auditor", "critic_agent"]:
                # Print the latest message from the active agent
                last_msg = node_state["dialogue"][-1]
                agent_title = "🕵️‍♂️ LEAD AUDITOR" if last_msg.name == "Lead_Auditor" else "⚡ CRITIC AGENT"
                print(f"{agent_title}:\n{last_msg.content}\n")
                print("-" * 60)

    # Fetch the compiled consensus report
    final_output = app.invoke(state)
    print("\n" + "="*50)
    print(" 📸 LINKEDIN TERMINAL SCREENSHOT TARGET")
    print("="*50)
    print(final_output["final_report"])
    print("="*50)