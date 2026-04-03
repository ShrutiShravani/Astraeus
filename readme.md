⚖️ Astraeus: Multi-Agent Forensic Audit & Divergence Engine

Astraeus is a high-precision orchestration platform designed to automate the forensic auditing of corporate financial reporting. By utilizing a Lead Auditor-Critic architecture, the system identifies factual inconsistencies between Official SEC 10-K Filings and Earnings Transcripts, ensuring management narratives align with audited financial reality.

System Architecture:
graph TD
    START[User Query] --> Cache{Semantic Cache}
    Cache -- Hit --> HITL[Human Review]
    Cache -- Miss --> Guard[PII & Security Guard]
    
    Guard --> Planner{Router Agent}
    
    Planner -- "Type A: Math" --> Ext[Math Extractor]
    Planner -- "Type B: Narrative" --> RetA[Hybrid Retriever]
    Planner -- "Type C: Forensic" --> RetC[Dual-Source Retriever]
    
    Extractor --> REPL[Python REPL]
    RetA & REPL & RetC --> Gen[Unified Generator]
    
    Gen --> Audit[Lead Auditor: Audit Engine]
    
    Audit -- "Hallucination/Error" --> Retriever/Generator
    Audit -- "Verified" --> Human_Verifier
    
    Human_Verifier --"PASS" -- Eval[MLflow & Ragas Metrics]
    Eval --> Checkpoint[Postgres Persistence]
    Eval --> END[Final Audit Report]

🏆 Audit Benchmarks (The "Golden Set")
Validated against a controlled 'Golden Set' of complex 10-K filings and transcripts (e.g., Nike, Apple, Tesla).

Metric,Value,Tech Source
End-to-End Latency,~120s (Targeting 40s),MLflow Tracking
Divergence Accuracy,94%,Lead Auditor Node
Faithfulness Score,0.78,RAGAS Evaluation
Answer Relevancy,0.87, RAGAS Evaluation
Audit Coverage,100%,Postgres Audit Trail

🌟 Engineering Highlights
Divergence Detection Engine (Type C)-
Astraeus doesn't just read documents; it looks for "The Gap."

Step 1: The Forensic Agent pulls the 'Outlook' section from the Transcript.

Step 2: It pulls the 'Inventory' and 'Cash Flow' tables from the 10-K.

Step 3: The Lead Auditor performs a cross-check. If the Transcript says "Skyrocketing Sales" but the 10-K shows "Rising Inventory/Falling Margins," Astraeus flags a Divergence Warning.

Forensic Guardrails: * PII Shielding: Uses a Presidio-based masking layer to ensure sensitive personnel data in private reports never reaches the LLM.

Semantic Cache Busting: Prevents year-over-year data leakage (e.g., 2019 vs 2020) by forcing fresh embeddings for specific audit years.

System Health Gatekeeper: A deterministic monitor that checks Host RAM (78% current usage) and ChromaDB Heartbeats. It blocks new audits if RAM exceeds 90% to prevent Docker OOM (Out of Memory) crashes.

State Persistence: Integrated PostgresSaver for checkpointing, allowing long-running forensic audits to persist even if the FastAPI server restarts.

Tech Stack & Infrastructure
Core: Python 3.11, FastAPI, Pydantic, LangGraph (MAS Orchestration)

Intelligence: GPT-4o, RAGAS (Evaluation), ChromaDB (Vector Search)

Infrastructure: Docker Compose, PostgreSQL (State), MLflow (Experiment Tracking)

Security: Presidio (PII Masking), Custom Hallucination Filters

⚠️ Critical Challenges Faced
Context Leakage: Fixed a critical issue where the retriever pulled cached 2019 evidence for 2020 queries by implementing strict State Clearing in the AgentState.

MLflow Serialization: Solved a TypeError where Ragas EvaluationResult objects (lists) failed to log to MLflow by implementing a Pandas-based mean-value extraction.

Postgres Async Mismatch: Resolved NotImplementedError in aget_state by standardizing on synchronous database drivers for local container reliability.

🚀 Currently Working On-
Active Optimization Sprint:

🛡️ 1. Advanced Guardrails & HITL FeedbackPrompt Injection & Bias Shield: Extending the system_1_guard to detect "jailbreak" attempts (DAN mode, role-play) and financial bias. You are ensuring that the auditor remains a neutral "third-party" observer, even if the user tries to lead the model toward a specific stock bias.Corrective HITL (Human-in-the-Loop): Upgrading the review node from a binary Pass/Reject to a Feedback Injection system. When an audit is "Rejected," the human's correction_note is injected back into the AgentState, forcing the Generator to treat it as a high-priority instruction for the next iteration.
⚡ 2. Latency Optimization (Target: <40s)Asynchronous Parallel Retrieval: Transitioning the forensic_flow (Type C) from sequential to parallel execution using asyncio.gather. This allows the system to fetch 10-K data and Earnings Transcripts simultaneously, cutting the current 112s bottleneck.Model Tiering (SLM vs LLM): Using GPT-4o-mini for low-compute tasks (Guardrails, Planning, Routing) and reserving the "Heavy" GPT-4o only for complex Forensic Synthesis and final Auditing to save time and compute.
💰 3. Token & Cost EfficiencySummarized Context Injection: Instead of passing raw, bulky document chunks to the Lead Auditor, the system now uses an Evidence Summarizer to create condensed Markdown tables, reducing the input token overhead by ~30%.Dynamic K-Value & Pruning: Implementing adaptive retrieval where Type A (Qualitative) queries pull fewer chunks ($K=5$) compared to Type C ($K=10$), and pruning the query_history after 5 turns to prevent "Prompt Bloat."
🧠 4. Hallucination Control & VersioningChain of Verification (CoVe): Implementing a "Grounding" prompt for the Auditor node that forces a two-step verification: first, extracting every figure, then cross-linking it to a specific coordinate in the source PDF.Prompt Versioning with MLflow: Using MLflow Tags to track the performance of different prompt versions (e.g., v1.2_strict_math). We compare RAGAS Faithfulness scores across versions to scientifically identify which prompt reduces hallucinations.
🔄 5. State-Aware Rollback StrategyIterative Rollbacks: If the auditor rejects the report more than twice, the system triggers a State Rewind. It reverts to the pre-generation state and attempts a new retrieval strategy (e.g., different keyword expansion) rather than repeating the same error.Pipeline Safety: Implementing a "Version Rollback" where the CD pipeline automatically reverts to the previous stable prompt version if the RAGAS "Golden Set" score drops by more than 10%.

6-Docker containerization and CI/CD 


🛠️ Getting Started
# LLM Intelligence
OPENAI_API_KEY="your_key"

# Observability & Agent Ops
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="Astraeus-Audit-Engine"
LANGCHAIN_API_KEY="your_key"

# Experiment Tracking
MLFLOW_TRACKING_URI="http://localhost:5000"
MLFLOW_EXPERIMENT_NAME="Nike_Dual_Source_Audit"

# Infrastructure: Postgres Checkpointing
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="postgres"
POSTGRES_DB="checkpoints"
# Use 'localhost' for local runs, 'db' for Docker network
DB_URI="postgresql://postgres:postgres@localhost:5432/checkpoints?sslmode=disable"

# Infrastructure: Vector DB (Chroma)
CHROMA_HOST="localhost"
CHROMA_PORT=8000

# Storage
DATA_DIR="data/input"

DATA_DIR ="data/"

2. How to Run
git clone https://github.com/yourusername/astraeus-audit.git


# Set Env
cp .env.example .env


# Start the Infrastructure (ChromaDB + Postgres + MLflow)
docker-compose up -d

# Install Dependencies
pip install -r requirements.txt

# Option A: Run the CLI Audit (Main Engine)
python main.py

# Option B: Launch the API & Auditor Dashboard
uvicorn app:app --reload
