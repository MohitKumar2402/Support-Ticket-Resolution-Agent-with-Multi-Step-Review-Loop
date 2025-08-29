# Support-Ticket-Resolution-Agent-with-Multi-Step-Review-Loop


An AI-powered support ticket resolution system built using LangGraph, designed to handle multi-step ticket resolution with classification, retrieval, drafting, review, and retry loops.

⚙ Overview

This project implements a Support Ticket Resolution Agent with a multi-step review loop as outlined in the assessment task
.

The agent:

Accepts a ticket (subject + description).

Classifies it into Billing, Technical, Security, or General.

Retrieves relevant context using FAISS + embeddings (RAG).

Generates a draft response.

Passes the draft through a review step.

If the draft fails review, retries up to 2 times with refined context.

If still failing, escalates to CSV log for human review.

🏗 Architecture
🔹 Nodes in the Graph

Input Node → Receives ticket data.

Classification Node → Uses LLM to categorize tickets.

Retrieval Node → Fetches relevant context snippets from vector store.

Draft Node → Generates polite, customer-facing draft replies.

Review Node → Automated reviewer ensures professionalism, policy compliance, and clarity.

Retry/Escalation Logic → If review fails, retries with reviewer feedback. After 2 failed attempts, escalates to CSV.

🔹 Knowledge Base

Mocked documents for Billing, Technical, Security, General.

Stored in a FAISS vector store with HuggingFace embeddings.

🔹 Logging

escalation_log.csv → Stores escalated tickets with drafts + reviewer feedback.

Also logs errors for debugging.

🚀 Setup Instructions
1️⃣ Clone Repository
git clone <your_repo_url>
cd <your_repo_name>

2️⃣ Create & Activate Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Environment Variables

Create a .env file:

AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
AZURE_OPENAI_API_VERSION=2024-05-01-preview

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

5️⃣ Run LangGraph Dev Server
langgraph dev --config langgraph.json

🧪 Usage

Start LangGraph dev server.

Provide a support ticket (subject + description).

The graph executes automatically:

Classifies → Retrieves → Drafts → Reviews → Retries/Escalates.

Check escalation_log.csv for failed tickets.

📂 Project Structure
.
├── main.py                       # Core workflow code
├── requirements.txt              # Dependencies
├── .env                          # Environment variables
├── escalation_log.csv            # Escalation log (generated at runtime)
├── Assessment Task.pdf           # Task description
└── README.md                     # Documentation

💡 Design Decisions

LangGraph Orchestration → Enables modular, graph-based agent execution.

FAISS Vector Store + HuggingFace Embeddings → Simple, efficient RAG implementation.

Azure OpenAI LLM → Used for classification, draft generation, and review.

Retry Loop (Max 2) → Matches real-world support workflows with bounded attempts.

Escalation CSV → Provides human fallback for unresolved tickets.

📊 Example Flow

Input Ticket:

Subject: "Refund not received"
Description: "I requested a refund last week but haven’t seen it processed yet."


System Flow:

Classifies as Billing.

Retrieves refund-related context.

Drafts polite reply.

Reviewer approves → final response delivered.

📤 Deliverables

✅ Codebase (this repo)

✅ README.md (setup + design rationale)

✅ Escalation log (CSV sample included after failed runs)

🎥 Demo video (to be recorded showing happy path, retry, escalation)

🧠 Evaluation Criteria (Met)

✅ Multi-step flow with retry loop

✅ Modular, reusable nodes

✅ Clear logging & error handling

✅ Prompt engineering for classification, drafting, reviewing

✅ Escalation handling
