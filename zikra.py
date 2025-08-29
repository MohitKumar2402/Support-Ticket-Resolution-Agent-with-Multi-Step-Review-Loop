from typing import Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver   # ✅ checkpointer
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.docstore.document import Document
from langchain_openai import AzureChatOpenAI

import os
import json
import csv
from dotenv import load_dotenv

# ================== ENV SETUP ==================
load_dotenv()  # ✅ Load variables from .env file

# ================== LLM ==================
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# ================== MOCK KNOWLEDGE BASE ==================
documents = [
    # Billing
    Document(page_content="Refund requests usually take up to a week to process.", metadata={"category": "Billing"}),
    Document(page_content="Track your payment status anytime via your account dashboard.", metadata={"category": "Billing"}),
    Document(page_content="Invoices and billing details can be accessed directly from your profile.", metadata={"category": "Billing"}),

    # Technical
    Document(page_content="If you get a 403 error, try clearing your browser cache or resetting your password.", metadata={"category": "Technical"}),
    Document(page_content="Having login issues? Double-check your credentials or contact tech support for assistance.", metadata={"category": "Technical"}),
    Document(page_content="Encountering unexpected errors? Restart the app or ensure your software is updated.", metadata={"category": "Technical"}),

    # Security
    Document(page_content="Security alert: Change your password immediately to keep your account safe.", metadata={"category": "Security"}),
    Document(page_content="Enable multi-step verification to strengthen account protection.", metadata={"category": "Security"}),
    Document(page_content="Review your account activity regularly to spot any suspicious behavior.", metadata={"category": "Security"}),

    # General
    Document(page_content="For general inquiries, contact our support team at support@example.com.", metadata={"category": "General"}),
    Document(page_content="Most answers to common questions are available in the FAQ section on our website.", metadata={"category": "General"}),
    Document(page_content="You can also reach out via live chat for assistance with general issues.", metadata={"category": "General"}),
]


# ================== EMBEDDINGS + FAISS ==================
embedding_function = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L12-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

vector_store = FAISS.from_texts(
    texts=[doc.page_content for doc in documents],
    embedding=embedding_function,
    metadatas=[doc.metadata for doc in documents]
)

print("✅ Vector store created successfully")

# ================== STATE ==================
class TicketState(TypedDict):
    ticket: Dict[str, str]  
    category: str
    context: List[str]
    draft: str
    review_feedback: str
    retry_count: int
    final_response: Optional[str]

# ================== CLASSIFICATION ==================
classify_prompt = PromptTemplate(
    input_variables=["subject", "description"],
    template="""You are a customer support classifier. Classify the following support ticket into one of: Billing, Technical, Security, General.

Return JSON only:
{{"category": "<Billing|Technical|Security|General>"}}

Ticket Subject: {subject}
Ticket Description: {description}"""
)

def classify_ticket(ticket: Dict[str, str]) -> str:
    try:
        prompt = classify_prompt.format(
            subject=ticket["subject"], description=ticket["description"]
        )
        response = llm.invoke(
            [
                SystemMessage(content="You are a ticket classification assistant."),
                HumanMessage(content=prompt)
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.content)
        category = data.get("category", "General")
        return category if category in {"Billing", "Technical", "Security", "General"} else "General"
    except Exception as e:
        print(f"Classification error: {e}")
        log_error(ticket, "Classification failed")
        return "General"

# ================== RETRIEVAL ==================
def retrieve_context(category: str, ticket: Dict[str, str], feedback: str = "") -> List[str]:
    try:
        query = f"{ticket['subject']} {ticket['description']}"
        if feedback:
            query += f" {feedback}"  
        docs = vector_store.similarity_search(query=query, k=2, filter={"category": category})
        return [doc.page_content for doc in docs] or ["No relevant context found."]
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        log_error(ticket, "Retrieval failed")
        return ["Retrieval failed."]

# ================== DRAFT GENERATION ==================
draft_prompt = PromptTemplate(
    input_variables=["subject", "description", "context", "feedback"],
    template="""Generate a polite, empathetic, and professional response to the support ticket. 
Use provided context and reviewer feedback if any. 
If retrying, ensure the reply adds warmth and includes clear contact options (phone/email/live chat).

Ticket Subject: {subject}
Ticket Description: {description}
Context: {context}
Reviewer Feedback: {feedback}

Response:"""
)

def generate_draft(ticket: Dict[str, str], context: List[str], feedback: str = "") -> str:
    try:
        context_str = "\n".join(context) if context else "No context available."
        prompt = draft_prompt.format(
            subject=ticket["subject"], 
            description=ticket["description"], 
            context=context_str,
            feedback=feedback
        )
        response = llm.invoke([
            SystemMessage(content="You are a customer support assistant. Respond in plain text."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        print(f"Draft error: {str(e)}")
        log_error(ticket, "Draft generation failed")
        return "Failed to generate draft."

# ================== REVIEW ==================
review_prompt = PromptTemplate(
    input_variables=["draft", "subject", "description"],
    template="""Review the draft reply for clarity, professionalism, and compliance. 
Return JSON only:
{{"pass": true/false, "feedback": "<improvement notes>"}} 

Ticket Subject: {subject}
Ticket Description: {description}
Draft: {draft}"""
)

def review_draft(draft: str, ticket: Dict[str, str]) -> Dict[str, str]:
    try:
        prompt = review_prompt.format(
            subject=ticket["subject"], description=ticket["description"], draft=draft
        )
        response = llm.invoke([
            SystemMessage("You are a draft reviewer. Respond only with JSON."),
            HumanMessage(content=prompt)
        ])
        data = json.loads(response.content)
        if "pass" in data and "feedback" in data:
            return data
        else:
            raise ValueError("Invalid JSON keys")
    except Exception as e:
        print(f"Review error: {str(e)}")
        log_error(ticket, "Review failed")
        return {"pass": False, "feedback": "Invalid review response, escalate."}

# ================== ESCALATION LOGGING ==================
def log_escalation(state: TicketState):
    with open("escalation_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            state["ticket"]["subject"],
            state["ticket"]["description"],
            state["category"],
            state["draft"],
            state["review_feedback"]
        ])

def log_error(ticket: Dict[str, str], error: str):
    with open("escalation_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ticket.get("subject", ""), ticket.get("description", ""), "Error", "", error
        ])

# ================== GRAPH NODES ==================
def input_node(state: TicketState) -> TicketState:
    return state

def classify_node(state: TicketState) -> TicketState:
    state["category"] = classify_ticket(state["ticket"])
    return state

def retrieve_node(state: TicketState) -> TicketState:
    state["context"] = retrieve_context(
        state["category"], state["ticket"], state.get("review_feedback", "")
    )
    return state

def draft_node(state: TicketState) -> TicketState:
    state["draft"] = generate_draft(
        state["ticket"], state["context"], state.get("review_feedback", "")
    )
    return state

def review_node(state: TicketState) -> TicketState:
    review = review_draft(state["draft"], state["ticket"])
    state["review_feedback"] = review["feedback"]
    if review["pass"]:
        state["final_response"] = state["draft"]
    return state

def retry_or_escalate(state: TicketState) -> str:
    if state.get("final_response"):
        return "end"
    if state["retry_count"] >= 2:
        log_escalation(state)
        return "end"
    state["retry_count"] += 1
    return "retrieve"

# ================== BUILD GRAPH ==================
workflow = StateGraph(TicketState)
workflow.add_node("input", input_node)
workflow.add_node("classify", classify_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("draft", draft_node)
workflow.add_node("review", review_node)

workflow.set_entry_point("input")
workflow.add_edge("input", "classify")
workflow.add_edge("classify", "retrieve")
workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "review")
workflow.add_conditional_edges(
    "review",
    retry_or_escalate,
    {"retrieve": "retrieve", "end": END}
)

# ================== ENTRYPOINT FOR LANGGRAPH DEV ==================
def build_workflow():
    """Return compiled graph for LangGraph CLI Studio"""
    memory = SqliteSaver.from_conn_string(":memory:")
    return workflow.compile(checkpointer=memory)
