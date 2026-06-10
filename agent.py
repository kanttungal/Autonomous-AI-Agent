import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="meta-llama/llama-3-8b-instruct",
    api_key=OPENROUTER_API_KEY
)



search = TavilySearchResults(max_results=3)

# ---------------- STATE ----------------
class AgentState(TypedDict):
    query: str
    plan: str
    research: str
    final_answer: str
    use_search: bool

# ---------------- DECIDE NODE ----------------
def decide_node(state: AgentState):
    prompt = f"""
You are an AI decision engine.

User query: {state['query']}

Answer ONLY yes or no: do we need web search?
"""
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()

    return {"use_search": "YES" in decision}

# ---------------- PLANNER ----------------
def planner_node(state: AgentState):
    response = llm.invoke(f"Create step-by-step plan for: {state['query']}")
    return {"plan": response.content}

# ---------------- RESEARCH ----------------
def research_node(state: AgentState):
    results = search.invoke(state["query"])
    return {"research": str(results)}

# ---------------- DIRECT ANSWER ----------------
def direct_node(state: AgentState):
    response = llm.invoke(state["query"])
    return {"final_answer": response.content}

# ---------------- WRITER ----------------
def writer_node(state: AgentState):
    prompt = f"""
Question: {state['query']}
Plan: {state['plan']}
Research: {state.get('research','')}

Write a detailed final answer.
"""
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# ---------------- GRAPH ----------------
workflow = StateGraph(AgentState)

workflow.add_node("decide", decide_node)
workflow.add_node("plan", planner_node)
workflow.add_node("research", research_node)
workflow.add_node("direct", direct_node)
workflow.add_node("write", writer_node)

workflow.set_entry_point("decide")

# ---------------- ROUTING ----------------
def route(state: AgentState):
    if state["use_search"]:
        return "plan"
    else:
        return "direct"

workflow.add_conditional_edges("decide", route)

workflow.add_edge("plan", "research")
workflow.add_edge("research", "write")
workflow.add_edge("direct", END)
workflow.add_edge("write", END)

# ---------------- COMPILE APP ----------------
app = workflow.compile()
