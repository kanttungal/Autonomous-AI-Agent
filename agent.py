# agent.py

import os
import time
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(override = True)


# ============================================================
# OPENROUTER API KEY
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    try:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        OPENROUTER_API_KEY = None


# ============================================================
# TAVILY API KEY
# ============================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    try:
        TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    except Exception:
        TAVILY_API_KEY = None


if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# ============================================================
# CHECK API KEY
# ============================================================

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Add it to .env or .streamlit/secrets.toml"
    )


if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY not found. "
        "Add it to .env or .streamlit/secrets.toml"
    )


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openrouter/free",
    temperature=0.2,
    max_tokens=700,
)


# ============================================================
# TAVILY SEARCH
# ============================================================

search = TavilySearchResults(
    max_results=3
)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    query: str
    plan: str
    research: str
    final_answer: str
    use_search: bool


# ============================================================
# SAFE LLM CALL
# ============================================================

def safe_llm_call(prompt: str, retries: int = 0):

    for attempt in range(retries + 1):

        try:

            response = llm.invoke(prompt)

            return response.content

        except Exception as e:

            print("\n========== LLM ERROR ==========")
            print("Error Type:", type(e).__name__)
            print("Error:", str(e))
            print("================================\n")

            error_message = str(e).lower()

            # -----------------------------------------------
            # RATE LIMIT
            # -----------------------------------------------

            if (
                "rate limit" in error_message
                or "429" in error_message
                or "too many requests" in error_message
                or "ratelimit" in error_message
            ):

                if attempt < retries:

                    wait_time = 2 ** attempt

                    print(
                        f"Rate limit detected. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                return (
                    "⚠️ OpenRouter rate limit reached.\n\n"
                    "The selected free model has reached "
                    "its current usage limit.\n\n"
                    "Please try again later."
                )

            # -----------------------------------------------
            # OTHER ERROR
            # -----------------------------------------------

            return (
                f"❌ LLM Error\n\n"
                f"Error Type: {type(e).__name__}\n\n"
                f"Details: {str(e)}"
            )

    return "❌ AI service temporarily unavailable."


# ============================================================
# SEARCH DECISION
# ============================================================

def should_use_search(query: str) -> bool:

    query_lower = query.lower()

    search_keywords = [

        # Current information
        "latest",
        "recent",
        "today",
        "current",
        "right now",
        "this week",
        "this month",
        "this year",

        # News
        "news",
        "headline",
        "breaking",

        # Research
        "research",
        "study",
        "paper",

        # Explicit web search
        "search",
        "look up",
        "find online",
        "on the internet",

        # Price
        "price",
        "pricing",

        # Travel / local
        "weather",
        "restaurant",
        "hotel",
        "travel",

        # Technology
        "release",
        "version",
        "documentation",
    ]

    return any(
        keyword in query_lower
        for keyword in search_keywords
    )


# ============================================================
# ROUTER NODE
# ============================================================

def router_node(state: AgentState):

    use_search = should_use_search(
        state["query"]
    )

    print(
        f"\nRouter decision: "
        f"{'WEB SEARCH' if use_search else 'DIRECT ANSWER'}"
    )

    return {
        "use_search": use_search
    }


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state: AgentState):

    prompt = f"""
You are a research planning assistant.

Create a short step-by-step research plan for:

{state["query"]}

Keep the plan concise.
Only include important research steps.
"""

    plan = safe_llm_call(prompt)

    return {
        "plan": plan
    }


# ============================================================
# RESEARCH NODE
# ============================================================

def research_node(state: AgentState):

    try:

        print("\nRunning Tavily web search...")

        results = search.invoke(
            state["query"]
        )

        return {
            "research": str(results)
        }

    except Exception as e:

        print("\n========== TAVILY ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("==================================\n")

        return {
            "research": (
                "Web search failed.\n"
                f"Error: {str(e)}"
            )
        }


# ============================================================
# DIRECT ANSWER NODE
# ============================================================

def direct_node(state: AgentState):

    prompt = f"""
You are a helpful AI assistant.

Answer the following user question clearly
and accurately.

USER QUESTION:
{state["query"]}

Instructions:

1. Directly answer the question.
2. Keep the answer concise but useful.
3. Give an example when helpful.
"""

    answer = safe_llm_call(prompt)

    return {
        "final_answer": answer
    }


# ============================================================
# WRITER NODE
# ============================================================

def writer_node(state: AgentState):

    prompt = f"""
You are an AI research assistant.

Answer the user's question using the web research
provided below.

USER QUESTION:
{state["query"]}

RESEARCH PLAN:
{state.get("plan", "")}

WEB RESEARCH:
{state.get("research", "")}

Instructions:

1. Answer the user's actual question.
2. Use the research information when relevant.
3. Do not invent facts.
4. Do not make unsupported claims.
5. If the research is insufficient, clearly say so.
6. Keep the final answer clear and useful.
"""

    answer = safe_llm_call(prompt)

    return {
        "final_answer": answer
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(AgentState)


# ============================================================
# NODES
# ============================================================

workflow.add_node(
    "router",
    router_node
)

workflow.add_node(
    "plan",
    planner_node
)

workflow.add_node(
    "research",
    research_node
)

workflow.add_node(
    "direct",
    direct_node
)

workflow.add_node(
    "write",
    writer_node
)


# ============================================================
# ENTRY POINT
# ============================================================

workflow.set_entry_point(
    "router"
)


# ============================================================
# ROUTING
# ============================================================

def route(state: AgentState):

    if state["use_search"]:
        return "plan"

    return "direct"


workflow.add_conditional_edges(
    "router",
    route
)


# ============================================================
# RESEARCH FLOW
# ============================================================

workflow.add_edge(
    "plan",
    "research"
)

workflow.add_edge(
    "research",
    "write"
)


# ============================================================
# DIRECT FLOW
# ============================================================

workflow.add_edge(
    "direct",
    END
)


# ============================================================
# RESEARCH FLOW END
# ============================================================

workflow.add_edge(
    "write",
    END
)


# ============================================================
# COMPILE
# ============================================================

app = workflow.compile()