import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]


os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    max_tokens=700,
    temperature=0.2,
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

def safe_llm_call(prompt: str, retries: int = 2):

    for attempt in range(retries + 1):

        try:

            response = llm.invoke(prompt)

            return response.content

        except Exception as e:

            error_message = str(e).lower()

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if (
                "rate limit" in error_message
                or "429" in error_message
                or "too many requests" in error_message
                or "ratelimit" in error_message
            ):

                if attempt < retries:

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                return (
                    "⚠️ OpenRouter rate limit reached.\n\n"
                    "The free model has reached its current "
                    "usage limit. Please try again later."
                )

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            return (
                "⚠️ AI service error.\n\n"
                "Please try again later."
            )

    return "⚠️ AI service temporarily unavailable."


# ============================================================
# SEARCH DECISION
# ============================================================

def should_use_search(query: str) -> bool:

    """
    Rule-based search detection.

    This replaces the old decide_node LLM call.

    Therefore every user query does NOT require an extra
    LLM request just to decide whether search is needed.
    """

    query_lower = query.lower()

    search_keywords = [

        # Current information
        "latest",
        "recent",
        "today",
        "current",
        "now",
        "this week",
        "this month",

        # News
        "news",
        "headline",
        "breaking",

        # Research
        "research",
        "study",
        "paper",

        # Search
        "search",
        "look up",
        "find online",
        "on the internet",

        # Price / comparison
        "price",
        "pricing",
        "cost",
        "compare",

        # Travel / local
        "weather",
        "restaurant",
        "hotel",
        "travel",

        # Product / company
        "company",
        "product",
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

    return {
        "use_search": use_search
    }


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state: AgentState):

    prompt = f"""
Create a short step-by-step research plan for:

{state['query']}

Keep the plan concise.
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

        results = search.invoke(
            state["query"]
        )

        return {
            "research": str(results)
        }

    except Exception as e:

        return {
            "research": (
                "Web search failed. "
                "Answer using the available information."
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

User question:
{state['query']}

Keep the answer concise but useful.
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

Answer the user's question using the research
information provided below.

USER QUESTION:
{state['query']}

RESEARCH PLAN:
{state.get('plan', '')}

WEB RESEARCH:
{state.get('research', '')}

Instructions:

1. Answer the actual user question.
2. Use the research information when relevant.
3. Do not invent facts.
4. If the research is insufficient, say so.
5. Keep the answer clear and useful.
"""

    answer = safe_llm_call(prompt)

    return {
        "final_answer": answer
    }


# ============================================================
# GRAPH
# ============================================================

workflow = StateGraph(AgentState)


# Nodes

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

