# 🧠 Autonomous AI Agent

An intelligent AI Agent built using **LangGraph**, **LangChain**, **OpenRouter LLMs**, **Tavily Search**, and **Streamlit**.

Unlike traditional chatbots, this agent can **autonomously decide** whether a user's query requires web research or can be answered directly using its reasoning capabilities.

---

## 🚀 Features

* 🤖 Autonomous decision making
* 🌐 Web search integration using Tavily
* 🧠 LLM-powered reasoning
* 🔀 Dynamic workflow routing with LangGraph
* 💬 ChatGPT-style Streamlit interface
* 📚 Research-backed responses
* ⚡ Real-time answer generation
* 🔍 Search only when necessary

---

## 🏗️ Architecture

User Query
↓
Decision Node
↓
┌─────────────┴─────────────┐
│                           │
Search Needed?         Direct Answer
│                           │
YES                         NO
│                           │
Planner Node                LLM
│                           │
Research Node               │
│                           │
Writer Node                 │
│                           │
└─────────────┬─────────────┘
↓
Final Answer

---

## 🛠️ Tech Stack

* Python
* LangGraph
* LangChain
* OpenRouter API
* Tavily Search API
* Streamlit
* LLMs (Llama, DeepSeek, Mistral)

---

## 📂 Project Structure

```bash
Autonomous-AI-Agent/
│
├── agent.py          # LangGraph workflow
├── ui.py             # Streamlit interface
├── requirements.txt
├── .env.example
├── README.md
└── assets/
```

