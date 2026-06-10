import streamlit as st
from agent import app

st.set_page_config(page_title="Autonomous AI Agent")

st.title("🧠 Autonomous AI Research Agent")



if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("Ask anything...")

if query:
    result = app.invoke({
        "query": query,
        "plan": "",
        "research": "",
        "final_answer": "",
        "use_search": False
    })

    answer = result["final_answer"]

    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("ai", answer))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)
