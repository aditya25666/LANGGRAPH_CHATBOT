import streamlit as st

from graph import graph
from langchain_core.messages import HumanMessage


# ------------------------------------
# Page Configuration
# ------------------------------------

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph Chatbot")


# ------------------------------------
# LangGraph Thread
# ------------------------------------

config = {
    "configurable": {
        "thread_id": "chat-1"
    }
}


# ------------------------------------
# Session State
# ------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------
# Display Previous Messages
# ------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------------------------------------
# User Input
# ------------------------------------

user_input = st.chat_input("Type your message...")


if user_input:

    # Display User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------------------------
    # Stream Assistant Response
    # ------------------------------------

    response = ""

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        for message_chunk, metadata in graph.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
            stream_mode="messages"
        ):

            if message_chunk.content:
                response += message_chunk.content
                response_placeholder.markdown(response + "▌")

        response_placeholder.markdown(response)

    # ------------------------------------
    # Save Assistant Response
    # ------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )