from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# ----------------------------
# Load Environment Variables
# ----------------------------

load_dotenv()


# ----------------------------
# LLM
# ----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True
)


# ----------------------------
# State
# ----------------------------

class ChatState(TypedDict):

    messages: Annotated[list, add_messages]


# ----------------------------
# Chatbot Node
# ----------------------------

def chatbot(state: ChatState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# ----------------------------
# Memory
# ----------------------------

conn = sqlite3.connect(
    "chat.db",
    check_same_thread=False
)

memory = SqliteSaver(conn)


# ----------------------------
# Build Graph
# ----------------------------

builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


graph = builder.compile(
    checkpointer=memory
)