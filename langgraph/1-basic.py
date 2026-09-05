import random
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    graph_state: str


def node_1(state):
    print("Node 1")
    return {"graph_state": state["graph_state"] + "! Good morning"}


def node_2(state):
    print("Node 2")
    return {"graph_state": state["graph_state"] + "! Good evening"}


def node_3(state):
    print("Node 3")
    return {"graph_state": state["graph_state"] + "! Good night"}


def decide_mood(state) -> Literal["Node 2", "Node 3"]:

    current_state = state["graph_state"]

    if random.random() < 0.5:
        return "Node 2"

    return "Node 3"


builder = StateGraph(State)
builder.add_node("Node 1", node_1)
builder.add_node("Node 2", node_2)
builder.add_node("Node 3", node_3)

builder.add_edge(START, "Node 1")
builder.add_conditional_edges("Node 1", decide_mood)
builder.add_edge("Node 2", END)
builder.add_edge("Node 3", END)

graph = builder.compile()

result = graph.invoke({"graph_state": "Leo Das"})
print(result)
