from typing import TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from scripts.synthetic_data import *

llm = ChatUpstage(model="solar-pro2", temperature=0.0, reasoning_effort="high")

class GenDataState(TypedDict):
    docs: list[Document]
    synthetic_data: list[SyntheticData]
    queries: list[list[dict]]
    pairs: list[tuple[int, int]]

def generate_prompt(state: GenDataState) -> GenDataState:
    queries, pairs = generate_prompt(state["docs"], query_count=50)
    return {
        "queries": queries,
        "pairs": pairs
    }

def generate_data(state: GenDataState) -> GenDataState:
    responses = generate_data(llm, state["queries"])
    return {
        "synthetic_data": responses
    }

def save_data(state: GenDataState) -> GenDataState:
    save_data(state["synthetic_data"], state["pairs"])
    return {
        "synthetic_data": state["synthetic_data"]
    }

data_generater = (
    StateGraph(GenDataState)
    .add_node("generate_prompt", generate_prompt)
    .add_node("generate_data", generate_data)
    .add_node("save_data", save_data)
    .add_edge(START, "generate_prompt")
    .add_edge("generate_prompt", "generate_data")
    .add_edge("generate_data", "save_data")
    .add_edge("save_data", END)
    .compile()
)