from typing import TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from scripts.human_feedback import *

class HumanFeedbackState(TypedDict):
    eval_data: pd.DataFrame
    outputs: list[dict]

def human_feedback(state: HumanFeedbackState) -> HumanFeedbackState:
    return state

human_feedback = (
    StateGraph(HumanFeedbackState)
    .add_node("display_row", display_row)
    .add_node("get_feedback", get_feedback)
    .add_node("save_feedback", save_feedback)
    .add_edge(START, "display_row")
    .add_edge("display_row", "get_feedback")
    .add_edge("get_feedback", "save_feedback")
    .add_edge("save_feedback", END)
    .compile()
)