from typing import TypedDict
from langchain_core.documents import Document
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, START, END
from scripts.bot_answer import *

class BotAnswerState(TypedDict):
    eval_data: pd.DataFrame
    outputs: list[dict]

def generate_outputs(state: BotAnswerState) -> BotAnswerState:
    outputs = generate_outputs(state["eval_data"])
    return {
        "outputs": outputs
    }

def add_outputs_to_df(state: BotAnswerState) -> BotAnswerState:
    df_eval = add_outputs_to_df(state["eval_data"], state["outputs"])
    return {
        "eval_data": df_eval
    }

bot_answer = (
    StateGraph(BotAnswerState)
    .add_node("generate_outputs", generate_outputs)
    .add_node("add_outputs_to_df", add_outputs_to_df)
    .add_edge(START, "generate_outputs")
    .add_edge("generate_outputs", "add_outputs_to_df")
    .add_edge("add_outputs_to_df", END)
    .compile()
)