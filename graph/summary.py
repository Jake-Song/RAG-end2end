from typing import TypedDict
from langchain_core.documents import Document
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, START, END
from scripts.summary_eval import *

class SummaryState(TypedDict):
    eval_data: pd.DataFrame

def summary(state: SummaryState) -> SummaryState:
    summary = summary(state["eval_data"])
    return {
        "eval_data": summary
    }

