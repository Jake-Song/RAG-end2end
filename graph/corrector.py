from typing import TypedDict
from langchain_core.documents import Document
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, START, END
from scripts.correct_eval import *

class CorrectorState(TypedDict):
    eval_data: pd.DataFrame

def correctness(state: CorrectorState) -> CorrectorState:
    for i, row in state["eval_data"].iterrows():
        inputs = {"query": row["query"]}
        outputs = {"answer": row["outputs.answer"]}
        reference_outputs = {"answer": row["answer"]}
        correctness = correctness_evaluator.correctness(inputs, outputs, reference_outputs)
        state["eval_data"].loc[i, 'correctness'] = correctness["correctness"]
        state["eval_data"].loc[i, 'explanation'] = correctness["explanation"]

    state["eval_data"].to_csv(f"{output_path_prefix}_eval_correct.csv", index=True)
    return {
        "eval_data": state["eval_data"]
    }

corrector = (
    StateGraph(CorrectorState)
    .add_node("correctness", correctness)
    .add_edge(START, "correctness")
    .add_edge("correctness", END)
    .compile()
)