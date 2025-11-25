from typing import TypedDict
from langchain_core.documents import Document
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, START, END
from scripts.rerank import rerank_RRF

class RerankState(TypedDict):
    split_documents: list[Document]
    high_ranked: list[Document]

def rerank_RRF(state: RerankState) -> RerankState:
    high_ranked = rerank_RRF(state["split_documents"], cutoff=3)
    return {
        "high_ranked": high_ranked
    }

reranker = (
    StateGraph(RerankState)
    .add_node("rerank_RRF", rerank_RRF)
    .add_edge(START, "rerank_RRF")
    .add_edge("rerank_RRF", END)
    .compile()
)