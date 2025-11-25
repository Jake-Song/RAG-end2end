from typing import TypedDict
from langchain_core.documents import Document
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, START, END
from scripts.chunk import *

class ChunkState(TypedDict):
    docs: list[Document]
    split_documents: list[Document]
    messages_for_image: list[dict]
    messages_for_text: list[dict]

llm = ChatUpstage(model="solar-pro2")

def prepare_image_summary(state: ChunkState) -> ChunkState:
    messages_for_image = prepare_image_summary(state["docs"])
    return {
        "messages_for_image": messages_for_image
    }

def prepare_text_summary(state: ChunkState) -> ChunkState:
    messages_for_text = prepare_text_summary(state["docs"])
    return {
        "messages_for_text": messages_for_text
    }

def summarize_image(state: ChunkState) -> ChunkState:
    docs = summarize_image(state["docs"], state["messages_for_image"], llm)
    return {
        "docs": docs
    }

def summarize_text(state: ChunkState) -> ChunkState:
    docs = summarize_text(state["docs"], state["messages_for_text"], llm)
    return {
        "docs": docs
    }

def split_docs(state: ChunkState) -> ChunkState:
    split_documents = split_docs(state["docs"])
    return {
        "split_documents": split_documents
    }

def save_text(state: ChunkState) -> ChunkState:
    save_text(state["split_documents"])
    return {
        "split_documents": state["split_documents"]
    }

def save_split_document(state: ChunkState) -> ChunkState:
    save_split_document(state["split_documents"])
    return {
        "split_documents": state["split_documents"]
    }

chunker = (
    StateGraph(ChunkState)
    .add_node("prepare_image_summary", prepare_image_summary)
    .add_node("prepare_text_summary", prepare_text_summary)
    .add_node("summarize_image", summarize_image)
    .add_node("summarize_text", summarize_text)
    .add_node("split_docs", split_docs)
    .add_node("save_text", save_text)
    .add_node("save_split_document", save_split_document)
    .add_edge(START, "prepare_image_summary")
    .add_edge("prepare_image_summary", "prepare_text_summary")
    .add_edge("prepare_text_summary", "summarize_image")
    .add_edge("summarize_image", "summarize_text")
    .add_edge("summarize_text", "split_docs")
    .add_edge("split_docs", "save_text")
    .add_edge("save_text", "save_split_document")
    .add_edge("save_split_document", END)
    .compile()
)

chunker.get_graph().draw_mermaid_png()