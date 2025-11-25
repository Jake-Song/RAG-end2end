from typing import TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from scripts.parse import *

class ParseState(TypedDict):
    input_file: str
    data: list[dict]
    docs: list[Document]
    markdown: str

def get_json_arr(state: ParseState) -> ParseState:
    json_data_arr = get_json_arr(state["input_file"])
    return {
        "data": json_data_arr
    }

def flatten_json(state: ParseState) -> ParseState:
    flattened = flatten_json(state["data"])
    return {
        "data": flattened
    }

def create_docs(state: ParseState) -> ParseState:
    docs = create_docs(state["data"])
    return {
        "docs": docs
    }

def extract_images(state: ParseState) -> ParseState:
    docs = extract_images(state["docs"])
    return {
        "docs": docs
    }

def merge_docs(state: ParseState) -> ParseState:
    merged = merge_docs(state["docs"])
    return {
        "docs": merged
    }

def save_docs(state: ParseState) -> ParseState:
    cleaned = remove_metadata(state["docs"])
    saved = save_docs(cleaned)
    return {
        "docs": saved
    }

def save_markdown(state: ParseState) -> ParseState:
    markdown = save_markdown(state["docs"])
    return {
        "markdown": markdown
    }

parser = (
    StateGraph(ParseState)
    .add_node("get_json_arr", get_json_arr)
    .add_node("flatten_json", flatten_json)
    .add_node("create_docs", create_docs)
    .add_node("extract_images", extract_images)
    .add_node("merge_docs", merge_docs)
    .add_node("save_docs", save_docs)
    .add_node("save_markdown", save_markdown)
    .add_edge(START, "get_json_arr")
    .add_edge("get_json_arr", "flatten_json")
    .add_edge("flatten_json", "create_docs")
    .add_edge("create_docs", "extract_images")
    .add_edge("extract_images", "merge_docs")
    .add_edge("merge_docs", "save_docs")
    .add_edge("save_docs", "save_markdown")
    .add_edge("save_markdown", END)
    .compile()
)

if __name__ == "__main__":
    parser.get_graph().draw_mermaid_png()
    result = parser.invoke({"input_file": input_file})
    print(result)

