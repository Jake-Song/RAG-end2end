"""
Multi Document Queries Agent
reference link:https://developers.llamaindex.ai/python/framework/understanding/putting_it_all_together/q_and_a#multi-document-queries
"""

from langchain_core.documents import Document
from langgraph.graph import add_messages
from langchain.tools import tool
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import AnyMessage
from langchain.messages import ToolMessage
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langgraph.graph import StateGraph, START, END
from utils.utils import format_context
from reranker.rrf import ReciprocalRankFusion
from config import output_path_prefix
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path().resolve()
sys.path.insert(0, str(project_root))

def load_retriever(split_documents, embeddings, file_name, search_k=1):
    vectorstore = FAISS.load_local(
        f"{project_root}/faiss_index", 
        embeddings,
        file_name,
        allow_dangerous_deserialization=True  # needed in newer versions
    )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = search_k
    
    return bm25_retriever, faiss_retriever


@tool
def retriever_2022(query: str) -> list[Document]:
    """Retrieve documents from the vector database.
    retrieve documents of AI 2022 index.
    Args:
        query: The query to retrieve documents from the vector database.
    """
    doc_path = Path(project_root).joinpath("outputs", "SPRI_2022_output_split_documents.pkl")
    file_name = "SPRI_2022"
    with open(doc_path, "rb") as f:
        split_documents = pickle.load(f)
    embeddings = UpstageEmbeddings(model="embedding-passage")
    bm25_retriever, faiss_retriever = load_retriever(split_documents, embeddings, file_name, search_k=10)
    retrieved_docs_faiss = faiss_retriever.invoke(query)
    retrieved_docs_bm25 = bm25_retriever.invoke(query)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}


@tool
def retriever_2023(query: str) -> list[Document]:
    """Retrieve documents from the vector database.
    retrieve documents of AI 2023 index.
    Args:
        query: The query to retrieve documents from the vector database.
    """
    doc_path = Path(project_root).joinpath("outputs", "SPRI_2023_output_split_documents.pkl")
    file_name = "SPRI_2023"
    with open(doc_path, "rb") as f:
        split_documents = pickle.load(f)
    embeddings = UpstageEmbeddings(model="embedding-passage")
    bm25_retriever, faiss_retriever = load_retriever(split_documents, embeddings, file_name, search_k=10)
    retrieved_docs_faiss = faiss_retriever.invoke(query)
    retrieved_docs_bm25 = bm25_retriever.invoke(query)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}


@tool
def retriever_2025(query: str) -> list[Document]:
    """Retrieve documents from the vector database.
    retrieve documents of AI 2025 index.
    Args:
        query: The query to retrieve documents from the vector database.
    """
    doc_path = Path(project_root).joinpath("outputs", "SPRI_2025_output_split_documents.pkl")
    file_name = "SPRI_2025"
    with open(doc_path, "rb") as f:
        split_documents = pickle.load(f)
    embeddings = UpstageEmbeddings(model="embedding-passage")
    bm25_retriever, faiss_retriever = load_retriever(split_documents, embeddings, file_name, search_k=10)
    retrieved_docs_faiss = faiss_retriever.invoke(query)
    retrieved_docs_bm25 = bm25_retriever.invoke(query)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}


tools = [retriever_2022, retriever_2023, retriever_2025]
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

response_model = ChatUpstage(model="solar-pro2", temperature=0)

ROUTER_PROMPT = (
    """
    You are an assistant for question-answering tasks. 
    You will be given a question and a list of tools to use. 
    You will need to decide which tool to use to answer the question. 
    You will need to return the tool name and the arguments to use. 
    Here is the messages: {messages}    
    """
)

def generate_query_or_respond(state: GraphState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    prompt = ROUTER_PROMPT.format(messages=state["messages"])
    response = (
        response_model
        .bind_tools([retriever_2022, retriever_2023, retriever_2025]).invoke(prompt)  
    )
    return {"messages": [response]}

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: GraphState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

def should_continue(state: GraphState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

workflow = StateGraph(GraphState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", tool_node)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    should_continue,
    {
        # Translate the condition outputs to nodes in our graph
        "tool_node": "retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile
graph = workflow.compile()

if __name__ == "__main__":
    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "책임있는 AI에 관해 알려줘",
                }
            ]
        },
        
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")





