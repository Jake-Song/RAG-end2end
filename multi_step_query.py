"""
Multi-Step Query Transformations 구현
reference link: 
https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations#multi-step-query-transformations

"""
from typing import TypedDict, Annotated, Literal
from langchain_upstage import ChatUpstage
from langchain_core.documents import Document
from langgraph.graph import START,END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from operator import add

from utils.utils import format_context
from rag import get_ensemble_retriever
from reranker.rrf import ReciprocalRankFusion


class MultiState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[list[str], add]
    query: Annotated[list[str], add]
    documents: Annotated[list[Document], "Documents"]
    query_count: Annotated[int, "Query Count"]
    answer: Annotated[str, "Answer"]

def init_state(state: MultiState) -> MultiState:
    
    return {"context": [], "query": [], "query_count": 0}

# 문서 검색 노드: 사용자 질문에 관련된 문서를 검색하는 노드
def retrieve_document(state: MultiState) -> MultiState:
    query_count = state.get("query_count", 0)
    # print(query_count)
    if query_count == 0:
        latest_question = state["question"]
    else:
        latest_question = state["query"][-1]
    # 앙상블 리트리버를 사용하여 질문과 관련성 높은 문서 검색
    # BM25(키워드 기반)와 FAISS(의미 기반)를 결합하여 검색 성능 향상
    bm25_retriever, faiss_retriever = get_ensemble_retriever()
    retrieved_docs_faiss = faiss_retriever.invoke(latest_question)
    retrieved_docs_bm25 = bm25_retriever.invoke(latest_question)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25

    return {"documents": retrieved_docs, "query_count": query_count}

def rerank_document(state: MultiState) -> MultiState:
    retrieved_docs = state["documents"]
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": [context]}

DECOMPOSE_QUERY_TRANSFORM = (
    "The original question is as follows: {query_str}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below. \n"
    "Given the context, return a new question that can be answered from "
    "the context. The question can be the same as the original question, "
    "or a new question that represents a subcomponent of the overall question.\n"
    "As an example: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: What is the current population of the city in which Paul Graham found "
    "his first company, Viaweb?\n"
    "Knowledge source context: Provides information about Paul Graham's "
    "professional career, including the startups he's founded. "
    "New question: In which city did Paul Graham found his first company, Viaweb? "
    "\n\n"
    "Question: {query_str}\n"
    "Knowledge source context: {context_str}\n"
    "New question: "
)
 
decomposer = ChatUpstage(model="solar-pro2", temperature=0)

def decompose_query(state: MultiState) -> MultiState:
   
    question = state["question"]
    context = state["context"]
    prompt = DECOMPOSE_QUERY_TRANSFORM.format(query_str=question, context_str=context)
    query_count = state.get("query_count", 0) + 1
    decomposed_query = decomposer.invoke(prompt)
   
    return {"query": [decomposed_query.content], "query_count": query_count}


AGGREGATE_ANSWER = """
You are a helpful assistant that synthesizes answers from multiple queries.

You are given a list of queries and their corresponding contexts.

You need to synthesize a final answer from the queries.

Here are the queries and contexts:

{completed_queries}
{completed_contexts}
"""

aggregator = ChatUpstage(model="solar-pro2", temperature=0)

def aggregate_answer(state: MultiState) -> MultiState:
    queries = state["query"]
    contexts = state["context"]
    completed_queries = "\n\n---\n\n".join(queries)
    completed_contexts = "\n\n---\n\n".join(contexts)
    prompt = AGGREGATE_ANSWER.format(completed_queries=completed_queries, completed_contexts=completed_contexts)
    answer = aggregator.invoke(prompt)
    return {"answer": answer.content}

def recursive_query_router(state: MultiState) -> Literal["retrieve_document", "aggregate_answer"]:
    query_count = state["query_count"]
    if query_count <= 2:
        return "retrieve_document"
    else:
        return "aggregate_answer"

builder = StateGraph(MultiState)
builder.add_node("init_state", init_state)
builder.add_node("retrieve_document", retrieve_document)
builder.add_node("rerank_document", rerank_document)
builder.add_node("decompose_query", decompose_query)
builder.add_node("aggregate_answer", aggregate_answer)

builder.add_edge(START, "init_state")
builder.add_edge("init_state", "retrieve_document")
builder.add_edge("retrieve_document", "rerank_document")
builder.add_edge("rerank_document", "decompose_query")
builder.add_conditional_edges(
    "decompose_query", 
    recursive_query_router,
    {
        "retrieve_document": "retrieve_document",
        "aggregate_answer": "aggregate_answer"
    }
)
builder.add_edge("aggregate_answer", END)

multi = builder.compile()

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
  
    for chunk in multi.stream({"question": "AI 트렌드는 무엇인가?"}, stream_mode="updates", config=config):
        for node, value in chunk.items():
            if "query_count" in value.keys():
                print(f"{node}: {value['query_count']}")
                print("-"*100)
            if "answer" in value.keys():
                print(f"{node}: {value['answer']}")
                print("-"*100)

 