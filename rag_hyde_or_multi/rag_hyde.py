"""
HyDE Query Transformation 구현
reference link: 
https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations#hyde-hypothetical-document-embeddings
"""
from typing import TypedDict, Annotated, Literal
from langchain_upstage import ChatUpstage
from langchain_core.documents import Document
from langgraph.graph import START,END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from utils.utils import format_context
from rag_basic.rag import get_ensemble_retriever
from reranker.rrf import ReciprocalRankFusion
from langchain.messages import AnyMessage
from langgraph.graph import add_messages

class HyDEState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: Annotated[str, "Question"]
    context: Annotated[list[str], "Context"]
    passage: Annotated[str, "Passage"]
    documents: Annotated[list[Document], "Documents"]
    is_retrieved: Annotated[bool, "Whether it is retrieved"]
    is_hyde: Annotated[bool, "Whether it is hyde-transformed"]


HYDE_PROMPT = (
    """
    Please write a passage to answer the question
    You MUST write a passage from cotent in the given context.
    Try to include as many key details as possible.
    
    Question: {query_str}
    Context: {context_str}
    Passage:
    """
)

hyde_composer = ChatUpstage(model="solar-pro2", temperature=0)

def init_state(state: HyDEState) -> HyDEState:
    question = state["messages"][-1].content
    return {"question": question, "context": [], "is_retrieved": False, "is_hyde": False}

def hyde_transform(state: HyDEState) -> HyDEState:
    if state.get("is_retrieved"):
        query_str = state["question"]
        context_str = state["context"]
        prompt = HYDE_PROMPT.format(query_str=query_str, context_str=context_str)
        response = hyde_composer.invoke(prompt)
        return {"passage": response.content, "is_hyde": True}
    else:
        return {"passage": ""}


# 문서 검색 노드: 사용자 질문에 관련된 문서를 검색하는 노드
def retrieve_document(state: HyDEState) -> HyDEState:
    if state["is_retrieved"]:
        latest_question = state["passage"]
    else:
        latest_question = state["question"]
   
    # 앙상블 리트리버를 사용하여 질문과 관련성 높은 문서 검색
    # BM25(키워드 기반)와 FAISS(의미 기반)를 결합하여 검색 성능 향상
    bm25_retriever, faiss_retriever = get_ensemble_retriever()
    retrieved_docs_faiss = faiss_retriever.invoke(latest_question)
    retrieved_docs_bm25 = bm25_retriever.invoke(latest_question)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25

    return {"documents": retrieved_docs}

def rerank_document(state: HyDEState) -> HyDEState:
    retrieved_docs = state["documents"]
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)
   
    return {"documents": rrf_docs, "context": context, "is_retrieved": True}


RAG_ANSWER = """
You are a helpful assistant that answer the question based on the passage and context.

You are given a question, passage and context.

You need to answer the question based on the passage and context.

Here are the question, passage and context:

Question: {question}
Passage: {passage}
Context: {context}

Answer:
"""


answer_model = ChatUpstage(model="solar-pro2", temperature=0)


def rag_answer(state: HyDEState) -> HyDEState:
    question = state["question"]
    passage = state["passage"]
    context = state["context"]
    prompt = RAG_ANSWER.format(question=question, passage=passage, context=context)
    response = answer_model.invoke(prompt)
    return {"messages": [response]}

def recursive_query_router(state: HyDEState) -> Literal["retrieve_document", "aggregate_answer"]:
    is_hyde_transformed = state["is_hyde"]
    if is_hyde_transformed:
        return "rag_answer"
    else:
        return "hyde_transform"

builder = StateGraph(HyDEState)
builder.add_node("init_state", init_state)
builder.add_node("hyde_transform", hyde_transform)
builder.add_node("retrieve_document", retrieve_document)
builder.add_node("rerank_document", rerank_document)
builder.add_node("rag_answer", rag_answer)

builder.add_edge(START, "init_state")
builder.add_edge("init_state", "hyde_transform")
builder.add_edge("hyde_transform", "retrieve_document")
builder.add_edge("retrieve_document", "rerank_document")
builder.add_conditional_edges(
    "rerank_document",
    recursive_query_router,
    {
        "hyde_transform": "hyde_transform",
        "rag_answer": "rag_answer"
    }
)

builder.add_edge("rag_answer", END)

hyde = builder.compile()

if __name__ == "__main__":
    from pprint import pprint
    config = {"configurable": {"thread_id": "1"}}
    # 넓은 범위의 질문이나 열린 결말의 질문은 부적합함
    # for chunk in hyde.stream({"question": "AI 트렌드는 무엇인가?"}, stream_mode="updates", config=config):
    #     print(chunk)
    for chunk in hyde.stream({"messages": [{"role": "user", "content": "상위 AI 논문 인용 순위 3개는 무엇인가?"}]}, stream_mode="updates", config=config):
        pprint(chunk)

