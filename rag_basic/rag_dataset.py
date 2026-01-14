"""
LangGraph RAG 모델
가장 기본적인 RAG 모델 구조
1. 문서 검색
2. 답변 생성

대화형 cli 모드 제공 `python rag.py`
"""
import os
import pickle
import cohere
from pathlib import Path
from typing import Annotated, TypedDict
from langchain.messages import AIMessage
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from utils.utils import format_context
from scripts.retrieve import load_retriever
from reranker.rrf import ReciprocalRankFusion
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent

llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0.0)
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

with open(project_root / "datasets" / "hotpotqa.pkl", "rb") as f:
    split_documents = pickle.load(f)


def get_ensemble_retriever():
    embeddings = UpstageEmbeddings(model="embedding-passage")
    bm25_retriever, faiss_retriever = load_retriever(
        split_documents, embeddings, db_name="hotpotqa_100", kiwi=False, search_k=100
    )
    return bm25_retriever, faiss_retriever


def cohere_rerank_internal(query: str, retrieved_docs: list[Document], top_n: int = 20) -> list[Document]:
    rerank_response = co.rerank(
        model="rerank-v4.0-fast", query=query, documents=[doc.page_content for doc in retrieved_docs], top_n=top_n
    )
    return [retrieved_docs[result.index] for result in rerank_response.results]


class GraphState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    documents: Annotated[list[Document], "Documents"]
    chunk_ids: Annotated[list[int], "Chunk IDs"]


# 노드
def retrieve_document(state: GraphState) -> GraphState:
    latest_question = state["question"]
    
    # 앙상블 리트리버를 사용하여 질문과 관련성 높은 문서 검색
    # BM25(키워드 기반)와 FAISS(의미 기반)를 결합하여 검색 성능 향상
    bm25_retriever, faiss_retriever = get_ensemble_retriever()
    retrieved_docs_faiss = faiss_retriever.invoke(latest_question)
    retrieved_docs_bm25 = bm25_retriever.invoke(latest_question)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=100)

    return {"documents": rrf_docs}


def rerank_document(state: GraphState) -> GraphState:
    rrf_docs = state["documents"]
    query = state["question"]
    
    reranked_docs = cohere_rerank_internal(query, rrf_docs, top_n=20)
    context = format_context(reranked_docs)

    return {"documents": reranked_docs, "context": context}


# 답변 생성 노드: LLM을 사용하여 검색된 문서를 기반으로 답변 생성
def llm_answer(state: GraphState) -> GraphState:
    
    latest_question = state["question"]
    context = state["context"]

    system_prompt = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
    """

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Context: " + context},
        {"role": "user", "content": "Question: " + latest_question},
    ]

    response = llm.invoke(prompt)

    chunk_ids = []
    for doc in state["documents"]:
        chunk_ids.append(doc.metadata["chunk_id"])

    return {"answer": response.content, "context": context, "chunk_ids": chunk_ids}


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("rerank", rerank_document)
workflow.add_node("llm_answer", llm_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "llm_answer")
workflow.add_edge("llm_answer", END)

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)


def rag_bot_invoke(question: str) -> dict:
    from langchain_core.runnables import RunnableConfig
    import uuid

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})

    inputs = {"question": question}
    result = app.invoke(inputs, config)
    return {'answer': result['answer'], 'context': result['context'], 'chunk_ids': result['chunk_ids']}


def rag_bot_batch(questions: list[str]) -> list[dict]:
    from langchain_core.runnables import RunnableConfig
    import uuid

    inputs = [{"question": question} for question in questions]
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})
    results = app.batch(inputs, config)

    return results

def rag_bot_graph(prompt: str) -> dict:
    from langchain_core.runnables import RunnableConfig
    import uuid

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})

    inputs = {"question": prompt}
    result = app.invoke(inputs, config)
    return AIMessage(content=result['answer'])


if __name__ == "__main__":
    for chunk in app.stream(
        {
            "question": "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?",
        },stream_mode="updates",config={"configurable": {"thread_id": "1"}}
    ):
        print(chunk)
  
    