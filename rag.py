"""
LangGraph RAG 모델
가장 기본적인 RAG 모델 구조
1. 문서 검색
2. 답변 생성

대화형 cli 모드 제공 `python rag.py`
"""

from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain.schema import Document
import pickle
import time
from utils.utils import format_context
from scripts.retrieve import create_retriever, load_retriever
from reranker.rrf import ReciprocalRankFusion
from config import output_path_prefix

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
# from langchain_teddynote import logging

# # 프로젝트 이름을 입력합니다.
# logging.langsmith("Langgraph")

# GraphState 상태 정의: 그래프 노드 간 전달되는 데이터 구조
class GraphState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    documents: Annotated[list[Document], "Documents"]
    page_number: Annotated[list[int], "Page Number"]

# 전역 변수로 캐싱 (lazy loading)
_ensemble_retriever = None
_app = None

def get_ensemble_retriever():
    """앙상블 리트리버를 lazy loading으로 가져옵니다 (최초 1회만 로드)"""
    global _ensemble_retriever
    global _faiss_retriever
    global _bm25_retriever

    if _ensemble_retriever is None:
        with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
            split_documents = pickle.load(f)

        # 앙상블 리트리버 로드: BM25 + FAISS 벡터 검색을 결합한 하이브리드 검색기
        embeddings = UpstageEmbeddings(model="embedding-passage")
        _ensemble_retriever, _bm25_retriever, _faiss_retriever = load_retriever(split_documents, embeddings, kiwi=True, search_k=10)
    return _ensemble_retriever, _bm25_retriever, _faiss_retriever

# 문서 검색 노드: 사용자 질문에 관련된 문서를 검색하는 노드
def retrieve_document(state: GraphState) -> GraphState:
    latest_question = state["question"]

    # 앙상블 리트리버를 사용하여 질문과 관련성 높은 문서 검색
    # BM25(키워드 기반)와 FAISS(의미 기반)를 결합하여 검색 성능 향상
    _, bm25_retriever, faiss_retriever = get_ensemble_retriever()
    retrieved_docs_faiss = faiss_retriever.invoke(latest_question)
    retrieved_docs_bm25 = bm25_retriever.invoke(latest_question)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25

    return {"documents": retrieved_docs}

def rerank_document(state: GraphState) -> GraphState:
    retrieved_docs = state["documents"]
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=3)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}

# 답변 생성 노드: LLM을 사용하여 검색된 문서를 기반으로 답변 생성
def llm_answer(state: GraphState) -> GraphState:
    start_time = time.time()

    latest_question = state["question"]
    context = state["context"]

    # OpenAI LLM 초기화 (temperature=0: 결정적 답변 생성)
    llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

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

    page_number = []
    for doc in state["documents"]:
        page_number.append(doc.metadata["page"])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"LLM answer generation time: {execution_time:.2f} seconds")

    return {"answer": response.content, "documents": state["documents"], "page_number": page_number}

def get_app():
    """LangGraph 앱을 lazy loading으로 가져옵니다 (최초 1회만 컴파일)"""
    global _app
    if _app is None:
        ensemble_retriever = get_ensemble_retriever()
        workflow = StateGraph(GraphState, ensemble_retriever=ensemble_retriever)

        workflow.add_node("retrieve", retrieve_document)
        workflow.add_node("rerank", rerank_document)
        workflow.add_node("llm_answer", llm_answer)

        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "llm_answer")
        workflow.add_edge("llm_answer", END)

        workflow.set_entry_point("retrieve")

        memory = MemorySaver()

        _app = workflow.compile(checkpointer=memory)
    return _app

def preload():
    """CLI 실행 시 모든 리소스를 미리 로드합니다"""
    print("Loading RAG system...")
    start_time = time.time()
    get_app()  # 이 함수가 내부적으로 get_ensemble_retriever()도 호출함
    load_time = time.time() - start_time
    print(f"RAG system loaded in {load_time:.2f} seconds")
    return load_time

def rag_bot_invoke(question: str) -> dict:
    from langchain_core.runnables import RunnableConfig
    from langchain_teddynote.messages import random_uuid

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    app = get_app()
    inputs = {"question": question}
    result = app.invoke(inputs, config)
    return {'answer': result['answer'], 'documents': result['documents'], 'page_number': result['page_number']}
    
def rag_bot_batch(questions: list[str]) -> dict:
    from langchain_core.runnables import RunnableConfig
    from langchain_teddynote.messages import random_uuid

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    app = get_app()
    inputs = [{"question": question} for question in questions]

    results = app.batch(inputs, config)
    return results

if __name__ == "__main__":
    import sys

    preload()

    # 인자가 주어진 경우 단일 질문 모드
    if len(sys.argv) > 1:
        question = sys.argv[1]
        result = rag_bot_invoke(question)
        print(result['answer'], "\n")
        print(result['documents'], "\n")
        print(result['page_number'], "\n")
    # 인자가 없는 경우 대화형 루프 모드
    else:
        print("=" * 60)
        print("RAG Bot Interactive Mode")
        print("Type 'exit' or 'quit' to end the session")
        print("=" * 60)

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                result = rag_bot_invoke(question)
                print("\nAnswer:", result['answer'])
                print("\nPage numbers:", result['page_number'])
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")
                continue
  
    