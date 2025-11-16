# 필요한 라이브러리 임포트
from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph  # LangGraph 그래프 구조 관련
from langgraph.checkpoint.memory import MemorySaver  # 메모리 기반 체크포인터
from langchain_openai import ChatOpenAI  # OpenAI 모델 및 임베딩
from langchain_upstage import UpstageEmbeddings
from langchain.schema import Document  # LangChain 문서 스키마
import pickle  # 직렬화된 데이터 로드용
import time  # 실행 시간 측정용
from utils.utils import format_context
from scripts.retrieve import create_retriever, load_retriever  # 리트리버 생성/로드 함수
from config import output_path_prefix  # 설정 파일에서 출력 경로 가져오기

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
# from langchain_teddynote import logging

# # 프로젝트 이름을 입력합니다.
# logging.langsmith("Langgraph")

# GraphState 상태 정의: 그래프 노드 간 전달되는 데이터 구조
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 사용자의 질문
    context: Annotated[str, "Context"]  # 문서에서 검색된 컨텍스트 (문자열로 결합됨)
    answer: Annotated[str, "Answer"]  # LLM이 생성한 답변
    documents: Annotated[list[Document], "Documents"]  # 검색된 Document 객체 리스트
    page_number: Annotated[list[int], "Page Number"]  # 관련 문서의 페이지 번호 리스트

# 전역 변수로 캐싱 (lazy loading)
_ensemble_retriever = None
_app = None

def get_ensemble_retriever():
    """앙상블 리트리버를 lazy loading으로 가져옵니다 (최초 1회만 로드)"""
    global _ensemble_retriever
    if _ensemble_retriever is None:
        # 사전에 처리된 문서 청크 로드 (pickle 파일에서)
        with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
            split_documents = pickle.load(f)

        # 앙상블 리트리버 로드: BM25 + FAISS 벡터 검색을 결합한 하이브리드 검색기
        embeddings = UpstageEmbeddings(model="embedding-passage")
        _ensemble_retriever = load_retriever(split_documents, embeddings, kiwi=True)
    return _ensemble_retriever

# 문서 검색 노드: 사용자 질문에 관련된 문서를 검색하는 노드
def retrieve_document(state: GraphState) -> GraphState:
    # 현재 상태에서 사용자의 질문 가져오기
    latest_question = state["question"]

    # 앙상블 리트리버를 사용하여 질문과 관련성 높은 문서 검색
    # BM25(키워드 기반)와 FAISS(의미 기반)를 결합하여 검색 성능 향상
    ensemble_retriever = get_ensemble_retriever()
    retrieved_docs = ensemble_retriever.invoke(latest_question)

    # 검색된 모든 문서의 내용을 하나의 문자열로 결합
    context = format_context(retrieved_docs)

    # 검색된 문서와 컨텍스트를 상태에 저장하여 다음 노드로 전달
    return {"documents": retrieved_docs, "context": context}

# 답변 생성 노드: LLM을 사용하여 검색된 문서를 기반으로 답변 생성
def llm_answer(state: GraphState) -> GraphState:
    # 시작 시간 기록
    start_time = time.time()

    # 상태에서 사용자의 질문 가져오기
    latest_question = state["question"]

    # 이전 노드(retrieve_document)에서 검색된 문서 컨텍스트 가져오기
    context = state["context"]

    # OpenAI LLM 초기화 (temperature=0: 결정적 답변 생성)
    llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

    # 시스템 프롬프트: LLM의 역할 및 답변 방식 정의
    system_prompt = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.

    """

    # 프롬프트 구성: 시스템 메시지 + 컨텍스트 + 질문
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Context: " + context},
        {"role": "user", "content": "Question: " + latest_question},
    ]

    # LLM 호출하여 답변 생성
    response = llm.invoke(prompt)

    # 검색된 각 문서의 페이지 번호 추출
    page_number = []
    for doc in state["documents"]:
        page_number.append(doc.metadata["page"])

    # 종료 시간 기록 및 실행 시간 출력
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"LLM answer generation time: {execution_time:.2f} seconds")

    # 생성된 답변, 문서, 페이지 번호를 상태에 저장하여 반환
    return {"answer": response.content, "documents": state["documents"], "page_number": page_number}

def get_app():
    """LangGraph 앱을 lazy loading으로 가져옵니다 (최초 1회만 컴파일)"""
    global _app
    if _app is None:
        # 그래프 생성
        ensemble_retriever = get_ensemble_retriever()
        workflow = StateGraph(GraphState, ensemble_retriever=ensemble_retriever)

        # 노드 정의
        workflow.add_node("retrieve", retrieve_document)
        workflow.add_node("llm_answer", llm_answer)

        # 엣지 정의
        workflow.add_edge("retrieve", "llm_answer")  # 검색 -> 답변
        workflow.add_edge("llm_answer", END)  # 답변 -> 종료

        # 그래프 진입점 설정
        workflow.set_entry_point("retrieve")

        # 체크포인터 설정
        memory = MemorySaver()

        # 컴파일
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

    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    app = get_app()
    inputs = {"question": question}
    result = app.invoke(inputs, config)
    return {'answer': result['answer'], 'documents': result['documents'], 'page_number': result['page_number']}
    
def rag_bot_batch(questions: list[str]) -> dict:
    from langchain_core.runnables import RunnableConfig
    from langchain_teddynote.messages import random_uuid

    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    app = get_app()
    inputs = [{"question": question} for question in questions]

    results = app.batch(inputs, config)
    return results

if __name__ == "__main__":
    import sys

    # CLI 실행 시 모든 리소스 미리 로드
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

                # 종료 명령어 체크
                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                # 빈 입력 무시
                if not question:
                    continue

                # 질문 처리
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
  
    