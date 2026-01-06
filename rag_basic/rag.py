"""
LangGraph RAG 모델
가장 기본적인 RAG 모델 구조
1. 문서 검색
2. 답변 생성

대화형 cli 모드 제공 `python rag.py`
"""

from typing import Annotated, TypedDict
from langchain.messages import AIMessage
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import START,END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from utils.utils import format_context
from scripts.retrieve import load_retriever
from reranker.rrf import ReciprocalRankFusion
from config import output_path_prefix
import pickle

with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
    split_documents = pickle.load(f)

def get_ensemble_retriever():
    embeddings = UpstageEmbeddings(model="embedding-passage")
    bm25_retriever, faiss_retriever = load_retriever(split_documents, embeddings, kiwi=False, search_k=10)
    return bm25_retriever, faiss_retriever


class GraphState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    documents: Annotated[list[Document], "Documents"]
    page_number: Annotated[list[int], "Page Number"]


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

    return {"documents": retrieved_docs}

def rerank_document(state: GraphState) -> GraphState:
    retrieved_docs = state["documents"]
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=2)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}

# 답변 생성 노드: LLM을 사용하여 검색된 문서를 기반으로 답변 생성
def llm_answer(state: GraphState) -> GraphState:
    
    latest_question = state["question"]
    context = state["context"]

    llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0.0)
    # llm = ChatUpstage(model="solar-pro2", temperature=0.0)

    system_prompt = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        CRITICAL RULES:
        1. 억 달러 → billion USD: divide by 10, DROP the 억
        (1,508억 달러 = 150.8 billion USD, NOT 150.8억)
        
        2. Every number needs a direct quote from source
        - No quote? Don't include it.
        
        3. Before saying "정보 없음", list what you DID find
        
        4. 생성형 AI ≠ 전체 AI — match exact category
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

    return {"answer": response.content, "documents": state["documents"], "page_number": page_number}

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
    return {'answer': result['answer'], 'documents': result['documents'], 'page_number': result['page_number']}

def rag_bot_batch(questions: list[str]) -> dict:
    from langchain_core.runnables import RunnableConfig
    import uuid

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})

    inputs = [{"question": question} for question in questions]

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
    import sys

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
  
    