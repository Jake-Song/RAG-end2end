from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
import pickle

from scripts.retriver import create_retriever, load_retriever
from config import output_path_prefix
from dotenv import load_dotenv
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("Langgraph")

# GraphState 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    answer: Annotated[str, "Answer"]  # 답변
    documents: Annotated[list[Document], "Documents"]  # 검색된 문서
    page_number: Annotated[list[int], "Page Number"]  # 페이지 번호

with open(f"outputs/{output_path_prefix}_split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)

ensemble_retriever = load_retriever(split_documents)

# 문서 검색 노드
def retrieve_document(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = ensemble_retriever.invoke(latest_question)
    context = "".join([doc.page_content for doc in retrieved_docs])
    # 검색된 문서를 context 키에 저장합니다.
    return {"documents": retrieved_docs, "context": context}


# 답변 생성 노드
def llm_answer(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 검색된 문서를 상태에서 가져옵니다.
    context = state["context"]
   
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
        
    # 체인을 호출하여 답변을 생성합니다.
    response = llm.invoke(prompt)
    page_number = []
    for doc in state["documents"]:
        page_number.append(doc.metadata["page"])
    # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.
    return {"answer": response.content, "documents": state["documents"], "page_number": page_number}

with open(f"outputs/{output_path_prefix}_split_documents.pkl", "rb") as f:
    split_documents = pickle.load(f)

# embeddings = OpenAIEmbeddings()
# ensemble_retriever = create_retriever(split_documents, embeddings)
ensemble_retriever = load_retriever(split_documents)

# 그래프 생성
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
app = workflow.compile(checkpointer=memory)

def rag_bot_invoke(question: str) -> dict:
    from langchain_core.runnables import RunnableConfig
    from langchain_teddynote.messages import random_uuid

    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    inputs = GraphState(question=question)
    result = app.invoke(inputs, config)
    return {'answer': result['answer'], 'documents': result['documents'], 'page_number': result['page_number']}
    

if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "What are the risks and challenges of Korea in global economy?"
    result = rag_bot_invoke(question)
    print(result['answer'], "\n")
    print(result['documents'], "\n")
    print(result['page_number'], "\n")
  
    