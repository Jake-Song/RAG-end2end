import os
from pathlib import Path
from typing import Annotated, TypedDict
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.retrievers import BM25Retriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from utils.utils import format_context
from reranker.rrf import ReciprocalRankFusion
from langchain_postgres import PGVector
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()

root = Path().resolve()

embeddings = UpstageEmbeddings(model="embedding-passage")
DB_URI = os.environ["POSTGRES_URI"]

from sqlalchemy import create_engine, text

engine = create_engine(DB_URI)

def sql_query(title: str) -> list[tuple[str, dict]]:
    """
    Sparse Vector Search(BM25등)을 위한 sql 쿼리. 
    하이브리드 방식(BM25 + PG Vector Search)을 사용하기 
    때문에 두 방식의 결과를 합치기 위해 사용.
    """
    with engine.connect() as conn:
        # Custom SQL query
        query = conn.execute(text(f"""
            SELECT 
                e.document,
                e.cmetadata            
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection_name
                AND e.cmetadata->>'title' = :title
        """), {
            "collection_name": "SPRI_ALL",
            "title": title,
        })
        
        return query.fetchall()

class GraphState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    documents: Annotated[list[Document], "Documents"]
    page_number: Annotated[list[int], "Page Number"]


pg_vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="SPRI_ALL",
    connection=DB_URI
)

result = sql_query("SPRI_2023")
docs = []
for r in result:
    docs.append(Document(page_content=r[0], metadata=r[1]))

# 앙상블 리트리버를 사용하여 질문과 관련성 높은 문서 검색
# BM25(키워드 기반)와 FAISS(의미 기반)를 결합하여 검색 성능 향상
pg_retriever = pg_vectorstore.as_retriever(search_kwargs={"k": 10, "filter": 
    {"$and": [
            {"title": {"$eq": "SPRI_2023"}}                
        ]}
    })
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

# 노드
def retrieve_document(state: GraphState) -> GraphState:
    latest_question = state["question"]
    retrieved_docs_pg = pg_retriever.invoke(latest_question)
    retrieved_docs_bm25 = bm25_retriever.invoke(latest_question)
    retrieved_docs_pg = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_pg)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_pg + retrieved_docs_bm25

    return {"documents": retrieved_docs}

def rerank_document(state: GraphState) -> GraphState:
    retrieved_docs = state["documents"]
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}

# 답변 생성 노드: LLM을 사용하여 검색된 문서를 기반으로 답변 생성
def llm_answer(state: GraphState) -> GraphState:
    
    latest_question = state["question"]
    context = state["context"]

    llm = ChatUpstage(model="solar-pro2", temperature=0.0)

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

    return {"answer": response.content, "documents": state["documents"], "page_number": page_number}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("rerank", rerank_document)
workflow.add_node("llm_answer", llm_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "llm_answer")
workflow.add_edge("llm_answer", END)

memory = InMemorySaver()

app = workflow.compile(checkpointer=memory)
    

if __name__ == "__main__":
    
    config = {"configurable": {"thread_id": "1"}}
    
    # result = app.invoke({"question": "AI 트렌드"}, config=config)
    # pprint(result["answer"])
    # pprint([doc.metadata for doc in result["documents"]])
    # pprint(result["page_number"])

    for chunk in app.stream({"question": "AI 트렌드"}, stream_mode="updates", config=config):
        pprint(chunk)