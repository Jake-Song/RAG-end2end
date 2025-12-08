import os
from pathlib import Path
from typing import Annotated, TypedDict
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langgraph.graph import StateGraph, START, END
from utils.utils import format_context
from elasticsearch import Elasticsearch
from pprint import pprint
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from dotenv import load_dotenv
root = Path(__file__).parent.resolve()
load_dotenv()
load_dotenv(root.joinpath("elastic-start-local/.env"))
URL = os.environ["ES_LOCAL_URL"]
API_KEY = os.environ["ES_LOCAL_API_KEY"]

embeddings = UpstageEmbeddings(model="embedding-passage")

index_name = "vector_search_demo"
es = Elasticsearch(
    URL,
    api_key=API_KEY
)
def hybrid_search(query_text: str, top_k: int = 5):
    query_vector = embeddings.embed_query(query_text)
    
    common_filters = {
        "bool": {
            "should": [ # or
                {"term": {"metadata.tag": 'AI인재'}},
                {"term": {"metadata.tag": 'AI스타트업'}},
                {"term": {"metadata.tag": 'AI인덱스'}},
            ],
            "must_not": [# not
                {"term": {"metadata.tag": '정부정책책'}},
            ]
        }
    }

    # Combine KNN with text search
    hybrid_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": query_text
                        }
                    }
                ],
                "filter": common_filters  # Use same filter
            }
        },
        "knn": {
            "field": "vector",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 100,
            "filter": common_filters  # Reuse same filter
        },
        "_source": ["text", "metadata"]
    }
    
    response = es.search(index=index_name, body=hybrid_query)
    return response


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: Annotated[str, "Context"]
    documents: Annotated[list[Document], "Documents"]
    page_number: Annotated[list[int], "Page Number"]

# 노드
def retrieve_document(state: GraphState) -> GraphState:
    latest_question = state["messages"][-1].content
    print("latest_question: ", latest_question)
    retrieved_docs_elastic = hybrid_search(latest_question, top_k=5)
    print("length of retrieved_docs_elastic: ", len(retrieved_docs_elastic['hits']['hits']))
    docs = []
    for r in retrieved_docs_elastic['hits']['hits'][:5]:
        doc = Document(page_content=r['_source']['text'], metadata=r['_source']['metadata'])
        docs.append(doc)
    return {"documents": docs}

def rerank_document(state: GraphState) -> GraphState:
    retrieved_docs = state["documents"]
    context = format_context(retrieved_docs)

    return {"documents": retrieved_docs, "context": context}

# 답변 생성 노드: LLM을 사용하여 검색된 문서를 기반으로 답변 생성
def llm_answer(state: GraphState) -> GraphState:
    
    latest_question = state["messages"][-1].content
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

    return {"messages": [response], "documents": state["documents"], "page_number": page_number}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("rerank", rerank_document)
workflow.add_node("llm_answer", llm_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "llm_answer")
workflow.add_edge("llm_answer", END)


app = workflow.compile()
    

if __name__ == "__main__":
    
    config = {"configurable": {"thread_id": "1"}}
    
    result = app.invoke({"messages": [{"role": "user", "content": "AI 트렌드"}]}, config=config)
    pprint(result["messages"])
    
    # for chunk in app.stream({"messages": [{"role": "user", "content": "AI 트렌드"}]}, stream_mode="values", config=config):
    #     pprint(chunk)