"""
metadata filter agent with human in the loop
"""
import os
from pathlib import Path
from typing import Annotated, TypedDict, Literal
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langgraph.types import interrupt, Command
from langchain.tools import tool
from langchain.messages import ToolMessage, HumanMessage, AIMessage, AnyMessage
from utils.utils import format_context
from elasticsearch import Elasticsearch


from dotenv import load_dotenv
root = Path(__file__).parent.resolve()
load_dotenv()
load_dotenv(root.joinpath("elastic-start-local/.env"))

import logging

# Configure logging (put this near the top)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            # "should": [ # or
            #     {"term": {"metadata.tag": 'AI인재'}},
            #     {"term": {"metadata.tag": 'AI스타트업'}},
            #     {"term": {"metadata.tag": 'AI인덱스'}},
            # ],
            # "must_not": [# not
            #     {"term": {"metadata.tag": '정부정책'}},
            # ]
            "must": [
                # {"term": {"metadata.title": 'SPRI_2022'}},
                # {"term": {"metadata.title": 'SPRI_2023'}},
                {"term": {"metadata.title": 'SPRI_2025'}},
            ],            
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


@tool
def retriever(query: str) -> list[Document]:
    """Retrieve documents from the vector database.

    Args:
        query: The query to retrieve documents from the vector database.
    """

    retrieved_docs_elastic = hybrid_search(query, top_k=5)
    # logger.info(f"length of retrieved_docs_elastic: {len(retrieved_docs_elastic['hits']['hits'])}")
    docs = []
    for r in retrieved_docs_elastic['hits']['hits'][:5]:   
        doc = Document(page_content=r['_source']['text'], metadata=r['_source']['metadata'])
        docs.append(doc)

    context = format_context(docs)
    # logger.info(f"Retrieved {len(docs)} documents")
    # logger.info(f"context: {context}")
    return {"documents": docs, "context": context}

tools = [retriever]
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: dict):
    """Performs the tool call"""
        
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
               
        
        observation = tool.invoke(tool_call["args"])
        
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))      
        # logger.info(f"result of tool_node: {result}")
    return {"messages": result, "documents": observation["documents"], "context": observation["context"]}

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    context: str
    documents: list[Document]

response_model = ChatUpstage(model="solar-pro2", temperature=0)
response_model_openai = ChatOpenAI(model="gpt-5-mini", temperature=0)
def generate_query_or_respond(state: GraphState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    question = state["messages"][-1].content
    response = (
        response_model_openai
        .bind_tools([retriever]).invoke(question)  
    )
    # logger.info(f"messages: {state['messages'][-1].content}")
    return {"messages": [response], "question": question}

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
    question = state["question"]
    context = state["context"]
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model_openai.invoke(prompt)
    # logger.info(f"messages: {state['messages'][-1].content}")
    return {"messages": [AIMessage(content=response.content)]}

def should_continue(state: GraphState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    last_message = state["messages"][-1]

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
checkpointer = InMemorySaver()
# Compile
graph = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    # initial = graph.invoke({"messages": [{"role": "user", "content": "AI 인덱스 2025 설명. 문서에서 찾아봐"}]}, config=config)

    # print(initial["messages"][-1].content)  
    
    for  chunk in graph.stream({"messages": [{"role": "user", "content": "AI 인덱스 2025 벤치마크 점수 동향 설명."}]}, 
    stream_mode="updates", config=config):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")

    