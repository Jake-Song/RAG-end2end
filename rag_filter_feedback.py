"""
metadata filter agent with human in the loop
"""
from pathlib import Path
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_core.messages import AnyMessage
from langchain.messages import ToolMessage
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START, END
from utils.utils import format_context
from reranker.rrf import ReciprocalRankFusion
import pickle

root = Path().resolve()

# 모든 문서 로드
papers = []
for path in sorted(Path(root).joinpath("outputs").glob("*_split_documents.pkl")):
    with open(path, "rb") as f:
        split_documents = pickle.load(f)
        for doc in split_documents:
            doc.metadata["title"] = path.stem.split("_output", 1)[0]
        papers.append(split_documents)

embeddings = UpstageEmbeddings(model="embedding-passage")

# 모든 문서 병합
docs = []
for paper in papers:
    docs.extend(paper)

vectorstore = FAISS.load_local(
    f"{root}/faiss_index", 
    embeddings,
    "SPRI_ALL",
    allow_dangerous_deserialization=True  # needed in newer versions
)

class RetrieverInput(BaseModel):
    query: str = Field(description="query to retrieve documents from the vector database")
    filter: dict[Literal["title"], list[Literal["SPRI_2022", "SPRI_2023", "SPRI_2025"]]] = Field(
        default={"title": ["SPRI_2022", "SPRI_2023", "SPRI_2025"]},
        description="The filter to apply to the vector database."
    )    

@tool(args_schema=RetrieverInput)
def retriever(query: str, filter: dict = {"title": ["SPRI_2022", "SPRI_2023", "SPRI_2025"]}) -> list[Document]:
    """Retrieve documents from the vector database."""

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10,"filter": filter})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    retrieved_docs_faiss = faiss_retriever.invoke(query)
    retrieved_docs_bm25 = bm25_retriever.invoke(query)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}

tools = [retriever]
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: dict):
    """Performs the tool call"""
        
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
               
        exception = ""
        while True:
            feedback = interrupt({
                "action": "Review and approve these tool calls",
                "tool": tools_by_name[tool_call["name"]].name,
                "query": tool_call["args"]["query"],
                "filter": (tool_call["args"]["filter"] if "filter" in tool_call["args"] else ""),
                "exception": exception
            })
            # Validate the input
            valid_actions = {"approve", "edit", "reject"}
            valid_filters = {"SPRI_2022", "SPRI_2023", "SPRI_2025"}

            if isinstance(feedback["action"], str) and feedback["action"] in valid_actions:
                if feedback["action"] == "approve" or feedback["action"] == "reject":
                    break
                if isinstance(feedback["filter"], str) and feedback["filter"] in valid_filters:
                    break
                else:
                    exception = f"'{feedback['filter']}' is not a valid filter. Please enter filter one of {sorted(valid_filters)}."
            else:
                exception = f"'{feedback}' is not a valid feedback. action must be one of {sorted(valid_actions)}."
        
        if feedback["action"] == "approve":
            print("\n"+"args: ", tool_call["args"])
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        
        elif feedback["action"] == "edit":
            tool_call["args"]["query"] = feedback["query"]
            tool_call["args"]["filter"] = {"title": feedback["filter"]}
            print(tool_call["args"])
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        
        elif feedback["action"] == "reject":
            result.append(ToolMessage(content="Tool call rejected", tool_call_id=tool_call["id"]))

    return {"messages": result}

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

response_model = ChatUpstage(model="solar-pro2", temperature=0)

def generate_query_or_respond(state: GraphState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever]).invoke(state["messages"])  
    )
    return {"messages": [response]}

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
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

def should_continue(state: GraphState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

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
    initial = graph.invoke({"messages": [{"role": "user", "content": "AI 규제 관련 문서"}]}, config=config)

    print(initial["__interrupt__"])  

    # Resume with the decision; True routes to proceed, False to cancel
    resumed = graph.invoke(Command(resume={"action": "recommand", "query": "AI 규제 관련 문서", "filter": ["SPRI_2024"]}), config=config)
    print(resumed["__interrupt__"])

    resumed = graph.invoke(Command(resume={"action": "edit", "query": "AI 규제 관련 문서", "filter": ["SPRI_2024"]}), config=config)
    print(resumed["__interrupt__"])