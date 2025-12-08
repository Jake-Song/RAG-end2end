from pydantic import BaseModel, Field
from typing import List, Annotated, TypedDict
from typing_extensions import Literal
from langchain_upstage import ChatUpstage
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage
from langgraph.graph import add_messages
import operator
from rag_basic.rag import rag_bot_graph
from langgraph.checkpoint.memory import InMemorySaver
llm = ChatUpstage(model="solar-pro2", temperature=0)

# Schema for structured output to use in planning
class Query(BaseModel):
    name: str = Field(
        description="Name for this query to answer the question.",
    )
    reasoning: str = Field(
        description="reasoning how this query is related to the question.",
    )

class Queries(BaseModel):
    queries: List[Query] = Field(
        description="Queries to answer the question.",
    )

class Route(BaseModel):
    step: Literal["single", "decompose"] = Field(
        None, description="The next step in the routing process"
    )

router = llm.with_structured_output(Route)
planner = llm.with_structured_output(Queries)


# Graph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str  # Report topic
    queries: list[Query]  # List of report sections
    completed_queries: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_answer: str  # Final report
    decision: str

# Worker state
class WorkerState(TypedDict):
    query: Query
    question: str
    completed_queries: Annotated[list, operator.add]


# Nodes
def init_state(state: State):
    return {"question": state["messages"][-1].content}

def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    question = state["question"]
    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the question to single query or multi query based on the user's request."
            ),
            HumanMessage(content=question),
        ]
    )

    return {"question": question, "decision": decision.step}

# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "single":
        return "single_query"
    elif state["decision"] == "decompose":
        return "decompose_query"

# Nodes
def single_query(state: State):
    result = rag_bot_graph(state["question"])
    return {"messages": [result]}

def decompose_query(state: State):
    """Decompose the question into smaller queries"""

    # Generate queries
    queries = planner.invoke(
        [
            SystemMessage(content="Generate atomic queries for the question."),
            HumanMessage(content=f"Here is the question: {state['question']}")
        ]
    )

    return {"queries": queries.queries}

RAG_PROMPT = ("""
answer the following question and reasoning:
question: {question}
reasoning: {reasoning}
""")


def llm_call(state: WorkerState):
    """Worker response a query"""

    # Generate query
    prompt = RAG_PROMPT.format(question=state['query'].name, reasoning=state['query'].reasoning)
    answer = rag_bot_graph(prompt)

    # Write the updated section to completed sections
    return {"completed_queries": [answer.content]}


def synthesizer(state: State):
    """Synthesize full answer from queries"""

    # List of completed sections
    completed_queries = state["completed_queries"]

    # Format completed section to str to use as context for final sections
    completed_queries = "\n\n---\n\n".join(completed_queries)

    return {"final_answer": completed_queries}

SUMMARIZER_PROMPT = ("""
summarize the following answer that is the most relevant to the original question:
question: {question}
answer: {answer}
""")

def summarizer(state: State):
    """Summarize the final answer"""
    final_answer = state["final_answer"]
    prompt = SUMMARIZER_PROMPT.format(question=state["question"], answer=final_answer)
    result = llm.invoke(prompt)
    
    return {"messages": [result]}

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off query writing in parallel via Send() API
    return [Send("llm_call", {"query": q}) for q in state["queries"]]

from langgraph.types import interrupt, Command

def review_answer(state: State):
    # Ask a reviewer to edit the generated content
    decision = interrupt({
        "action": "Review and approve this answer",
        "question": state['question'],
        "messages": state["messages"][-1].content,
        "step": state["decision"],
    })

    if decision["action"] == "approve":
        return Command(goto=END)
    elif decision["action"] == "edit":
        question = decision["query"]
        return Command(goto="llm_call_router", update={"question": question})
    elif decision["action"] == "reject":
        step = decision["step"]
        if step == "single":
            return Command(goto="single_query", update={"decision": step})
        elif step == "decompose":
            return Command(goto="decompose_query", update={"decision": step})

# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("init_state", init_state)
router_builder.add_node("single_query", single_query)
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("decompose_query", decompose_query)
router_builder.add_node("llm_call", llm_call)
router_builder.add_node("synthesizer", synthesizer)
router_builder.add_node("summarizer", summarizer)
router_builder.add_node("review_answer", review_answer)

# Add edges to connect nodes
router_builder.add_edge(START, "init_state")
router_builder.add_edge("init_state", "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "single_query": "single_query",
        "decompose_query": "decompose_query",
    },
)
router_builder.add_edge("single_query", "review_answer")
router_builder.add_conditional_edges(
    "decompose_query", assign_workers, ["llm_call"]
)
router_builder.add_edge("llm_call", "synthesizer")
router_builder.add_edge("synthesizer", "summarizer")
router_builder.add_edge("summarizer", "review_answer")
router_builder.add_edge("review_answer", END)

# Compile the workflow
checkpointer = InMemorySaver()
graph = router_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # Invoke
    config = {"configurable": {"thread_id": "rag-single-or-decom-human-123"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "AI 트렌드 2025"}]}, config=config)
    print(result["__interrupt__"])

    # resumed = router_app.invoke(Command(resume={
    #     "action": "edit", "query": "AI 트렌드 2025와 AI 기술자 임금 동향 설명해줘. 여러 쿼리로 나누어서 정보를 찾아줘"
    #     }), config=config)
    # print(resumed["messages"][-1].content)
    resumed = graph.invoke(Command(resume={"action": "reject", "step": "decompose"}), config=config)
    print(resumed["__interrupt__"])