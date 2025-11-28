from pydantic import BaseModel, Field
from typing import List
from langchain_upstage import ChatUpstage
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
import operator
from rag import rag_bot_graph
from typing_extensions import Literal
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
    step: Literal["single", "multi"] = Field(
        None, description="The next step in the routing process"
    )

router = llm.with_structured_output(Route)
planner = llm.with_structured_output(Queries)



# Graph state
class State(TypedDict):
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
    completed_queries: Annotated[list, operator.add]


# Nodes
def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the question to single query or orchestrator based on the user's request."
            ),
            HumanMessage(content=state["question"]),
        ]
    )

    return {"decision": decision.step}

# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "single":
        return "single_query"
    elif state["decision"] == "multi":
        return "orchestrator"

# Nodes
def single_query(state: State):
    result = rag_bot_graph(state["question"])
    return {"final_answer": result.content}

def orchestrator(state: State):
    """Orchestrator that generates queries for the question"""

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


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off query writing in parallel via Send() API
    return [Send("llm_call", {"query": q}) for q in state["queries"]]


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("single_query", single_query)
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("orchestrator", orchestrator)
router_builder.add_node("llm_call", llm_call)
router_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "single_query": "single_query",
        "orchestrator": "orchestrator",
    },
)
router_builder.add_edge("single_query", END)
router_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
router_builder.add_edge("llm_call", "synthesizer")
router_builder.add_edge("synthesizer", END)

# Compile the workflow
router_app = router_builder.compile()

if __name__ == "__main__":
    # Invoke
    state = router_app.invoke({"question": "AI 트렌드 2025와 AI 기술자 임금 동향 설명해줘. 여러 쿼리로 나누어서 정보를 찾아줘"})
    print(state)