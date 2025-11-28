from pydantic import BaseModel, Field
from typing import List
from langchain_upstage import ChatUpstage
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
import operator
from rag import rag_bot_graph

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


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Queries)


# Graph state
class State(TypedDict):
    question: str  # Report topic
    queries: list[Query]  # List of report sections
    completed_queries: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_answer: str  # Final report


# Worker state
class WorkerState(TypedDict):
    query: Query
    completed_queries: Annotated[list, operator.add]


# Nodes
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

from langgraph.types import interrupt, Command

def review_answer(state: State):
    # Ask a reviewer to edit the generated content
    decision = interrupt({
        "question": "Review and approve this answer",
        "query": state["queries"],
        "final_answer": state["final_answer"],
    })

    return Command(goto="orchestrator" if not decision else "END")

# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)
orchestrator_worker_builder.add_node("review_answer", review_answer)
# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", "review_answer")
orchestrator_worker_builder.add_edge("review_answer", END)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # Invoke
    config = {"configurable": {"thread_id": "rag-multi-queries-human-123"}}
    initial = orchestrator_worker.invoke({"question": "AI 트렌드 2025"}, config=config)
    
    print(initial["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]

    # Resume with the decision; True routes to proceed, False to cancel
    resumed = orchestrator_worker.invoke(Command(resume=False), config=config)
    print(resumed["__interrupt__"])
    resumed = orchestrator_worker.invoke(Command(resume=True), config=config)
    print(resumed)