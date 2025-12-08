from pydantic import BaseModel, Field
from typing import List
from langchain_upstage import ChatUpstage
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
import operator
from rag_basic.rag import rag_bot_graph
from langchain.messages import AnyMessage
from langgraph.graph import add_messages

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
    messages: Annotated[list[AnyMessage], add_messages]


# Worker state
class WorkerState(TypedDict):
    query: Query
    completed_queries: Annotated[list, operator.add]



# Nodes
def init_state(state: State):
    return {"question": state["messages"][-1].content}

def orchestrator(state: State):
    """Orchestrator that generates queries for the question"""
    question = state["question"]
    # Generate queries
    queries = planner.invoke(
        [
            SystemMessage(content="Generate atomic queries for the question."),
            HumanMessage(content=f"Here is the question: {question}")
        ]
    )

    return {"question": question, "queries": queries.queries}

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
    })

    if decision["action"] == "approve":
        return Command(goto=END)
    elif decision["action"] == "reject":
        question = decision["query"]
        return Command(goto="orchestrator", update={"question": question})
    

# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("init_state", init_state)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)
orchestrator_worker_builder.add_node("review_answer", review_answer)
orchestrator_worker_builder.add_node("summarizer", summarizer)
# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "init_state")
orchestrator_worker_builder.add_edge("init_state", "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", "summarizer")
orchestrator_worker_builder.add_edge("summarizer", "review_answer")
orchestrator_worker_builder.add_edge("review_answer", END)

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
# Compile the workflow
graph = orchestrator_worker_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # Invoke
    config = {"configurable": {"thread_id": "rag-multi-queries-human-123"}}
    initial = graph.invoke({"messages": [{"role": "user", "content": "AI 트렌드 2025"}]}, config=config)
    if "__interrupt__" in initial:
        print(initial["__interrupt__"][0].value)
  
    # Resume with the decision; True routes to proceed, False to cancel
    resumed = graph.invoke(Command(resume={"action": "reject", "query": "AI 트렌드 2025와 AI 기술자 임금 동향 설명해줘. 여러 쿼리로 나누어서 정보를 찾아줘"}), config=config)
    print(resumed["messages"][-1].content)
   
    # resumed = graph.invoke(Command(resume=True), config=config)
    # print(resumed["messages"][-1].content)