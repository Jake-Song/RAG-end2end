from pydantic import BaseModel, Field
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from hyde import hyde
from multi_step_query import multi
from typing import Annotated, TypedDict
from typing_extensions import Literal


class State(TypedDict):
    question: Annotated[str, "Question"]
    answer: Annotated[str, "Answer"]
    decision: Annotated[str, "Decision"]
    query_count: Annotated[int, "Query Count"]

class Route(BaseModel):
    node: Literal["hyde", "multi"] = Field(
        None, description="The next step in the routing process"
    )
ROUTE_PROMPT = """
You are a helpful assistant that route the question to the appropriate node.

You are given a question.

You need to route the question to the appropriate node.
# 규칙
- 좁은 범위의 질문이나 특정 목적의 질문은 hyde 노드로 라우팅
- 넓은 범위의 질문이나 열린 결말의 질문은 multi 노드로 라우팅


Here is the question:

{question}
"""


llm = ChatUpstage(model="solar-pro2", temperature=0)
router = llm.with_structured_output(Route)
# Nodes
def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    prompt = ROUTE_PROMPT.format(question=state["question"])
    decision = router.invoke(prompt)
   
    return {"decision": decision.node}

def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "hyde":
        return "hyde"
    elif state["decision"] == "multi":
        return "multi"

builder = StateGraph(State)
builder.add_node("llm_call_router", llm_call_router)
builder.add_node("hyde", hyde)
builder.add_node("multi", multi)

builder.add_edge(START, "llm_call_router")
builder.add_conditional_edges(
    "llm_call_router",
     route_decision, 
     {"hyde": "hyde", "multi": "multi"}
    )
builder.add_edge("hyde", END)
builder.add_edge("multi", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    from pprint import pprint
    config = {"configurable": {"thread_id": "1"}}
    for chunk in app.stream({"question": "AI 트렌드는 무엇인가?"}, stream_mode="updates", config=config):
        pprint(chunk)
    for chunk in app.stream({"question": "상위 AI 논문 인용 순위 3개는 무엇인가?"}, stream_mode="updates", config=config):
        pprint(chunk)
    