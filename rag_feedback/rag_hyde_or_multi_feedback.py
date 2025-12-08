from pydantic import BaseModel, Field
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from rag_hyde_or_multi.rag_hyde import hyde
from rag_hyde_or_multi.rag_multi_query import multi
from typing import Annotated, TypedDict
from typing_extensions import Literal
from langchain.messages import AnyMessage
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    decision: Annotated[str, "Decision"]

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
def init_state(state: State):
    
    return {"question": state["messages"][-1].content}

def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    question = state["question"]
    # Run the augmented LLM with structured output to serve as routing logic
    prompt = ROUTE_PROMPT.format(question=question)
    decision = router.invoke(prompt)
   
    return {"decision": decision.node}

def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "hyde":
        return "hyde"
    elif state["decision"] == "multi":
        return "multi"

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
        node = decision["step"]
        if node == "hyde":
            return Command(goto="hyde", update={"decision": node})
        elif node == "multi":
            return Command(goto="multi", update={"decision": node})

builder = StateGraph(State)
builder.add_node("init_state", init_state)
builder.add_node("llm_call_router", llm_call_router)
builder.add_node("hyde", hyde)
builder.add_node("multi", multi)
builder.add_node("review_answer", review_answer)

builder.add_edge(START, "init_state")
builder.add_edge("init_state", "llm_call_router")
builder.add_conditional_edges(
    "llm_call_router",
     route_decision, 
     {"hyde": "hyde", "multi": "multi"}
    )
builder.add_edge("hyde", "review_answer")
builder.add_edge("multi", "review_answer")
builder.add_edge("review_answer", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    from pprint import pprint
    config = {"configurable": {"thread_id": "1"}}
    # for chunk in graph.stream({"messages": [{"role": "user", "content": "AI 트렌드는 무엇인가?"}]}, stream_mode="updates", config=config):
    #     pprint(chunk)
    # for chunk in graph.stream({"messages": [{"role": "user", "content": "상위 AI 논문 인용 순위 3개는 무엇인가?"}]}, stream_mode="updates", config=config):
    #     pprint(chunk)

     
    result = graph.invoke({"messages": [{"role": "user", "content": "AI 트렌드 2025"}]}, config=config)
    print(result["__interrupt__"])

    resumed = graph.invoke(Command(resume={"action": "reject", "step": "multi"}), config=config)
    print(resumed["__interrupt__"])

    resumed = graph.invoke(Command(resume={"action": "edit", "query": "AI 트렌드 2025와 AI 기술자 임금 동향 설명해줘."}), config=config)
    print(resumed["__interrupt__"])

    resumed = graph.invoke(Command(resume={"action": "approve"}), config=config)
    print(resumed["messages"][-1].content)
    
    