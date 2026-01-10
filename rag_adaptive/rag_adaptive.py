from langgraph.graph import add_messages
from langchain.tools import tool
from langchain.messages import ToolMessage, AnyMessage
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Literal, List, Annotated, TypedDict
from scripts.retrieve import load_retriever
from utils.utils import format_context
from reranker.rrf import ReciprocalRankFusion
from config import output_path_prefix
import pickle

with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)

@tool
def retriever(query: str) -> list[Document]:
    """Retrieve documents from the vector database.

    Args:
        query: The query to retrieve documents from the vector database.
    """
    embeddings = UpstageEmbeddings(model="embedding-passage")
    bm25_retriever, faiss_retriever = load_retriever(split_documents, embeddings, kiwi=False, search_k=10)
    retrieved_docs_faiss = faiss_retriever.invoke(query)
    retrieved_docs_bm25 = bm25_retriever.invoke(query)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=4)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}

web_search_tool = TavilySearch(max_results=3)

@tool
def web_search(query: str) -> list[Document]:
    """Search the web for information.
    
    Args:
        query: The query to search the web for.
    """

    docs = web_search_tool.invoke({"query": query})['results']
    context = "\n".join([d["content"] for d in docs])
    web_results = [Document(page_content=d["content"]) for d in docs]

    return {"documents": web_results, "context": context}

tools = [retriever, web_search]
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result, "documents": observation["documents"], "context": observation["context"]}

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    generation: str
    decision: str
    documents: List[Document]
    context: str    


class Route(BaseModel):
    step: Literal["vectorstore", "web_search"] = Field(
        ..., description="Given a user question choose to route it to web search or a vectorstore."
    )

router = llm.with_structured_output(Route)

ROUTE_PROMPT = """
        You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to 
        The following context is a summary report published by the Software Policy & Research Institute (SPRi). 
        It discusses the findings of the original 'AI Index 2025' published by Stanford University.
        Use the vectorstore for questions on these topics. Otherwise, use web-search.
        question: {question}
    """

# nodes
def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    question = state["messages"][-1].content
    # Run the augmented LLM with structured output to serve as routing logic
    prompt = ROUTE_PROMPT.format(question=question)
    decision = router.invoke(prompt)
   
    return {"decision": decision.step, "question": question}

def retrieve(state: State) -> State:
    question = state["question"]

    # Retrieval
    result = retriever.invoke(question)
    return {"context": result["context"], "documents": result["documents"], "question": question}

def generate(state: State) -> State:
    question = state["question"]
    context = state["context"]
    # RAG generation
    GENERATE_PROMPT = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context}
    """
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    generation = llm.invoke([{"role": "user", "content": prompt}])
    return {
        "context": context, 
        "question": question, 
        "generation": generation,
        "messages": [generation]
    }

class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: State) -> State:
  
    question = state["question"]
    documents = state["documents"]
    retrieval_grader = llm.with_structured_output(GradeDocuments)

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {document} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    
    # Score each doc
    filtered_docs = []
    for d in documents:
        prompt = GRADE_PROMPT.format(question=question, document=d.page_content)
        score = retrieval_grader.invoke([{"role": "user", "content": prompt}])
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state: State) -> State:
    question = state["question"]
   
    REWRITE_PROMPT = """
        You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
        Here is the initial question: \n\n {question} \n Formulate an improved question.
    """
    
    prompt = REWRITE_PROMPT.format(question=question)
    better_question = llm.invoke([{"role": "user", "content": prompt}])

    return {"question": better_question}


def run_web_search(state: State) -> State:
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})['results']
    context = "\n".join([d["content"] for d in docs])
    web_results = [Document(page_content=d["content"]) for d in docs]

    return {"documents": web_results, "context": context}


def route_decision(state: State) -> Literal["run_web_search", "retrieve"]:
    # Return the node name you want to visit next
    if state["decision"] == "web_search":
        return "run_web_search"
    elif state["decision"] == "vectorstore":
        return "retrieve"

def decide_to_generate(state: State) -> Literal["transform_query", "generate"]:
    
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def grade_generation_v_documents_and_question(
    state: State
    ) -> Literal["not supported", "useful", "not useful"]:
   
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    context = state["context"]
    generation = state["generation"]
    
    hallucination_grader = llm.with_structured_output(GradeHallucinations)
    HALLUCINATION_PROMPT = """
        You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the generation is grounded in / supported by the set of facts.
        Here is the LLM generation: \n\n {generation} \n\n"
        Here is the set of facts: \n\n {context} \n\n"
    """

    prompt = HALLUCINATION_PROMPT.format(generation=generation, context=context)
    score = hallucination_grader.invoke([{"role": "user", "content": prompt}])
    grade = score.binary_score

    ANSWER_PROMPT = """
        You are a grader assessing whether an generation addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the generation addresses the question.
        Here is the question: {question} \n
        Here is the generation: {generation} \n
    """

    prompt = ANSWER_PROMPT.format(question=question, generation=generation)
    answer_grader = llm.with_structured_output(GradeAnswer)
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke([{"role": "user", "content": prompt}])
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


from langgraph.graph import END, StateGraph, START

workflow = StateGraph(State)

# Define the nodes
workflow.add_node("llm_call_router", llm_call_router)
workflow.add_node("run_web_search", run_web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_edge(START, "llm_call_router")
workflow.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "run_web_search": "run_web_search",
        "retrieve": "retrieve",
    },
)
workflow.add_edge("run_web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

if __name__ == "__main__":

    from pprint import pprint
    # Run
    # inputs = {"messages": [{"role": "user",
    #     "content": "What player at the Bears expected to draft first in the 2024 NFL draft?"
    # }]}
    # for output in app.stream(inputs, stream_mode="updates"):
    #     for key, value in output.items():
    #         # Node
    #         pprint(f"Node '{key}':")
    #         pprint(f"Value: {value}")
    #         # Optional: print full state at each node
    #         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    #     pprint("\n---\n")

    # # Final generation
    # pprint(value["generation"])

    # Run
    inputs = {"messages": [{"role": "user",
        "content": "AI Index 2025 연례보고서의 발행 기관과 발행 시기는 언제인가?"
    }]}
    for output in app.stream(inputs, stream_mode="updates"):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    value["messages"][-1].pretty_print()
