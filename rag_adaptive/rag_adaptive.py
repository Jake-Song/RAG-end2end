from langgraph.graph import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
from operator import add
from langgraph.types import Send, Overwrite
from langchain.tools import tool
from langchain.messages import ToolMessage, AnyMessage
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Literal, List, Annotated, TypedDict
from scripts.retrieve import load_retriever
from utils.utils import format_context
from reranker.rrf import ReciprocalRankFusion
from config import output_path_prefix
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=6)
    context = format_context(rrf_docs)

    return {"documents": rrf_docs, "context": context}

tools = [retriever]
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
    documents: Annotated[List[Document], "Documents"]
    filtered_documents: Annotated[List[Document], add]
    context: str    

class WorkerState(TypedDict):
    question: str
    document: Document
    filtered_documents: Annotated[List[Document], add]

class Route(BaseModel):
    step: Literal["inference", "vectorstore"] = Field(
        ..., description="Given a user question choose to route it to inference or a vectorstore."
    )

router = llm.with_structured_output(Route)

ROUTE_PROMPT = """
        You are an expert at routing a user question to a vectorstore or inference.
        The vectorstore contains documents related to 
        The following context is a summary report published by the Software Policy & Research Institute (SPRi). 
        It discusses the findings of the original 'AI Index 2025' published by Stanford University.
        Use the vectorstore for questions on these topics. Otherwise, use inference.
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
    return {
        "context": result["context"], 
        "documents": result["documents"], 
        "question": question, 
        }

def retrieve_until_exhausted(state: State) -> State:
    question = state["question"]
    documents = state["documents"]
    while len(documents) > 0:
        result = retriever.invoke(question)
        documents.extend(result["documents"])
        return {
            "context": result["context"], 
            "documents": documents, 
            "question": question, 
        }

def generate(state: State) -> State:
    question = state["question"]
    context = state["context"]
    # RAG generation
    preamble = """
    The following context is a summary report published by the Software Policy & Research Institute (SPRi). 
    It discusses the findings of the original 'AI Index 2025' published by Stanford University.
    Distinguish clearly between the author of this summary and the author of the original report.
    """
    GENERATE_PROMPT = """
        You are an assistant for question-answering tasks. 
        {preamble}
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context}
    """
    prompt = GENERATE_PROMPT.format(question=question, context=context, preamble=preamble)
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

def grade_document(state: WorkerState) -> WorkerState:
    question = state["question"]
    document = state["document"]
    retrieval_grader = llm.with_structured_output(GradeDocuments)

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {document} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    prompt = GRADE_PROMPT.format(question=question, document=document.page_content)
    score = retrieval_grader.invoke([{"role": "user", "content": prompt}])
    grade = score.binary_score
    if grade == "yes":
        logger.info("---GRADE: DOCUMENT RELEVANT---")
        return {"filtered_documents": [document]}
    else:
        logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
        

def synthesizer(state: State):
    """Synthesize full answer from queries"""
    logger.info("---SYNTHESIZER---")
    # List of completed sections
    filtered_documents = state["filtered_documents"]
    logger.info(f"Filtered Documents: {[d.metadata['page'] for d in filtered_documents]}")

    context = format_context(filtered_documents)  

    return {"context": context, "filtered_documents": Overwrite([])}

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off query writing in parallel via Send() API
    return [Send("grade_document", {"question": state["question"], "document": d}) for d in state["documents"]]

def transform_query(state: State) -> State:
    question = state["question"]
   
    REWRITE_PROMPT = """You are an expert query optimizer for semantic vector search retrieval.

        Your task is to rewrite the user's question into a format that will retrieve the most relevant documents from a vectorstore.

        Follow these guidelines:
        1. Extract and emphasize key entities, technical terms, and proper nouns (e.g., "AI Index 2025", "Stanford HAI")
        2. Expand abbreviations and acronyms (e.g., "AI" → "Artificial Intelligence (AI)")
        3. Remove filler words, pronouns, and conversational language
        4. Include synonyms or related terms that might appear in documents
        5. Focus on noun phrases and factual keywords rather than question format
        6. Keep the core semantic meaning intact

        Examples:
        - Input: "AI가 일자리에 어떤 영향을 미치나요?"
        Output: "인공지능 AI 고용 일자리 영향 노동시장 변화"
        
        - Input: "2024년 AI 투자는 얼마나 됐어?"
        Output: "2024년 인공지능 AI 글로벌 투자 금액 규모 투자액"

        Here is the initial question:
        {question}

        Output only the rewritten query, nothing else."""
    
    prompt = REWRITE_PROMPT.format(question=question)
    better_question = llm.invoke([{"role": "user", "content": prompt}])

    return {"question": better_question.content}

def inference(state: State) -> State:
    question = state["messages"][-1].content
    logger.info("---INFERENCE---")
    response = llm.invoke([{"role": "user", "content": question}])

    return {"question": question, "messages": [response]}

def route_decision(state: State) -> Literal["inference", "retrieve"]:
    # Return the node name you want to visit next
    if state["decision"] == "inference":
        return "inference"
    elif state["decision"] == "vectorstore":
        return "retrieve"

def decide_to_generate(state: State) -> Literal["transform_query", "generate"]:
    
    logger.info("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        logger.info("---DECISION: GENERATE---")
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
    ) -> Literal["useful", "not useful"]:
   
    question = state["question"]
    generation = state["generation"]
    
    ANSWER_PROMPT = """
        You are a grader assessing whether an generation addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the generation addresses the question.
        Here is the question: {question} \n
        Here is the generation: {generation} \n
    """

    prompt = ANSWER_PROMPT.format(question=question, generation=generation)
    answer_grader = llm.with_structured_output(GradeAnswer)

    # Check question-answering
    logger.info("---GRADE GENERATION vs QUESTION---")
    score = answer_grader.invoke([{"role": "user", "content": prompt}])
    answer_grade = score.binary_score
    if answer_grade == "yes":
        logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"
  

workflow = StateGraph(State)

# Define the nodes
workflow.add_node("llm_call_router", llm_call_router)
workflow.add_node("inference", inference)  # inference
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_document", grade_document)  # grade document
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("synthesizer", synthesizer)  # synthesize

# Build graph
workflow.add_edge(START, "llm_call_router")
workflow.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "inference": "inference",
        "retrieve": "retrieve",
    },
)
workflow.add_edge("inference", END)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "retrieve", assign_workers, ["grade_document"]
)
workflow.add_edge("grade_document", "synthesizer")
workflow.add_conditional_edges(
    "synthesizer",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

def rag_bot_batch(questions: list[str]) -> list[dict]:
    from langchain_core.runnables import RunnableConfig
    import uuid

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})

    inputs = [{"messages": [{"role": "user", "content": question}]} for question in questions]

    results = app.batch(inputs, config)

    return results

if __name__ == "__main__":

    from pprint import pprint
    # Run
    input = {
        "messages": [
            {
                "role": "user",
                "content": "AI Index 2025에 따르면 책임있는 AI 관련 논문 수는 2023년과 2024년에 각각 몇 편이며 증가율은 얼마인가요?"
            }
        ]
    }
    for output in app.stream(input, stream_mode="updates", config={"configurable": {"thread_id": "1"}}):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    value["messages"][-1].pretty_print()
