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
import pickle
from pathlib import Path
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent

llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0.0)
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

with open(project_root / "datasets" / "hotpotqa.pkl", "rb") as f:
    split_documents = pickle.load(f)

def cohere_rerank_internal(query: str, retrieved_docs: list[Document], top_n: int = 20) -> list[Document]:
    rerank_response = co.rerank(
        model="rerank-v4.0-fast", query=query, documents=[doc.page_content for doc in retrieved_docs], top_n=top_n
    )
    return [retrieved_docs[result.index] for result in rerank_response.results]

@tool
def retriever(query: str) -> list[Document]:
    """Retrieve documents from the vector database.

    Args:
        query: The query to retrieve documents from the vector database.
    """
    embeddings = UpstageEmbeddings(model="embedding-passage")
    bm25_retriever, faiss_retriever = load_retriever(split_documents, embeddings, db_name="hotpotqa_100", kiwi=False, search_k=100)
    retrieved_docs_faiss = faiss_retriever.invoke(query)
    retrieved_docs_bm25 = bm25_retriever.invoke(query)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=100)
    reranked_docs = cohere_rerank_internal(query, rrf_docs, top_n=5)
    context = format_context(reranked_docs)

    return {"documents": reranked_docs, "context": context}

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
        answer: LLM answer
        documents: list of documents
    """
    question: str
    answer: str
    documents: Annotated[List[Document], "Documents"]
    filtered_documents: Annotated[List[Document], add]
    chunk_ids: Annotated[List[int], "Chunk IDs"]
    context: str    

class WorkerState(TypedDict):
    question: str
    document: Document
    filtered_documents: Annotated[List[Document], add]

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
    chunk_ids = state["chunk_ids"]
    # RAG answer
   
    GENERATE_PROMPT = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context}
    """
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    answer = llm.invoke([{"role": "user", "content": prompt}])
    return {
        "context": context, 
        "question": question, 
        "answer": answer.content,        
        "chunk_ids": chunk_ids,
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
        return {"filtered_documents": [document]}
          

def synthesizer(state: State):
    """Synthesize full answer from queries"""
    # List of completed sections
    filtered_documents = state["filtered_documents"]
    context = format_context(filtered_documents)  
    chunk_ids = [d.metadata["chunk_id"] for d in filtered_documents]
    return {"context": context, "filtered_documents": Overwrite([]), "chunk_ids": chunk_ids}

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

def decide_to_generate(state: State) -> Literal["transform_query", "generate"]:
    
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        return "generate"

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in answer answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def grade_answer_v_documents_and_question(
    state: State
    ) -> Literal["useful", "not useful"]:
   
    question = state["question"]
    answer = state["answer"]
    
    ANSWER_PROMPT = """
        You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer addresses the question.
        Here is the question: {question} \n
        Here is the answer: {answer} \n
    """

    prompt = ANSWER_PROMPT.format(question=question, answer=answer)
    answer_grader = llm.with_structured_output(GradeAnswer)

    # Check question-answering
    score = answer_grader.invoke([{"role": "user", "content": prompt}])
    answer_grade = score.binary_score
    if answer_grade == "yes":
        return "useful"
    else:
        return "not useful"
  

workflow = StateGraph(State)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_document", grade_document)  # grade document
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("synthesizer", synthesizer)  # synthesize

# Build graph
workflow.add_edge(START, "retrieve")
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
    grade_answer_v_documents_and_question,
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

    inputs = [{"question": question} for question in questions]

    results = app.batch(inputs, config)

    return results

if __name__ == "__main__":

    from pprint import pprint
    # Run
    input = {
        "question": "What was the duck character's name in the Disney cartoon with the music that sounded like The Mexican Hat Dance?"
    }
    for output in app.stream(input, stream_mode="updates", config={"configurable": {"thread_id": "1"}}):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final answer
    pprint(value["answer"])
