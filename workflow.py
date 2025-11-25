from typing import TypedDict
from graph.parser import parser
from graph.chunker import chunker
from graph.reranker import reranker
from graph.gen_data import data_generater
from graph.bot_answer import bot_answer
from graph.human_feedback import human_feedback
from rag import get_app
from utils.utils import format_context
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_upstage import ChatUpstage
from langchain_core.documents import Document

class State(TypedDict):
    docs: list[Document]
    markdown: str
   
# OpenAI LLM 초기화 (temperature=0: 결정적 답변 생성)
# llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)
llm = ChatUpstage(model="solar-pro2", temperature=0.0)

workflow = StateGraph(State)
workflow.add_node("parser", parser)
workflow.add_node("chunker", chunker)
workflow.add_node("reranker", reranker)
workflow.add_node("llm", get_app())
workflow.add_node("data_generater", data_generater)
workflow.add_node("bot_answer", bot_answer)
workflow.add_node("human_feedback", human_feedback)