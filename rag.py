from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle

from scripts.retriver import ensemble_retriever
from dotenv import load_dotenv
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("Langgraph")

def main():
    with open("outputs/split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)

query = "what are risks and challenges of Korea in global economy?"
ensemble_result = ensemble_retriever.invoke(query)

if __name__ == "__main__":
    main()