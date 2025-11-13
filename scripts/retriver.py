from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH10-Retriever")

def main():
    with open("outputs/split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = 3
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3],
    )

    vectorstore.save_local("faiss_index")
  
if __name__ == "__main__":
    main()