from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
from dotenv import load_dotenv
load_dotenv()

def create_retriever(split_documents, embeddings):
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = 3
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3],
    )
    return ensemble_retriever

def save_retriever(split_documents, embeddings):
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def load_retriever(split_documents):
    vectorstore = FAISS.load_local(
        "faiss_index", 
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True  # needed in newer versions
    )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = 3
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3],
    )
    return ensemble_retriever

def main():
    with open("outputs/split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)

    embeddings = OpenAIEmbeddings()    

    _ = create_retriever(split_documents, embeddings)
    save_retriever(split_documents, embeddings)
  
if __name__ == "__main__":
    main()