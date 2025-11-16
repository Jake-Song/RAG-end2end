from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle

from config import output_path_prefix

# 토큰화 함수를 생성
def kiwi_tokenize(text):
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    return [token.form for token in kiwi.tokenize(text)]

def create_retriever(split_documents, embeddings, kiwi=False):
   
        
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    if kiwi:
        bm25_retriever = BM25Retriever.from_documents(split_documents, preprocess_func=kiwi_tokenize)
    else:
        bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = 1
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3],
    )
    return ensemble_retriever

def save_retriever(split_documents, embeddings):
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def load_retriever(split_documents, embeddings, kiwi=False):
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True  # needed in newer versions
    )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    if kiwi:
        bm25_retriever = BM25Retriever.from_documents(split_documents, preprocess_func=kiwi_tokenize)
    else:
        bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = 1
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3],
    )
    return ensemble_retriever

def main():
    with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)

    # embeddings = OpenAIEmbeddings()    
    embeddings = UpstageEmbeddings(model="embedding-passage")
    _ = create_retriever(split_documents, embeddings, kiwi=True)
    save_retriever(split_documents, embeddings)
    print("✅ 모든 작업 완료")
if __name__ == "__main__":
    main()