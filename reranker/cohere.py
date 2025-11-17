from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from config import output_path_prefix
import pickle
from langchain.schema import Document

with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
    split_documents = pickle.load(f)

def cohere_reranker(query: str) -> list[Document]:
    """
    Cohere Reranker를 사용하여 문서를 재정렬합니다.
    """

    retriever = load_retriever(CohereEmbeddings(model="embed-multilingual-v3.0"), search_k=10)

    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=2)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.invoke(query)

    return compressed_docs

def save_retriever(split_documents, embeddings):
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def load_retriever(embeddings, search_k=10):
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True  # needed in newer versions
    )
    return vectorstore.as_retriever(search_kwargs={"k": search_k})

if __name__ == "__main__":
    save_retriever(split_documents, CohereEmbeddings(model="embed-multilingual-v3.0"))
