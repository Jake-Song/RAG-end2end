from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import output_path_prefix
import pickle

def cross_encoder_reranker(query: str) -> list[Document]:
    """
    Cohere Reranker를 사용하여 문서를 재정렬합니다.
    """
    # 모델 초기화
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    embeddings = UpstageEmbeddings(model="embedding-passage")
    retriever = load_retriever(embeddings, search_k=10)

    # 상위 3개의 문서 선택
    compressor = CrossEncoderReranker(model=model, top_n=2)
    
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
    with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)
    embeddings = UpstageEmbeddings(model="embedding-passage")
    save_retriever(split_documents, embeddings)

    query = "G7, 히로시마 AI 프로세스를 통해 합의한 국제 행동강령"
    compressed_docs = cross_encoder_reranker(query)
    print(compressed_docs)
    
