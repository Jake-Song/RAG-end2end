from scripts.retrieve import load_retriever
from reranker.rrf import ReciprocalRankFusion
from config import output_path_prefix
import pickle
from langchain_upstage import UpstageEmbeddings

with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
    split_documents = pickle.load(f)

embeddings = UpstageEmbeddings(model="embedding-passage")

_, bm25_retriever, faiss_retriever = load_retriever(split_documents, embeddings, kiwi=True, search_k=10)

def rerank_RRF(question, cutoff):
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(faiss_retriever.invoke(question), bm25_retriever.invoke(question), cutoff)
    return rrf_docs

if __name__ == "__main__":
    question = "본문에서 제시된 2022년 개정 교육과정 관련 3가지 핵심 사실은 무엇인가요?"
    cutoff = 3
    print(rerank_RRF(question, cutoff))