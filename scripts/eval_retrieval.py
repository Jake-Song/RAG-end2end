from langchain_upstage import UpstageEmbeddings
from reranker.rrf import ReciprocalRankFusion
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pickle
import pandas as pd
import cohere
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

with open("outputs/SPRI_2025_output_split_documents.pkl", "rb") as f:
    split_documents = pickle.load(f)

embeddings = UpstageEmbeddings(model="embedding-passage")
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def load_retriever(db_name: str, search_k: int = 10):
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = search_k
    vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings,
            db_name,
            allow_dangerous_deserialization=True  # needed in newer versions
        )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})

    return bm25_retriever, faiss_retriever


def retrieve_document(question: str, search_k: int = 10, top_k: int = 10) -> list[Document]:
    bm25_retriever, faiss_retriever = load_retriever("SPRI_2025_contextual", search_k=search_k)
    retrieved_docs_faiss = faiss_retriever.invoke(question)
    retrieved_docs_bm25 = bm25_retriever.invoke(question)
    retrieved_docs_faiss = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_faiss)
    retrieved_docs_bm25 = ReciprocalRankFusion.calculate_rank_score(retrieved_docs_bm25)
    retrieved_docs = retrieved_docs_faiss + retrieved_docs_bm25
    rrf_docs = ReciprocalRankFusion.get_rrf_docs(retrieved_docs, cutoff=top_k)
    return rrf_docs


def rerank_document(df: pd.DataFrame, retrieved_docs: list[Document], top_n: int = 5) -> list[Document]:
    rerank_results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Reranking documents"):
        query = row["query"]
        documents = [doc.page_content for doc in retrieved_docs[i]]
        rerank_response = co.rerank(
            model="rerank-multilingual-v3.0", query=query, documents=documents, top_n=top_n
        )
        rerank_results.append(rerank_response.results)

    reranked_chunks_arr = []
    for idx, retrieved_doc in enumerate(retrieved_docs):
        reranked_chunks = []
        for result in rerank_results[idx]:
            reranked_chunks.append(retrieved_doc[result.index])
        reranked_chunks_arr.append(reranked_chunks)

    return reranked_chunks_arr

def recall(df: pd.DataFrame, retrieved_docs: list[str]) -> dict:
    true_positives = 0
    false_negatives = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Calculating recall"):
        # 중복 페이지 제거
        # reference_page_number = list({int(page) for page in row["page_number"].strip("[]").split(",")})
        reference_chunk_id = row["chunk_id"]
        retrieved_chunk_id = [doc.metadata["chunk_id"] for doc in retrieved_docs[i]]
       
        if reference_chunk_id in retrieved_chunk_id:
            true_positives += 1
        else:
            print("index: ", i)
            print(row["query"])
            print(row["chunk_id"])
            print(retrieved_chunk_id)
            false_negatives += 1

    score = true_positives / (true_positives + false_negatives)
   
    return {"score": score, "pass_at_n": true_positives}

def evaluate_retrieval(dataset: pd.DataFrame, rerank: bool = False):
    print(f"{'=' * 60}")
    print(f"Evaluation Results: Contextual Embeddings")
    print(f"{'=' * 60}\n")
    k_values = [5, 10, 20]
    results = {}
    for k in k_values:
        print(f"Evaluating Pass@{k} with context retrieval...")
       
        retrieved_docs = [retrieve_document(
            dataset.iloc[i]["query"], search_k=k, top_k=k
        ) for i in tqdm(range(len(dataset)), desc=f"Retrieving documents Pass@{k}")]
    
        if rerank:
            reranked_docs = rerank_document(dataset, retrieved_docs, top_n=5)
            results[k] = recall(dataset, reranked_docs)
        else:
            results[k] = recall(dataset, retrieved_docs)
    
    # Print summary table
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Pass Rate':<15} {'Score':<15}")
    print(f"{'-' * 60}")
    for k in k_values:
        pass_rate = f"{results[k]['pass_at_n']:.2f}%"
        score = f"{results[k]['score']:.4f}"
        print(f"{'Pass@' + str(k):<15} {pass_rate:<15} {score:<15}")
    print(f"{'=' * 60}\n")
   

if __name__ == "__main__":
    df = pd.read_csv("outputs/SPRI_2025_output_synthetic_single_chunk.csv")

    evaluate_retrieval(df, rerank=False)




