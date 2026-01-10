"""
합성 데이터 생성
멀티홉 데이터 생성 방식 - 여러 문서에서 정보를 조합해야 답할 수 있는 질문 생성
GPT-5 모델 사용
"""

from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from config import output_path_prefix
from langchain_core.documents import Document
import random
random.seed(42)


def pick_random_page_pair(docs: list[Document], num_pages: int = 2) -> list[int]:
    """Pick multiple random pages for multi-hop questions."""
    all_pages = list(set(doc.metadata["page"] for doc in docs))
    return random.sample(all_pages, min(num_pages, len(all_pages)))


class SyntheticData(BaseModel):
    """Synthetic data with details for multi-hop questions."""
    page_numbers: list[int] = Field(..., description="The page numbers used for the question")
    query: str = Field(..., description="The query of the data")
    answer: str = Field(..., description="The answer of the data")


def generate_prompt(docs: list[Document], query_count: int = 10) -> list[list[dict]]:
    system_prompt = "You are a careful dataset generator for multi-hop RAG questions. Only answer from the provided passages."
    
    user_prompt = """
        Task: Write 1 multi-hop QA pair that REQUIRES information from ALL provided documents to answer.
        The question should need reasoning across multiple documents - it should NOT be answerable from just one document.
        Generate query and answers in Korean.
        
        DO NOT GENERATE QUERY LIKE FOLLOWING:
        - "다음 문서에서 확인되는 3가지 사실은 무엇인가?" 
        - "이 문서에서 도출할 수 있는 3가지 핵심 사실은 무엇인가?" 
        - Questions that can be answered from just ONE document
        
        Good multi-hop examples:
        - "2023년 중국의 AI 특허 비중과 같은 해 미국의 AI 규제 입법 동향을 비교하면?"
        - "AI Index 보고서 발간 시작 연도와 2024년 글로벌 AI 투자액의 관계는?"
        - "G7의 AI 행동강령 합의 시점과 EU AI Act 통과 시점의 차이는?"
        - "AI 기술자의 임금 변화 추이와 AI 스타트업 투자 동향 사이의 연관성은?"
    """

    queries = []
    for _ in range(query_count):
        # Pick 2 pages for multi-hop
        page_numbers = pick_random_page_pair(docs, num_pages=2)
        selected_docs = [doc for doc in docs if doc.metadata["page"] in page_numbers]
        
        # Build context from multiple documents
        docs_content = ""
        page_nums_str = ""
        for i, doc in enumerate(selected_docs, 1):
            docs_content += f"### Document {i} (Page {doc.metadata['page']})\n{doc.page_content}\n\n"
            page_nums_str += f"{doc.metadata['page']}, "
        
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": f"## Documents:\n{docs_content}\n## Page Numbers: [{page_nums_str.rstrip(', ')}]"}
        ]
        queries.append(prompt)
    return queries


def generate_data(llm: ChatUpstage, queries: list[list[dict]]) -> list[SyntheticData]:
    model_with_structure = llm.with_structured_output(SyntheticData)
    responses = model_with_structure.batch(queries)
    return responses


def save_data(responses: list[SyntheticData]) -> pd.DataFrame:
    arr = []
    
    for r in responses:
        obj = {
            "query": r.query,
            "answer": r.answer,
            "page_numbers": r.page_numbers,
        }
        arr.append(obj)
    
    df = pd.DataFrame(arr)
    df.to_csv(f"{output_path_prefix}_synthetic_multi_hop.csv", index=False)
    return df


def main():
    with open(f"{output_path_prefix}_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)
    # llm = ChatUpstage(model="solar-pro2", temperature=0.0, reasoning_effort="high")

    queries = generate_prompt(docs, query_count=50)
    print("쿼리 생성")
    print(f"쿼리 개수: {len(queries)}")

    responses = generate_data(llm, queries)
    print("응답", responses)
    print(f"응답 개수: {len(responses)}")
    print("응답 생성")
    
    save_data(responses)
    print("데이터 저장")
    print("✅ 모든 작업 완료")


if __name__ == "__main__":
    main()
