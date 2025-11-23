"""
합성 데이터 생성
간단한 싱글턴 데이터 생성 방식
GPT-5 모델 사용
"""

from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from config import output_path_prefix
from langchain.schema import Document
import random
random.seed(42)

def pick_random_pages(docs) -> str:
    return random.choice(docs).metadata["page"]

def pick_random_page_range(docs, min_doc: int = 0, max_doc: int = 10):
    # Filter docs where the page number is within the range [min_page, max_page]
    candidates = [
        doc for doc in docs 
        if min_doc <= doc.metadata.get("page", -1) <= max_doc
    ]
    
    if not candidates:
        return None
        
    return random.choice(candidates).metadata["page"]


class SyntheticData(BaseModel):
    """Synthetic data with details."""
    query: str = Field(..., description="The query of the data")
    answer: str = Field(..., description="The answer of the data")
    
def generate_prompt(docs: list[Document], query_count: int = 10) -> list[list[dict]]:
    system_prompt = "You are a careful dataset generator for RAG. Only answer from the provided passage."
    
    user_prompt = f"""
                Task: write 1 QA pair whose answer relates to following documents.
                Generate query and answers in Korean. include atomic facts in the query
                DO NOT GENERATE QUERY LIKE FOLLOWING : 
                - "다음 문서에서 확인되는 3가지 사실은 무엇인가?" 
                - "이 문서에서 도출할 수 있는 3가지 핵심 사실은 무엇인가?" 
                - "다음 문서에서 안전 테스트 관련 핵심 사실 3가지를 요약하시오."
                
                Generate query like following :
                - "AI 기술자의 임금 변화와 주요 행사는 무엇인가?"
                - "G7은 2023년 어떤 국제 행동강령에 합의했나요?
                - "FTC가 저작권청의 질의공고에 대해 제시한 주요 관심사는 무엇인가?"
            """

    queries = []
    choices = [pick_random_page_range(docs, min_doc=0, max_doc=len(docs)-1) for _ in range(query_count*2)]
    pairs = [tuple(choices[i:i+2]) for i in range(0, len(choices), 2)]
    
    for pair in pairs:
       
        doc1 = next(doc.page_content for doc in docs if doc.metadata["page"] == pair[0])
        doc2 = next(doc.page_content for doc in docs if doc.metadata["page"] == pair[1])
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", 
            "content": 
            "## documents : " + "### Document 1" + "\n" + doc1 + "\n" + "### Document 2" + "\n" + doc2}
        ]
        queries.append(prompt)
    return queries, pairs

def generate_data(llm: ChatUpstage, queries: list[list[dict]]) -> list[SyntheticData]:
    model_with_structure = llm.with_structured_output(SyntheticData)
    responses = model_with_structure.batch(queries)
    return responses

def save_data(responses: list[SyntheticData], pairs: list[tuple[int, int]]) -> None:
    arr = []
    
    for r, pair in zip(responses, pairs):
           
        obj = {
            "query": r.query,
            "answer": r.answer,
            "page_number": [pair[0], pair[1]],            
        }
        arr.append(obj)
    
    df = pd.DataFrame(arr)
    df.to_csv(f"{output_path_prefix}_synthetic.csv", index=True)

def main():
    with open(f"{output_path_prefix}_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    # llm = ChatOpenAI(model_name="gpt-5", temperature=0)
    llm = ChatUpstage(model="solar-pro2", temperature=0.0, reasoning_effort="high")

    queries, pairs = generate_prompt(docs, query_count=50)
    print("페이지 짝", pairs)
    print("쿼리 생성")
    print(f"쿼리 개수: {len(queries)}")

    responses = generate_data(llm, queries)
    print("응답", responses)
    print(f"응답 개수: {len(responses)}")
    print("응답 생성")
    
    save_data(responses, pairs)
    print("데이터 저장")
    print("✅ 모든 작업 완료")

if __name__ == "__main__":
    main()