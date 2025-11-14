from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from config import output_path_prefix
from dotenv import load_dotenv
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("synthetic")

class SyntheticData(BaseModel):
    """Synthetic data with details."""
    query: str = Field(..., description="The query of the data")
    answers: str = Field(..., description="The answers of the data")
    chunk_id: str = Field(..., description="The chunk id of the evidence")
    page_number: str = Field(..., description="The page number of the evidence")
    evidence: str = Field(..., description="The evidence of the data")

def generate_prompt(trimmed) -> list[list[dict]]:
    system_prompt = "You are a careful dataset generator for RAG. Only answer from the provided passage."
    
    user_prompt = f"""
                Task: Extract 3 atomic facts and write 1 QA whose answer from following document 
                Response: query : question, answers : answer, evidence : page content for the answer
            """

    queries = []
    for doc in trimmed:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": "chunk id: " + str(doc.metadata["id"])},
            {"role": "user", "content": "page number: " + str(doc.metadata["page"])},
            {"role": "user", "content": "content : " + doc.page_content},
        ]
        queries.append(prompt)
    return queries

def generate_data(llm: ChatOpenAI, queries: list[list[dict]]) -> list[SyntheticData]:
    model_with_structure = llm.with_structured_output(SyntheticData)
    responses = model_with_structure.batch(queries)
    return responses

def save_data(responses: list[SyntheticData]) -> None:
    arr = []
    for r in responses:
        obj = {
            "query": r.query,
            "answers": r.answers,
            "evidence": r.evidence,
            "page_number": r.page_number,
            "chunk_id": r.chunk_id,
        }
        arr.append(obj)
    
    df = pd.DataFrame(arr)
    df.to_csv(f"{output_path_prefix}_synthetic.csv", index=True)

def main():
    with open(f"{output_path_prefix}_split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)

    trimmed = split_documents[2:]
    llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)

    queries = generate_prompt(trimmed)
    print("쿼리 생성")
    print(f"쿼리 개수: {len(queries)}")

    responses = generate_data(llm, queries)
    print(f"응답 개수: {len(responses)}")
    print("응답 생성")
    
    save_data(responses)
    print("데이터 저장")
    print("✅ 모든 작업 완료")

if __name__ == "__main__":
    main()