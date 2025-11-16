"""
PDF 파일을 분할하고, 분할된 PDF 파일을 업스테이지 API를 사용하여 파싱하여 JSON 파일로 저장합니다.
"""

import os
import fitz
from glob import glob
import json
import requests

from config import input_file, batch_size

# 업스테이지 api key 환경변수 설정
API_KEY = os.environ.get("UPSTAGE_API_KEY")

 
def split_pdf(input_file, batch_size):
    # Open input_pdf
    input_pdf = fitz.open(input_file)
    num_pages = len(input_pdf)
    print(f"Total number of pages: {num_pages}")
 
    # Split input_pdf
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page + batch_size, num_pages) - 1
 
        # Write output_pdf to file
        input_file_basename = os.path.splitext(input_file)[0]
        output_file = f"{input_file_basename}_{start_page}_{end_page}.pdf"
        print(output_file)
        with fitz.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
 
    input_pdf.close()
 
def call_document_parse(input_file, output_file):
    
    response = requests.post(
        "https://api.upstage.ai/v1/document-digitization",
        headers={"Authorization": f"Bearer {API_KEY}"},
        data={"base64_encoding": "['figure', 'chart', 'table']", "model": "document-parse"}, # base64 이미지 인코딩
        files={"document": open(input_file, "rb")})
 
    if response.status_code == 200:
        with open(output_file, "w") as f:
            json.dump(response.json(), f, ensure_ascii=False)
    else:
        raise ValueError(f"Unexpected status code {response.status_code}.")

def main():

    split_pdf(input_file, batch_size)
    print("📄 JSON 파일 생성 중...")

    # PDF 파일 목록 조회
    short_input_files = glob(os.path.splitext(input_file)[0] + "_*.pdf")
    
    # 파싱 json 파일 생성
    for short_input_file in short_input_files:
        print(short_input_file)
        short_output_file = os.path.splitext(short_input_file)[0] + ".json"
        call_document_parse(short_input_file, short_output_file)
    
    print("✅ JSON 파일 생성 완료 ")

if __name__ == "__main__":
    main()