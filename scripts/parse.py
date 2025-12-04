"""
JSON 파일을 파싱하여 차트, 그래프, 표 이미지를 추출하고, 문서를 생성
1. JSON 파일을 파싱
2. id, image id, page 순서 정렬
2. 추출한 차트, 그래프, 표 이미지를 문서에 넣음
3. 문서를 페이지 별로 병합
4. 필요없는 메타데이터 제거
5. 문서를 pickle 파일로 저장
6. 문서를 markdown 파일로 저장
"""

import os
from glob import glob
import json
import base64
import pickle
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

from config import input_file, image_output_path_prefix, output_path_prefix

def get_json_arr(input_file) -> list:
    # 분할 PDF 파일 목록 조회
    short_input_files = glob(os.path.splitext(input_file)[0] + "_*.pdf")
    
    arr = []
    for short_input_file in sorted(short_input_files):
        short_output_file = os.path.splitext(short_input_file)[0] + ".json"
     
        with open(short_output_file, "r") as f:
            arr.append(json.load(f))

    return arr

def flatten_json(json_data_arr) -> list:
    
    last_id, last_page = None, None
    for data in json_data_arr:
        
        for idx, element in enumerate(data['elements']):
            
            if last_id is not None and last_page is not None:
                start_id = last_id + 1 # id는 0부터 시작하기 때문에 다음 시작 아이디는 1을 더하고 시작
                element['id'] = start_id + element['id'] 
                element['page'] = last_page + element['page']

            if idx == len(data['elements']) - 1:
                last_id = element['id']
                last_page = element['page']

    return json_data_arr

def create_docs(json_data_arr) -> list:
    docs = []
    for data in json_data_arr:
        doc = []   
        for element in data['elements']:        
            metadata = {
                "id": element.get("id"),
                "page": element.get("page"),
                "category": element.get("category"),
                "html": element.get("content", {}).get("html"),
                "base64_encoding": element.get("base64_encoding", None),
                "image_id": [],
                "image_path": [],
                "text_summary": [],
                "image_summary": []                
            }
            doc.append(Document(page_content="", metadata=metadata))
        docs.extend(doc)

    return docs

def extract_images(docs) -> list:
    for idx, doc in enumerate(docs):
        if doc.metadata["category"] == "figure" or doc.metadata["category"] == "chart" or doc.metadata["category"] == "table":
            output_file = f"{image_output_path_prefix}_{doc.metadata['category']}_{idx}.png"
            output_file_path = output_file[1:]
            soup = BeautifulSoup(doc.metadata['html'], 'html.parser')
            if doc.metadata['category'] == 'figure':
                soup.find('img')['src'] = output_file
                replaced_html = str(soup)
                image_path = output_file
                doc.metadata['html'] = replaced_html
                
            elif doc.metadata['category'] == 'chart':
                soup.find('img')['src'] = output_file
                replaced_html = str(soup)
                image_path = output_file
                doc.metadata['html'] = replaced_html
                
            elif doc.metadata['category'] == 'table':
                img = soup.new_tag("img", src=output_file)
                soup.insert(0, img)
                replaced_html = str(soup)
                image_path = output_file
                doc.metadata['html'] = replaced_html
                
            doc.metadata['image_id'].append(doc.metadata['id'])
            doc.metadata['image_path'].append(image_path)
            
            with open (output_file_path, 'wb') as fh:
                fh.write(base64.decodebytes(str.encode(doc.metadata["base64_encoding"])))

        doc.page_content = md(doc.metadata['html'])

    return docs

def merge_docs(docs) -> list:
    merged = {}
    for doc in docs:
        current_page = doc.metadata['page']

        new_page = True if current_page not in merged.keys() else False
        bucket = merged.setdefault(current_page, doc.model_copy(deep=True))
        
        if len(doc.metadata['image_path']) > 0:
            if not new_page:
                bucket.metadata['image_id'].extend(doc.metadata['image_id'])
                bucket.metadata['image_path'].extend(doc.metadata['image_path'])
                bucket.page_content += "\n" + doc.page_content + "\n"

            else:
                bucket.page_content = "\n\n" + bucket.page_content + "\n\n" 
        else:
            if not new_page:
                bucket.page_content += "\n" + doc.page_content + "\n"

            else:
                bucket.page_content = "\n\n" + bucket.page_content + "\n\n" 

    return list(merged.values())

def remove_metadata(objects) -> list:
    for object in objects:
        del object.metadata['base64_encoding']
        del object.metadata['html']
        del object.metadata['category']
        del object.metadata['id']
    return objects

def save_docs(docs) -> list:
    with open(f'{output_path_prefix}_docs.pkl', 'wb') as f:
        pickle.dump(docs, f)
    return docs

def save_markdown(docs) -> str:
    arr = []
    for doc in docs:
        arr.append(doc.page_content)
    markdown = "\n".join(arr)
    with open(f'{output_path_prefix}_markdown.md', 'w') as f:
        f.write(markdown)
    return markdown

def parse_folder(folder_path):
    output_folder_path = folder_path / "outputs"
    image_folder_path = folder_path / "images"
    
    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True, exist_ok=True)
    if not image_folder_path.exists():
        image_folder_path.mkdir(parents=True, exist_ok=True)
    
    file_names = set()
    for path in folder_path.glob("*.pdf"):
        name = "_".join(path.stem.split("_")[:2])
        file_names.add(name)
    
    for file_name in file_names:
        json_data_arr = get_json_arr(file_name)
        flattened = flatten_json(json_data_arr)
        print("📄 문서 생성 완료")
        docs = create_docs(flattened)
        image_prefix = image_folder_path / file_name
        docs = extract_images(docs, image_prefix)
        print("📄 이미지 추출 완료")
        merged = merge_docs(docs)
        cleaned = remove_metadata(merged)  
        print("📄 메타데이터 제거 완료")

        output_prefix = output_folder_path / file_name
        save_docs(cleaned, output_prefix)
        print("📄 문서 저장 완료")

        save_markdown(cleaned, output_prefix)
        print("📄 마크다운 저장 완료")
        print("✅ 모든 작업 완료")

def main():

    json_data_arr = get_json_arr(input_file)
    flattened = flatten_json(json_data_arr)
    print("📄 문서 생성 완료")
    
    docs = create_docs(flattened)
    docs = extract_images(docs)
    print("📄 이미지 추출 완료")
    
    merged = merge_docs(docs)
    print("📄 문서 병합 완료")
    
    cleaned = remove_metadata(merged)  
    print("📄 메타데이터 제거 완료")
   
    save_docs(cleaned)
    print("📄 문서 저장 완료")

    save_markdown(cleaned)
    print("📄 마크다운 저장 완료")
    print("✅ 모든 작업 완료")

if __name__ == "__main__":
    main()
