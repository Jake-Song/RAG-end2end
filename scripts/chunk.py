"""
페이지안에서 데이터를 분할합니다. (예 1페이지에서 2개의 청크, 2페이지에서 3개의 청크 등등)
각 청크는 메타데이터(id, image_id, page, image_path, text_summary, image_summary)는 공유합니다.

1. 이미지 요약 프롬프트 생성
2. 텍스트 요약 프롬프트 생성
3. 이미지 요약(LLM)
4. 텍스트 요약(LLM)
5. 문서 분할(Text Splitter)
6. 문서 저장(Text, Pickle)
"""


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import base64, pathlib
import pickle
from config import output_path_prefix

def prepare_image_summary(docs: list) -> list:
    messages_for_image = []

    for doc in docs:
        file_paths = doc.metadata["image_path"]
        context = doc.page_content

        if len(file_paths) > 1:
            image_data_arr = []
            for file_path in file_paths:
                file_path_str = "".join(file_path)[1:]
                path = pathlib.Path(file_path_str)
                image_data_arr.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
                
            for idx, image_data in enumerate(image_data_arr):
                
                messages_for_image.append(
                    {  
                        "image_id": doc.metadata["image_id"][idx],
                        "prompt":  {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": f"Here is the text of the context.{context}"},
                                            {"type": "text", "text": "Describe content of the image in Korean."},
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},            
                                            },
                                        ]
                                    }
                    }
                )
            
        elif len(file_paths) == 1:
            file_path_str = "".join(file_paths)[1:]
            path = pathlib.Path(file_path_str)
            image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

            messages_for_image.append(
                {  
                    "image_id": doc.metadata["image_id"],
                    "prompt":  {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Here is the text of the context.{context}"},
                                        {"type": "text", "text": "Describe content of the image in Korean."},
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},            
                                        },
                                    ]
                                }
                }
            )

    return messages_for_image

def word_count(text):
    return len(text.split())

def prepare_text_summary(docs: list) -> list:
    messages_for_text = []
    for idx, doc in enumerate(docs):
        # TODO: 요약하는 기준을 더 좋은 방법으로 수정
        if word_count(doc.page_content) > 1000:
            context = doc.page_content
            message = {
                        "page": doc.metadata["page"],
                        "prompt": {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Here is the text of the context.{context}"},
                                {"type": "text", "text": "Summarize the text in Korean."},
                            ]
                        }
                    }
                        
            messages_for_text.append(message)
    return messages_for_text

def summarize_image(docs: list, messages_for_image: list, llm: ChatOpenAI) -> list:
    queries = []
    for message in messages_for_image:
        queries.append([message["prompt"]])
    
    responses = llm.batch(queries)
    for idx, message in enumerate(messages_for_image):
        message["image_summary"] = responses[idx].content

    for doc in docs:
        for message in messages_for_image:
            if len(doc.metadata["image_id"]) == 1:
                if doc.metadata["image_id"] == message["image_id"]:
                    doc.metadata["image_summary"] = message["image_summary"]
            else:
                for image_id in doc.metadata["image_id"]:
                    if image_id == message["image_id"]:
                        doc.metadata["image_summary"].append(message["image_summary"])
    return docs

def summarize_text(docs: list, messages_for_text: list, llm: ChatOpenAI) -> list:
    
    queries_text = []
    for message in messages_for_text:
        queries_text.append([message["prompt"]])

    responses_text = llm.batch(queries_text)
    for idx, message in enumerate(messages_for_text):
        message["summary"] = responses_text[idx].content

    for doc in docs:
        for message in messages_for_text:
            if doc.metadata["page"] == message["page"]:
                doc.metadata["text_summary"] = message["summary"]
    return docs

def split_docs(docs: list) -> list:
    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    print(f"분할된 청크의수: {len(split_documents)}")
    for i, doc in enumerate(split_documents):
        doc.metadata['id'] = i
    return split_documents

def save_text(split_documents: list) -> None:
    # 청킹 구분 확인하기 위해 문서 저장장
    delimiter = "\n\n\n" + ("---" * 50) + "\n\n\n"
    split_documents_text = delimiter.join([doc.page_content for doc in split_documents])
    with open(f"{output_path_prefix}_split_documents.txt", "w", encoding="utf-8") as f:
        f.write(split_documents_text)

def save_split_document(split_documents: list) -> None:
    with open(f'{output_path_prefix}_split_documents.pkl', 'wb') as f:
        pickle.dump(split_documents, f)

def main():
    with open(f"{output_path_prefix}_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0.0)

    messages_for_image = prepare_image_summary(docs)
    messages_for_text = prepare_text_summary(docs)
    print("이미지 요약 시작")
    docs = summarize_image(docs, messages_for_image, llm)
    print("이미지 요약 완료")
    print("텍스트 요약 시작")
    docs = summarize_text(docs, messages_for_text, llm)
    print("텍스트 요약 완료")
    print("문서 분할 시작")
    split_documents = split_docs(docs)
    print("문서 분할 완료")
    print("텍스트 저장 시작")
    save_text(split_documents)
    print("텍스트 저장 완료")
    save_split_document(split_documents)
    print("문서 저장 완료")
    print("✅ 모든 작업 완료")
    
if __name__ == "__main__":
    main()