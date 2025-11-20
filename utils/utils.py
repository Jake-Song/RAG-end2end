"""
Utility functions
1. format_context: LLM이 답변할 때 사용하는 컨택스트 포맷팅 도구. 이미지 요약과 텍스트 요약을 넣음음

"""

from langchain.schema import Document

def format_context(retrieved_docs: list[Document]) -> str:
    contexts = []
    for i, doc in enumerate(retrieved_docs):
        #LLM에 들어갈 컨택스트이기 때문에 띄어쓰기 없이 출력.
        text = f"### Context #{i+1}<document><page_content>{doc.page_content}</page_content><text_summary>{doc.metadata['text_summary'] if len(doc.metadata['text_summary']) > 0 else 'nothing'}</text_summary><image_summary>{doc.metadata['image_summary'] if len(doc.metadata['image_summary']) > 0 else 'nothing'}</image_summary></document>"
        contexts.append(text)
    return "".join(contexts)

if __name__ == "__main__":
    # 테스트 출력
    docs = [
        Document(
            page_content="This is the main content of document 1.",
            metadata={
                "text_summary": "Summary of text 1",
                "image_summary": "Description of image 1"
            }
        ),
        Document(
            page_content="This is document 2 with no image summary.",
            metadata={
                "text_summary": "Summary of text 2",
                "image_summary": ""
            }
        ),
        Document(
            page_content="This is document 3 with no summaries.",
            metadata={
                "text_summary": "",
                "image_summary": ""
            }
        )
    ]
    
    formatted_output = format_context(docs)
    
    print(formatted_output)