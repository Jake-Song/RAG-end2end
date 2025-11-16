"""
Utility functions
1. format_context: LLM이 답변할 때 사용하는 컨택스트 포맷팅 도구. 이미지 요약과 텍스트 요약을 넣음음

"""

from langchain.schema import Document

def format_context(retrieved_docs: list[Document]) -> str:
    contexts = []
    for i, doc in enumerate(retrieved_docs):
        text = f"""### Context #{i+1}
                    <document>
                        <page_content>
                            {doc.page_content}
                        </page_content>
                        <text_summary>
                            {doc.metadata['text_summary'] if len(doc.metadata['text_summary']) > 0 else 'nothing'}
                        </text_summary>
                        <image_summary>
                            {doc.metadata['image_summary'] if len(doc.metadata['image_summary']) > 0 else 'nothing'} 
                        </image_summary>
                    </document>
                """
        contexts.append(text)
    return "".join(contexts)