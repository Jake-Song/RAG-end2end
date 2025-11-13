from langchain.schema import Document

def format_context(results: list[Document]) -> str:
    arr = []
    for i, doc in enumerate(results):
        text = f"""Retrieved #{i+1}
                    {doc.page_content}
                    {doc.metadata['text_summary'] if len(doc.metadata['text_summary']) > 0 else ''}
                    {doc.metadata['image_summary'] if len(doc.metadata['image_summary']) > 0 else ''}
                """
        arr.append(text)
    return "\n".join(arr)