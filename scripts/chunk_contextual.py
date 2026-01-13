import pickle
import uuid
import base64
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).parent.parent
llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
embeddings = UpstageEmbeddings(model="embedding-passage")

def prompt_image_caption(docs: list) -> list:
    messages_for_image = []

    for doc in docs:
        if doc.metadata.get("image_path"):
            file_paths = doc.metadata["image_path"]
            doc_content = doc.page_content
            
            image_data_arr = []
            for file_path in file_paths:
                file_path_str = "".join(file_path)[1:]
                path = project_root / file_path_str
                image_data_arr.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
                
            for idx, image_data in enumerate(image_data_arr):
                
                messages_for_image.append(
                    {  
                        "doc_page": doc.metadata["page"],
                        "image_id": doc.metadata["image_id"][idx],
                        "prompt":  {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": f"Here is the content of the document.{doc_content}"},
                                            {"type": "text", 
                                            "text": """Please give a short succinct image caption to situate this image 
                                            within the overall document for the purposes of improving search retrieval
                                            of the chunk. Answer only with the succinct image caption and nothing else. 
                                            Describe content of the image in Korean."""},
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},            
                                            },
                                        ]
                                    }
                    }
                )
                
    return messages_for_image

def insert_image_description(docs: list, messages_for_image: list) -> list:
    queries = []
    for message in messages_for_image:
        queries.append([message["prompt"]])
    
    responses = llm.batch(queries)
    for idx, message in enumerate(messages_for_image):
        message["image_description"] = responses[idx].content

    for doc in docs:
        doc.metadata["image_description"] = ""
        for message in messages_for_image:
            for image_id in doc.metadata["image_id"]:
                if image_id == message["image_id"]:
                    doc.metadata["image_description"] += message["image_description"] + "\n"
    return docs

def pre_situate_context(docs: list[Document]) -> list[Document]:
    for doc in tqdm(docs, desc="Generating document IDs"):
        doc.metadata["doc_id"] = str(uuid.uuid4())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    
    page = 0
    chunk_index = 1
    for doc in tqdm(split_documents, desc="Assigning chunk IDs"):
        if page != doc.metadata["page"]:
            page = doc.metadata["page"]
            chunk_index = 1
        doc.metadata["chunk_id"] = "page_" + str(doc.metadata["page"]) + "_chunk_" + str(chunk_index)
        chunk_index += 1   

    return split_documents

def situate_context_batch(docs: list[Document], split_documents: list[Document], batch_size: int = 10) -> dict[str, str]:
    
    DOCUMENT_CONTEXT_PROMPT = """
    <document>
    {doc_content}
    {image_description_section}
    </document>
    """
    
    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within image description and content in the whole document.
    <chunk>
    {chunk_content}
    </chunk>
    
    Please give a short succinct context to situate this chunk within 
    the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    prompts = []
    for doc in docs:
        for chunk in split_documents:
            if doc.metadata['page'] == chunk.metadata['page']:
                image_description = doc.metadata.get("image_description")
                image_description_section = ""
                if image_description:
                    if isinstance(image_description, list):
                        image_description = "\n".join(image_description)
                    image_description_section = f"<image_description>\n{image_description}\n</image_description>"

                prompt = [
                    {"role": "system", "content": "You MUST answer in Korean."},
                    {"role": "user", 
                    "content": DOCUMENT_CONTEXT_PROMPT.format(
                        doc_content=doc.page_content, 
                        image_description_section=image_description_section
                        )
                    },
                    {"role": "user", 
                    "content": CHUNK_CONTEXT_PROMPT.format(
                        chunk_content=chunk.page_content
                        )
                    },
                ]
                prompts.append(prompt)
    
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Contextualizing chunks"):
        batch = prompts[i:i+batch_size]
        responses.extend(llm.batch(batch))

    return responses, split_documents

def post_situate_context(response: list[Document], split_documents: list[Document]) -> list[Document]:
    
    for r, chunk in tqdm(
        zip(response, split_documents), 
        total=len(split_documents), 
        desc="Finalizing contextualized chunks"):
        
        chunk.page_content =  r.content + "\n\n" + chunk.page_content
        chunk.metadata["contextualized_content"] = r.content + "\n\n" + chunk.page_content
        chunk.metadata["original_content"] = chunk.page_content   
    
    return split_documents

def save_data(split_documents: list[Document], file_name: str):
    with open(project_root / "outputs" / "SPRI_2025_output_split_documents.pkl", "wb") as f:
        pickle.dump(split_documents, f)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local("faiss_index", file_name)

if __name__ == "__main__":
    with open("outputs/SPRI_2025_output_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    prompts = prompt_image_caption(docs)
    inserted_docs = insert_image_description(docs, prompts)
    pre_split_documents = pre_situate_context(inserted_docs)
    response, post_split_documents = situate_context_batch(inserted_docs, pre_split_documents)
    split_documents = post_situate_context(response, post_split_documents)
    save_data(split_documents, "SPRI_2025_contextual")






