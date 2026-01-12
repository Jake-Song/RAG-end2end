import pickle
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
embeddings = UpstageEmbeddings(model="embedding-passage")

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
    </document>
    """
    
    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>
    
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    prompts = []
    for doc in docs:
        for chunk in split_documents:
            if doc.metadata['page'] == chunk.metadata['page']:
                prompt = [
                    {"role": "system", "content": "You MUST answer in Korean."},
                    {"role": "user", "content": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc.page_content)},
                    {"role": "user", "content": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk.page_content)},
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
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local("faiss_index", file_name)

if __name__ == "__main__":
    with open("outputs/SPRI_2025_output_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    split_documents = pre_situate_context(docs[:4])
    response, split_documents = situate_context_batch(docs[:4], split_documents)
    split_documents = post_situate_context(response, split_documents)
    save_data(split_documents, "SPRI_2025_contextual")






