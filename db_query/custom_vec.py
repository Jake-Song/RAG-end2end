import os
import sys
from pathlib import Path
project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore
from dotenv import load_dotenv
load_dotenv()

embeddings = UpstageEmbeddings(model="embedding-passage")

DB_URI = os.environ["POSTGRES_URI"]
pg_engine = PGEngine.from_connection_string(
    url=DB_URI
)

vector_store = PGVectorStore.create_sync(
    engine=pg_engine,
    table_name='test_table',
    embedding_service=embeddings
)

documents = [Document(page_content="Hello, world!", metadata={"source": "test"})]
vector_store.add_documents(documents)
results = vector_store.similarity_search("Hello, world!")
print(results)



