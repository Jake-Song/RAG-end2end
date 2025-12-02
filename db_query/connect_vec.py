import os
import sys
from pathlib import Path
from langchain_upstage import UpstageEmbeddings
from langchain_postgres import PGVector

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

DB_URI = os.environ["POSTGRES_URI"]

embeddings = UpstageEmbeddings(model="embedding-passage")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="SPRI_ALL",
    connection=DB_URI
)

