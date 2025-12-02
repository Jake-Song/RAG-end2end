import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

DB_URI = os.environ["POSTGRES_URI"]

def create_table():
    conn = psycopg2.connect(url=DB_URI)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_table(
        id SERIAL PRIMARY KEY,
        langchain_id VARCHAR(255) UNIQUE,
        content TEXT,
        embedding vector(4096),
        langchain_metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    print("Tables created successfully")
    
    cursor.close()
    conn.close()


# Run setup
create_table()
