import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

DB_URI = os.environ["POSTGRES_URI"]

def drop_table():
    conn = psycopg2.connect(url=DB_URI)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS test_table")
    conn.commit()
    print("Table dropped successfully")
    cursor.close()
    conn.close()

# Run setup
drop_table()