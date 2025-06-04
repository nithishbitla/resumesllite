import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE = os.getenv("SQLITE_DB_PATH", "resumes.db")

def init_db():
    if os.path.exists(DATABASE):
        print(f"Database {DATABASE} already exists.")
        return
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_uid TEXT NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database {DATABASE} created with resumes table.")

if __name__ == "__main__":
    init_db()
