import sqlite3

def init_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            position TEXT,
            filepath TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_resume(path, name, email, position, filepath):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO resumes (name, email, position, filepath) VALUES (?, ?, ?, ?)", (name, email, position, filepath))
    conn.commit()
    conn.close()

def get_all_resumes(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, email, position, filepath FROM resumes")
    resumes = cursor.fetchall()
    conn.close()
    return [{'name': r[0], 'email': r[1], 'position': r[2], 'filepath': r[3]} for r in resumes]