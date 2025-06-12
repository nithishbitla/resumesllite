import os
import sqlite3
import time
import csv
from io import StringIO
from flask import (
    Flask, request, render_template, redirect, url_for, jsonify, flash,
    session, send_from_directory, Response
)
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from contextlib import closing
import tempfile

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), "uploads")

# Firebase config
RENDER_SECRET_PATH = "/var/render/secrets/firebase_config.json"
LOCAL_FALLBACK_PATH = "firebase_config.json"

cred_path = os.getenv("FIREBASE_CREDENTIAL_PATH") or (
    RENDER_SECRET_PATH if os.path.exists(RENDER_SECRET_PATH) else LOCAL_FALLBACK_PATH
)
if not os.path.exists(cred_path):
    raise FileNotFoundError("Firebase config file not found at expected paths.")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# Database
DATABASE = os.path.join(tempfile.gettempdir(), "resumes.db")
HOST_EMAIL = os.getenv("HOST_EMAIL", "host@example.com")
HOST_PASSWORD = os.getenv("HOST_PASSWORD", "hostpass123")

# Load SentenceTransformer once
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ✅ Ensure uploads folder and DB table exist even on Render
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
with closing(get_db_connection()) as conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_uid TEXT NOT NULL,
            user_name TEXT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

def verify_token(token):
    try:
        return auth.verify_id_token(token)
    except Exception as e:
        app.logger.warning(f"Token verification failed: {e}")
        return None

@app.route('/')
def login():
    if 'user' in session:
        return redirect(url_for('upload'))
    if 'host' in session:
        return redirect(url_for('host_dashboard'))
    return render_template('login.html')

@app.route('/verify-token', methods=['POST'])
def verify_token_route():
    data = request.get_json()
    token = data.get('token')
    user = verify_token(token)
    if user:
        session['user'] = user['uid']
        session['user_email'] = user.get('email')
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid token'}), 401

@app.route('/upload')
def upload():
    if 'user' not in session:
        flash("Please login first.")
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    token = request.form.get('idToken')
    user = verify_token(token)
    if not user:
        return jsonify({"error": "Invalid or missing token"}), 401

    uid = user['uid']
    name = user.get('name', user.get('email', 'Unknown'))

    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uid}_{int(time.time())}_{filename}"
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(save_path)

    with closing(get_db_connection()) as conn:
        conn.execute(
            "INSERT INTO resumes (user_uid, user_name, filename, filepath) VALUES (?, ?, ?, ?)",
            (uid, name, unique_filename, save_path)
        )
        conn.commit()

    return jsonify({"success": True, "message": "Resume uploaded successfully."})

@app.route('/host-login', methods=['GET', 'POST'])
def host_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == HOST_EMAIL and password == HOST_PASSWORD:
            session['host'] = email
            return redirect(url_for('host_dashboard'))
        else:
            flash("Invalid host credentials.")
    return render_template('host_login.html')

@app.route('/host-dashboard', methods=['GET', 'POST'])
def host_dashboard():
    if 'host' not in session:
        flash("Please login as host.")
        return redirect(url_for('host_login'))

    job_description = ''
    ranked_resumes = None

    with closing(get_db_connection()) as conn:
        resumes = conn.execute('SELECT * FROM resumes ORDER BY uploaded_at DESC').fetchall()

    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        if job_description and resumes:
            job_emb = model.encode(job_description, convert_to_tensor=True)
            resume_texts = []
            for r in resumes:
                try:
                    with open(r['filepath'], 'r', encoding='utf-8', errors='ignore') as f:
                        resume_texts.append(f.read())
                except:
                    resume_texts.append('')

            resume_embs = model.encode(resume_texts, convert_to_tensor=True)
            cosine_scores = util.cos_sim(job_emb, resume_embs)[0].tolist()
            ranked_resumes = sorted(zip(resumes, cosine_scores), key=lambda x: x[1], reverse=True)
        else:
            flash("Enter job description and ensure resumes exist.")
    else:
        ranked_resumes = [(r, 0) for r in resumes]

    return render_template('host_dashboard.html', resumes=ranked_resumes, job_description=job_description)

@app.route('/download-ranked-resumes-csv', methods=['POST'])
def download_ranked_resumes_csv():
    if 'host' not in session:
        flash("Login as host first.")
        return redirect(url_for('host_login'))

    job_description = request.form.get('job_description', '').strip()

    with closing(get_db_connection()) as conn:
        resumes = conn.execute('SELECT * FROM resumes ORDER BY uploaded_at DESC').fetchall()

    if not job_description or not resumes:
        flash("Enter job description and ensure resumes exist.")
        return redirect(url_for('host_dashboard'))

    job_emb = model.encode(job_description, convert_to_tensor=True)
    resume_texts = []
    for r in resumes:
        try:
            with open(r['filepath'], 'r', encoding='utf-8', errors='ignore') as f:
                resume_texts.append(f.read())
        except:
            resume_texts.append('')

    resume_embs = model.encode(resume_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(job_emb, resume_embs)[0].tolist()
    ranked_resumes = sorted(zip(resumes, cosine_scores), key=lambda x: x[1], reverse=True)

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['Rank', 'Name', 'Filename', 'Uploaded At', 'Similarity Score'])

    for idx, (resume, score) in enumerate(ranked_resumes, start=1):
        writer.writerow([
            idx,
            resume['user_name'] or 'Unknown',
            resume['filename'],
            resume['uploaded_at'],
            f"{score:.4f}"
        ])

    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=ranked_resumes.csv"}
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/host-logout')
def host_logout():
    session.clear()
    return redirect(url_for('host_login'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    if 'host' not in session and 'user' not in session:
        return "Unauthorized", 401
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    return "OK", 200

# Production-ready entry point for gunicorn
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
