import os
import sqlite3
import time
import csv
import tempfile
import gc
import torch
from io import StringIO
from flask import (
    Flask, request, render_template, redirect, url_for, jsonify, flash,
    session, send_from_directory, Response
)
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

# Limit Torch CPU threads to reduce memory usage
torch.set_num_threads(1)

# Load lightweight transformer modules only when needed
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------------------------------------------
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), "uploads")

# -----------------------------------------------------------------------------
# Firebase Admin setup
RENDER_SECRET_PATH = "/var/render/secrets/firebase_config.json"
LOCAL_FALLBACK_PATH = "firebase_config.json"
cred_path = os.getenv("FIREBASE_CREDENTIAL_PATH") or (
    RENDER_SECRET_PATH if os.path.exists(RENDER_SECRET_PATH) else LOCAL_FALLBACK_PATH
)
if not os.path.exists(cred_path):
    raise FileNotFoundError("Firebase config file not found at expected paths.")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# -----------------------------------------------------------------------------
# SQLite DB setup
DATABASE = os.path.join(tempfile.gettempdir(), "resumes.db")
HOST_EMAIL = os.getenv("HOST_EMAIL", "host@example.com")
HOST_PASSWORD = os.getenv("HOST_PASSWORD", "hostpass123")

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Ensure upload folder and DB table exist early
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
with get_db_connection() as conn:
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

# -----------------------------------------------------------------------------
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
    user = verify_token(data.get('token'))
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
    user = verify_token(request.form.get('idToken'))
    if not user:
        return jsonify({"error": "Invalid or missing token"}), 401
    uid = user['uid']
    name = user.get('name', user.get('email', 'Unknown'))

    file = request.files.get('resume')
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uid}_{int(time.time())}_{filename}"
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(save_path)

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO resumes (user_uid, user_name, filename, filepath) VALUES (?, ?, ?, ?)",
            (uid, name, unique_filename, save_path)
        )
        conn.commit()
    return jsonify({"success": True, "message": "Resume uploaded successfully."})

@app.route('/host-login', methods=['GET', 'POST'])
def host_login():
    if request.method == 'POST':
        if request.form.get('email') == HOST_EMAIL and request.form.get('password') == HOST_PASSWORD:
            session['host'] = True
            return redirect(url_for('host_dashboard'))
        flash("Invalid host credentials.")
    return render_template('host_login.html')

@app.route('/host-dashboard', methods=['GET', 'POST'])
def host_dashboard():
    if not session.get('host'):
        flash("Please login as host.")
        return redirect(url_for('host_login'))

    with get_db_connection() as conn:
        resumes = conn.execute('SELECT * FROM resumes ORDER BY uploaded_at DESC').fetchall()

    ranked = [(r, 0.0) for r in resumes]
    if request.method == 'POST':
        job_desc = request.form.get('job_description', '').strip()
        if job_desc and resumes:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            job_emb = model.encode(job_desc, convert_to_tensor=True)
            texts = [open(r['filepath'], 'r', encoding='utf-8', errors='ignore').read() for r in resumes]
            embs = model.encode(texts, convert_to_tensor=True)
            scores = util.cos_sim(job_emb, embs)[0].tolist()
            ranked = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
            del job_emb, embs, scores
            gc.collect()

    return render_template('host_dashboard.html', resumes=ranked, job_description=request.form.get('job_description', ''))

@app.route('/download-ranked-resumes-csv', methods=['POST'])
def download_csv():
    if not session.get('host'):
        return redirect(url_for('host_login'))
    job_desc = request.form.get('job_description', '').strip()
    with get_db_connection() as conn:
        resumes = conn.execute('SELECT * FROM resumes ORDER BY uploaded_at DESC').fetchall()
    if job_desc and resumes:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        job_emb = model.encode(job_desc, convert_to_tensor=True)
        texts = [open(r['filepath'], 'r', encoding='utf-8', errors='ignore').read() for r in resumes]
        embs = model.encode(texts, convert_to_tensor=True)
        scores = util.cos_sim(job_emb, embs)[0].tolist()
        ranked = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
        del job_emb, embs, scores
        gc.collect()
    else:
        ranked = [(r, 0.0) for r in resumes]

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['Rank', 'Name', 'Filename', 'Uploaded At', 'Similarity'])
    for idx, (r, s) in enumerate(ranked, 1):
        writer.writerow([idx, r['user_name'] or 'Unknown', r['filename'], r['uploaded_at'], f"{s:.4f}"])
    return Response(si.getvalue(), mimetype='text/csv', headers={"Content-Disposition":"attachment;filename=ranked.csv"})

@app.route('/uploads/<filename>')
def serve_file(filename):
    if 'user' not in session and not session.get('host'):
        return "Unauthorized", 401
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
