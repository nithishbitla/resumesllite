from sentence_transformers import SentenceTransformer, util
import PyPDF2

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(path):
    with open(path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def rank_resumes(job_description, resumes):
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    scored = []
    for resume in resumes:
        text = extract_text_from_pdf(resume['filepath'])
        resume_embedding = model.encode(text, convert_to_tensor=True)
        score = util.cos_sim(job_embedding, resume_embedding).item()
        resume['score'] = round(score, 3)
        scored.append(resume)
    return sorted(scored, key=lambda x: x['score'], reverse=True)