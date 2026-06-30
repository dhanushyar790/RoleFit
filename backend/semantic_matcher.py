from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_semantic_similarity(resume_text, jd_text):
    model = get_model()
    embeddings = model.encode([resume_text, jd_text])
    sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    score = float(max(0, min(100, round(sim * 100, 1))))
    return score


def chunk_text_similarity(resume_text, jd_skills_list):
    model = get_model()
    resume_chunks = [c.strip() for c in resume_text.split("\n") if len(c.strip()) > 10]
    if not resume_chunks:
        resume_chunks = [resume_text]

    resume_embeddings = model.encode(resume_chunks)
    results = {}

    for skill in jd_skills_list:
        skill_embedding = model.encode([skill])
        sims = cosine_similarity(skill_embedding, resume_embeddings)[0]
        best_score = float(round(float(np.max(sims)) * 100, 1))
        results[skill] = best_score

    return results