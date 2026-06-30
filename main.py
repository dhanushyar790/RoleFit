"""
Resume Radar — FastAPI backend.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000 in your browser — this serves index.html
directly and exposes the API under /api/*, so there's no separate
frontend server and no CORS issues.
"""

import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from backend.file_parser import extract_text_from_bytes
from backend.semantic_matcher import compute_semantic_similarity, chunk_text_similarity
from backend.llm_analyzer import analyze_resume_vs_jd, rewrite_bullet_point
from backend.database import init_db, save_result, get_history, clear_history

app = FastAPI(title="Resume Radar API")

# Allow the frontend to call the API even if it's ever served from a
# different origin (e.g. opened as a local file, or a different port
# during development). Harmless to leave on even when same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


# ============================================================
# API ROUTES
# ============================================================

@app.post("/api/analyze")
async def analyze(jd_text: str = Form(...), files: List[UploadFile] = File(...)):
    results = []

    for upload in files:
        file_bytes = await upload.read()
        resume_text = extract_text_from_bytes(upload.filename, file_bytes)

        # 1. Semantic similarity (embeddings, fast, free, local)
        semantic_score = compute_semantic_similarity(resume_text, jd_text)

        # 2. LLM structured analysis
        llm_result = analyze_resume_vs_jd(resume_text, jd_text)

        # 3. Per-skill semantic scoring against key JD skills
        skill_scores = {}
        if llm_result.get("key_jd_skills"):
            skill_scores = chunk_text_similarity(resume_text, llm_result["key_jd_skills"])

        save_result(
            upload.filename,
            jd_text,
            llm_result.get("match_percentage", 0),
            semantic_score,
            ", ".join(llm_result.get("missing_skills", [])),
            ", ".join(llm_result.get("matching_skills", [])),
            " | ".join(llm_result.get("improvement_suggestions", [])),
            llm_result.get("summary", "")
        )

        results.append({
            "filename": upload.filename,
            "semantic_score": semantic_score,
            "match_percentage": llm_result.get("match_percentage", 0),
            "matching_skills": llm_result.get("matching_skills", []),
            "missing_skills": llm_result.get("missing_skills", []),
            "improvement_suggestions": llm_result.get("improvement_suggestions", []),
            "summary": llm_result.get("summary", ""),
            "skill_scores": skill_scores,
        })

    return JSONResponse({"results": results})


@app.get("/api/history")
async def history():
    rows = get_history()
    # rows columns: id, resume_name, jd_snippet, match_pct, semantic_score,
    # missing_skills, matching_skills, suggestions, summary, created_at
    history_list = [
        {
            "id": row[0],
            "resume_name": row[1],
            "jd_snippet": row[2],
            "match_pct": row[3],
            "semantic_score": row[4],
            "missing_skills": row[5],
            "matching_skills": row[6],
            "suggestions": row[7],
            "summary": row[8],
            "created_at": row[9],
        }
        for row in rows
    ]
    return JSONResponse({"history": history_list})


@app.delete("/api/history")
async def delete_history():
    clear_history()
    return JSONResponse({"status": "cleared"})


@app.post("/api/rewrite-bullet")
async def rewrite_bullet(bullet: str = Form(...), jd_text: str = Form(...)):
    improved = rewrite_bullet_point(bullet, jd_text)
    return JSONResponse({"rewritten": improved})


# ============================================================
# FRONTEND (serves index.html and any other static assets)
# ============================================================

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount the rest of the frontend folder (css/js/images if you add any later)
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")