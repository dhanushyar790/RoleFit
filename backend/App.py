from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from file_parser import extract_text_from_bytes
from semantic_matcher import compute_semantic_similarity, chunk_text_similarity
from llm_analyzer import analyze_resume_vs_jd, rewrite_bullet_point
from database import init_db, save_result, get_history, clear_history

app = FastAPI(title="Resume Radar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.post("/api/analyze")
async def analyze(jd_text: str = Form(...), files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        file_bytes = await f.read()
        resume_text = extract_text_from_bytes(f.filename, file_bytes)

        semantic_score = compute_semantic_similarity(resume_text, jd_text)
        llm_result = analyze_resume_vs_jd(resume_text, jd_text)
        skill_scores = chunk_text_similarity(resume_text, llm_result.get("key_jd_skills", []))

        save_result(
            f.filename,
            jd_text,
            llm_result.get("match_percentage", 0),
            semantic_score,
            ", ".join(llm_result.get("missing_skills", [])),
            ", ".join(llm_result.get("matching_skills", [])),
            " | ".join(llm_result.get("improvement_suggestions", [])),
            llm_result.get("summary", "")
        )

        results.append({
            "filename": f.filename,
            "semantic_score": semantic_score,
            "skill_scores": skill_scores,
            **llm_result
        })

    return {"results": results}


@app.post("/api/rewrite-bullet")
async def rewrite_bullet(bullet: str = Form(...), jd_text: str = Form(...)):
    rewritten = rewrite_bullet_point(bullet, jd_text)
    return {"rewritten": rewritten}


@app.get("/api/history")
def history():
    rows = get_history()
    cols = ["id", "resume_name", "jd_snippet", "match_pct", "semantic_score",
            "missing_skills", "matching_skills", "suggestions", "summary", "created_at"]
    return {"history": [dict(zip(cols, r)) for r in rows]}


@app.delete("/api/history")
def delete_history():
    clear_history()
    return {"status": "cleared"}


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")