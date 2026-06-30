import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def init_db():
    """Create the history table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_name TEXT,
            jd_snippet TEXT,
            match_pct INTEGER,
            semantic_score REAL,
            missing_skills TEXT,
            matching_skills TEXT,
            suggestions TEXT,
            summary TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_result(resume_name, jd_text, match_pct, semantic_score,
                 missing, matching, suggestions, summary):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO history
        (resume_name, jd_snippet, match_pct, semantic_score, missing_skills,
         matching_skills, suggestions, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        resume_name, jd_text[:200], match_pct, semantic_score,
        missing, matching, suggestions, summary,
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows


def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()