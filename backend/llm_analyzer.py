import os
import json
from openai import OpenAI

_client = None


def _get_client():
    """Lazily create the OpenAI/Groq client so a missing API key doesn't
    crash the whole app at import time — only when actually needed."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set GROQ_API_KEY or OPENAI_API_KEY as an "
                "environment variable before starting the server."
            )
        base_url = "https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else None
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


# openai/gpt-oss-120b: strong reasoning, good accuracy, still fast on Groq's
# free tier. (llama-3.1-8b-instant is deprecated and too weak for accurate
# skill-matching — it was hallucinating matches/misses on real resumes.)
MODEL = "openai/gpt-oss-120b" if os.environ.get("GROQ_API_KEY") else "gpt-4o-mini"


def analyze_resume_vs_jd(resume_text, jd_text):
    """
    Returns a structured dict (parsed from JSON) with:
    match_percentage, missing_skills, matching_skills, suggestions, summary
    """
    client = _get_client()

    prompt = f"""
You are an expert technical recruiter. Compare this RESUME against this JOB DESCRIPTION.

RESUME:
{resume_text[:6000]}

JOB DESCRIPTION:
{jd_text[:3000]}

Return ONLY valid JSON (no markdown, no backticks, no explanation) in exactly this schema:

{{
  "match_percentage": <integer 0-100>,
  "matching_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "improvement_suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "summary": "2-3 sentence overall assessment",
  "key_jd_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"]
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"} if "gpt" in MODEL else None,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case model adds markdown fences anyway
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback minimal structure if parsing fails
        return {
            "match_percentage": 0,
            "matching_skills": [],
            "missing_skills": [],
            "improvement_suggestions": ["Could not parse AI response. Try again."],
            "summary": raw[:300],
            "key_jd_skills": []
        }


def rewrite_bullet_point(bullet_text, jd_text):
    """Rewrite a single resume bullet point to better align with the JD."""
    client = _get_client()

    prompt = f"""
Rewrite this resume bullet point to be more impactful and aligned with the
target job description. Use strong action verbs, quantify impact if possible,
and naturally include relevant keywords from the JD. Keep it to 1-2 lines.

ORIGINAL BULLET: {bullet_text}

JOB DESCRIPTION CONTEXT: {jd_text[:1000]}

Return ONLY the rewritten bullet point, nothing else.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()