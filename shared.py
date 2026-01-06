# shared.py
# ---------------------------------------------------------------------
# Shared utilities used across the SEO Automator app.
# - Env loading
# - SerpAPI Google search helper
# - OpenAI Chat wrapper
# - SERP parsers (returns FULL AI Overview object)
# - LLM-based semantic term extraction (JSON in / JSON out)
# - Small helpers (timestamp string, DataFrame -> Excel bytes)
# ---------------------------------------------------------------------

from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; if not present, env-only is fine.
    pass

# -----------------------------
# Environment / keys
# -----------------------------

def load_env_keys() -> Tuple[str, str, str]:
    """
    Returns (OPENAI_API_KEY, SERPAPI_KEY, AHREFS_API_TOKEN) from environment/.env.
    Missing ones are returned as empty strings.
    """
    return (
        os.getenv("OPENAI_API_KEY", "").strip(),
        os.getenv("SERPAPI_KEY", "").strip(),
        os.getenv("AHREFS_API_TOKEN", "").strip(),
    )

# -----------------------------
# Small helpers
# -----------------------------

def ts() -> str:
    """Timestamp string for file names."""
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

def to_excel_bytes(df, sheet_name: str = "Sheet1") -> bytes:
    """
    Convert a pandas DataFrame to .xlsx bytes.
    (Import pandas in the caller’s module to avoid heavyweight import here if you want.)
    """
    import pandas as pd  # local import to keep shared.py light
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()

# -----------------------------
# SerpAPI
# -----------------------------

def serpapi_google_search(
    keyword: str,
    num: int,
    gl: str,
    hl: str,
    serpapi_key: str,
    extra_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Fetch Google SERP JSON from SerpAPI.
    Returns the parsed JSON dict (raises for HTTP errors).
    """
    if not serpapi_key:
        raise RuntimeError("Missing SERPAPI_KEY")

    params = {
        "engine": "google",
        "q": keyword,
        "num": max(1, min(int(num or 10), 100)),
        "gl": gl or "us",
        "hl": hl or "en",
        "api_key": serpapi_key,
        # You can tune these if needed:
        "device": "desktop",
        "google_domain": "google.com",
        "no_cache": True,
    }
    if extra_params:
        params.update(extra_params)

    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
    resp.raise_for_status()
    return resp.json()

# -----------------------------
# OpenAI Chat wrapper
# -----------------------------

def openai_chat(
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    model: str | None = None,               # allow None
    max_tokens: int | None = None,
) -> str:
    """
    Minimal wrapper for OpenAI Chat Completions.
    Returns the assistant message content as a string.
    """
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    # pick model from param, else env, else fallback
    effective_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": effective_model,
        "messages": messages,
        "temperature": float(temperature),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()

    try:
        return data["choices"][0]["message"]["content"] or ""
    except Exception:
        # Fall back to raw text if the schema ever changes
        return json.dumps(data, ensure_ascii=False)

# -----------------------------
# SERP parsing
# -----------------------------

def extract_serp_info(
    serp_json: Dict[str, Any],
) -> Tuple[List[str], str | None, List[str], Union[Dict[str, Any], List[Any], str, None], List[str]]:
    """
    Returns:
        headings: List[str]
        featured_snippet_str: Optional[str]  (a single quick FS line if available)
        related_questions: List[str]
        ai_overview_obj_or_text: FULL AI Overview object (dict/list) if present,
                                 or a plain string if SerpAPI only returns text,
                                 or None if not available
        related_searches: List[str]
    """
    j = serp_json or {}

    # Headings from organic results
    headings: List[str] = []
    for r in (j.get("organic_results") or []):
        t = (r.get("title") or "").strip()
        if t:
            headings.append(t)

    # A single featured snippet string for quick display (we still collect ALL in the page module)
    featured_snippet_str = None
    fs_obj = j.get("featured_snippet") or {}
    ab_obj = j.get("answer_box") or {}
    for obj in (fs_obj, ab_obj):
        if isinstance(obj, dict):
            for k in ("snippet", "answer", "title"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    featured_snippet_str = v.strip()
                    break
        if featured_snippet_str:
            break

    # People Also Ask
    related_questions: List[str] = []
    for item in (j.get("related_questions") or []):
        q = (item.get("question") or "").strip()
        if q:
            related_questions.append(q)

    # >>> FULL AI Overview object (no truncation)
    ai_overview_obj_or_text: Union[Dict[str, Any], List[Any], str, None]
    ai_overview_obj_or_text = (
        j.get("ai_overview")
        or j.get("ai_overview_v2")
        or j.get("ai_overview_results")
        or j.get("sg_results")
        or j.get("sge_results")
        or j.get("ai_answer")
        or j.get("ai_answer_box")
    )
    # If not structured, keep plain string if present
    if not isinstance(ai_overview_obj_or_text, (dict, list)):
        txt = (ai_overview_obj_or_text or "").strip()
        ai_overview_obj_or_text = txt if txt else None

    # Related searches
    related_searches: List[str] = []
    for r in (j.get("related_searches") or []):
        s = (r.get("query") or r.get("text") or "").strip()
        if s:
            related_searches.append(s)

    return headings, featured_snippet_str, related_questions, ai_overview_obj_or_text, related_searches

# -----------------------------
# LLM: semantic terms & clusters
# -----------------------------

def _json_from_llm(text: str) -> Dict[str, Any] | List[Any]:
    """
    Best-effort JSON extractor from model output (handles code fences, pre/post text).
    """
    text = (text or "").strip()
    # try straight parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # strip code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # last resort: first JSON-looking block
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        return json.loads(m.group(1))

    # give up
    raise ValueError("Failed to parse JSON from LLM output")

def extract_semantic_terms_with_llm(
    *,
    headings: List[str] | None,
    snippet: str | None,
    related_questions: List[str] | None,
    ai_overview: Union[Dict[str, Any], List[Any], str, None],
    related_searches: List[str] | None,
    keyword: str,
    max_terms: int = 25,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Ask the LLM to produce a JSON object with:
      - terms: up to `max_terms` short semantic terms/phrases
      - clusters: 4–7 named clusters grouping those terms

    Returns (terms_list, clusters_list).
    """
    # Format AO briefly for the prompt (avoid dumping giant objects)
    if isinstance(ai_overview, (dict, list)):
        def _flatten_ao_preview(x, lim=1200):
            s = json.dumps(x, ensure_ascii=False)[:lim]
            return s + ("…}" if len(s) == lim else "")
        ao_preview = _flatten_ao_preview(ai_overview)
    else:
        ao_preview = (ai_overview or "")[:1200]

    prompt = f"""
You are an SEO/NLP assistant. Given SERP signals for the keyword "{keyword}", produce a JSON object with:
- "terms": an array (max {max_terms}) of short semantic terms/phrases (2-4 words) relevant to the topic.
  Mix entities, modifiers (features/attributes), and common intents. No duplicates, no brand spam unless essential.
  Normalize casing, remove punctuation, avoid pure stopwords.
- "clusters": an array of 4-7 named clusters. Each cluster item has:
  - "name": short label
  - "items": 3-6 of the terms from "terms"

SERP signals (summarized):
- Headings: {(headings or [])[:10]}
- Featured snippet (sample): {snippet or 'None'}
- AI Overview (preview): {ao_preview or 'None'}
- People Also Ask: {(related_questions or [])[:10]}
- Related searches: {(related_searches or [])[:10]}

Return ONLY JSON. No Markdown.
""".strip()

    raw = openai_chat(
        api_key=api_key,
        messages=[
            {"role": "system", "content": "You are a precise SEO assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        model=model,
    )

    data = _json_from_llm(raw)

    # Normalize result
    terms: List[str] = []
    clusters: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        # terms
        for t in (data.get("terms") or []):
            s = str(t).strip().lower()
            if s and s not in terms:
                terms.append(s)
        # clusters
        for c in (data.get("clusters") or []):
            name = str((c.get("name") or "")).strip()
            items = [str(it).strip() for it in (c.get("items") or []) if str(it).strip()]
            clusters.append({"name": name, "items": items})

    # Fallback minimal structure
    if not terms:
        terms = list({kw.strip().lower() for kw in (headings or []) if kw.strip()})[:max_terms]
    if not clusters:
        clusters = [{"name": "Core", "items": terms[: min(6, len(terms))]}]

    return terms[:max_terms], clusters
