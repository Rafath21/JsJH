#!/usr/bin/env python3
"""
Top-50 ML/AI jobs for Mariah — one-file MVP
Sources: Greenhouse, Lever, Ashby (public job-board endpoints only)
No DB, no containers. Daily-friendly. macOS-ready.

Usage (via your launcher):
  python src/main.py --config config/mariah.yaml --out out/Top50_Mariah_YYYY-MM-DD.xlsx --seen out/seen_jobs.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from dateutil import parser as dateparser
from rapidfuzz import fuzz, process

# ---------------------------
# Config + constants
# ---------------------------

DEFAULT_TIMEOUT = 20
USER_AGENT = "Top50-JobFinder/1.0 (+personal use; respectful rate limit)"
TODAY = datetime.now(timezone.utc)
DROP_COUNTS = {"seniority":0,"location":0,"employment":0,"negative":0,"age":0}

# Titles & skill keywords tailored to Mariah (from your plan)
TITLE_SYNONYMS = {
    "Machine Learning Engineer": ["ML Engineer", "AI/ML Engineer", "AI Engineer"],
    "Computer Vision Engineer": ["CV Engineer", "Vision Engineer", "Perception Engineer", "Imaging Engineer"],
    "LLM Engineer": ["Generative AI Engineer", "NLP Engineer", "RAG Engineer", "Applied LLM Engineer"],
    "Applied Scientist": ["Applied ML Scientist", "Research Engineer", "Research Scientist (applied)"],
    "AI Engineer": ["AI Software Engineer"],
    "Data Scientist": ["ML Data Scientist", "AI Data Scientist"],
}

SERVING_INFRA = [
    "fastapi","docker","kubernetes","eks","kserve","seldon","bentoml",
    "grpc","onnx","tensorrt","mlflow","airflow","kubeflow","argo cd",
    "terraform","helm","prometheus","grafana","datadog","opentelemetry","batching","autoscaling"
]
CV_KEYWORDS = [
    "computer vision","segmentation","detection","map","yolo","mask r-cnn","opencv",
    "pytorch","tensorflow","onnx","tensorrt","real-time","tracking"
]
LLM_KEYWORDS = [
    "llm","generative ai","transformer","rag","retrieval","faiss","pinecone","vector db",
    "lora","peft","langchain","llamaindex","tool-calling","guardrails","prompt","evaluation harness"
]
GENERAL_ML = [
    "scikit-learn","xgboost","lightgbm","spark","ray","feature store","feast",
    "great expectations","a/b","ab test","drift","evidently"
]

# Company slugs to scan (quick but useful cross-section; add more anytime)
COMPANY_SOURCES = {
    "greenhouse": [
        # AI/Infra/DevTools/SaaS (many are remote-friendly)
        "openai","databricks","stripe","datadog","notion","snowflakeinc","cloudflare",
        "pinterest","airbnb","instacart","robinhood","reddit","discord","roblox",
        "doordash","dropbox","affirm","brex","scaleai","getcruise","andurilindustries",
        "anthropic",  # some orgs moved; if empty, it's skipped gracefully
        "huggingface", "figma", "samsara", "confluent", "ramp", "asana", "plaid",
        "snyk", "coursera", "mongodb", "shopify", "grafana", "hashicorp","openai", "databricks",
    ],
    "lever": [
        "runwayml","retool","rasa","weightsandbiases","octoai","gong","rivet","replicate",
        "mosaicml","scaleai",  # duplicates handled by dedupe
        "hex","prefect","cerebras","snorkelai","loom","samsara", "databricks",
    ],
    "ashby": [
        "Anthropic","Cohere","Mistral","PerplexityAI","ElevenLabs","Adept","Harvey",
        "Roboflow","CharacterAI","LumaAI","Poolside","TogetherAI","HuggingFace",
    ],
}

# --- US-only helpers ---

US_STATE_NAMES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia",
    "hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts",
    "michigan","minnesota","mississippi","missouri","montana","nebraska","nevada","new hampshire","new jersey",
    "new mexico","new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont","virginia","washington","west virginia",
    "wisconsin","wyoming","district of columbia","washington, dc","dc"
}
US_STATE_ABBRS = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia","ks","ky","la","me","md","ma","mi",
    "mn","ms","mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut",
    "vt","va","wa","wv","wi","wy","dc"
}

NON_US_HARD_BLOCK = [
    "canada","emea","europe","eu","united kingdom","uk","ireland","australia","new zealand","anz",
    "apac","asia","japan","korea","singapore","india","latam","mexico","brazil","argentina",
    "colombia","chile","peru","africa","nigeria","south africa","philippines","remote - canada"
]

COUNTRY_TLD_BLOCK = [".co.uk",".de",".fr",".ca",".au",".in",".sg",".jp",".uk",".eu",".ie",".nz",".br",".mx"]

def looks_non_us(text: str) -> bool:
    t = to_lower(text)
    return any(p in t for p in NON_US_HARD_BLOCK)

def url_looks_non_us(url: str) -> bool:
    u = (url or "").lower()
    return any(u.endswith(tld) or f"{tld}/" in u for tld in COUNTRY_TLD_BLOCK)

def has_us_marker(text: str) -> bool:
    t = to_lower(text)
    return bool(re.search(r"\b(united states|u\.s\.a\.|u\.s\.|usa|us)\b", t))

def is_us_location_string(s: str) -> bool:
    t = to_lower(s)
    if looks_non_us(t):
        return False
    if has_us_marker(t):
        return True
    if any(name in t for name in US_STATE_NAMES):
        return True
    # patterns like "Remote - US", "US-Remote"
    if re.search(r"\b(us[-\s]remote|remote[-\s]us)\b", t):
        return True
    return False

def remote_is_us(loc_str: str, desc: str, url: str, strict: bool) -> bool:
    # reject anything explicitly non-US
    if looks_non_us(loc_str) or looks_non_us(desc) or url_looks_non_us(url):
        return False
    # accept if any clear US marker in location or description
    if is_us_location_string(loc_str) or is_us_location_string(desc):
        return True
    # if strict, require an explicit US marker; else allow ambiguous "Remote"
    return not strict


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def http_get(url: str, params: dict | None = None) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
        if r.status_code == 200:
            return r
        return None
    except requests.RequestException:
        return None

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

def to_lower(s: Optional[str]) -> str:
    return norm(s).lower()

def parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    try:
        dt = dateparser.parse(s)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def days_ago(dt: Optional[datetime]) -> Optional[int]:
    if not dt: return None
    return int((TODAY - dt).total_seconds() // 86400)

def detect_seniority(title: str) -> str:
    t = to_lower(title)

    # strong exclusions that truly exceed your band
    if any(x in t for x in ["principal","staff","architect","director","vp","vice president","head of"]):
        return "lead+"

    # treat "lead" IC titles as senior, but avoid "leadership" and "manager"
    if "lead" in t and "leadership" not in t and "manager" not in t and "mgr" not in t:
        return "senior"

    if any(x in t for x in ["senior","sr.","sr "]):
        return "senior"
    if any(x in t for x in ["new grad","entry-level","junior","jr.","jr "]):
        return "junior"
    return "mid"


def seniority_allowed(s: str, min_s: str, max_s: str) -> bool:
    order = ["junior","mid","senior","lead+"]
    i = order.index(s if s in order else "mid")
    return order.index(min_s) <= i <= order.index(max_s)

def guess_work_mode(location_str: str) -> str:
    l = to_lower(location_str)
    if "remote" in l or "anywhere" in l or "distributed" in l:
        return "remote"
    if "hybrid" in l:
        return "hybrid"
    # if it mentions a city only
    return "onsite"

def is_local_stl(location_str: str, radius_km: int = 50) -> bool:
    # String-based heuristic: accept if includes St. Louis or nearby IL side.
    l = to_lower(location_str)
    hints = ["st. louis", "st louis", "saint louis", "chesterfield", "clayton", "creve coeur", "st charles", "st. charles", "edwardsville", "o'fallon", "o fallon", "belleville"]
    states = ["mo", "missouri", "il", "illinois"]
    if any(h in l for h in hints) and any(st in l for st in states):
        return True
    # If location explicitly lists "Missouri" or "MO" without city, accept conservatively.
    if ("missouri" in l or " mo" in l) and ("remote" not in l):
        return True
    return False

def allowed_by_location_and_mode(loc_str: str, mode: str, cfg: dict, desc: str = "", url: str = "") -> bool:
    work_modes = [m.lower() for m in (cfg.get("work_mode") or [])]
    if mode not in work_modes:
        return False

    if mode == "remote":
        if not cfg.get("locations",{}).get("remote_ok", False):
            return False
        strict = bool((cfg.get("locations", {}) or {}).get("remote_strict", False))
        return remote_is_us(loc_str, desc, url, strict)

    if mode in ("hybrid","onsite"):
        regs = cfg.get("locations",{}).get("local_roles_regions",[])
        for r in regs:
            if is_local_stl(loc_str, r.get("radius_km", 50)):
                return True
        return False

    return False


def jaccard_overlap(a_tokens: List[str], b_tokens: List[str]) -> float:
    a = set(a_tokens); b = set(b_tokens)
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\+\#\.]+", to_lower(text))

def short_company(c: str) -> str:
    return re.sub(r"(?i)\s+inc\.?|\.com|,? llc|,? ltd\.?|,? corp\.?|,? co\.?", "", c or "").strip()

def hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# ---------------------------
# Fetchers (ATS)
# ---------------------------

def fetch_greenhouse(slug: str) -> List[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    r = http_get(url, params={"content":"true"})
    if not r: return []
    data = r.json()
    out = []
    for j in data.get("jobs", []):
        title = j.get("title") or ""
        company = slug  # GH does not include company; use slug
        location = (j.get("location") or {}).get("name") or ""
        posted = j.get("updated_at") or j.get("created_at")
        urlj = j.get("absolute_url") or ""
        desc = j.get("content") or ""
        out.append({
            "source": "greenhouse",
            "company": company,
            "title": title,
            "location": location,
            "url": urlj,
            "posted_at": posted,
            "description": desc,
            "employment_type": None,
            "salary_min": None,
            "salary_max": None,
        })
    return out

def fetch_lever(slug: str) -> List[dict]:
    url = f"https://api.lever.co/v0/postings/{slug}"
    r = http_get(url, params={"mode":"json"})
    if not r: return []
    out = []
    for j in r.json():
        title = j.get("text") or ""
        company = j.get("categories", {}).get("team") or slug
        location = (j.get("categories", {}).get("location")) or ""
        urlj = j.get("hostedUrl") or j.get("applyUrl") or ""
        posted = j.get("createdAt")
        if isinstance(posted, (int, float)):
            posted = datetime.utcfromtimestamp(posted/1000).isoformat()
        desc = j.get("descriptionPlain") or j.get("description") or ""
        out.append({
            "source": "lever",
            "company": company,
            "title": title,
            "location": location,
            "url": urlj,
            "posted_at": posted,
            "description": desc,
            "employment_type": (j.get("categories") or {}).get("commitment"),
            "salary_min": None,
            "salary_max": None,
        })
    return out

def fetch_ashby(org: str) -> List[dict]:
    url = f"https://jobs.ashbyhq.com/api/non-user-facing/job-board/{org}"
    r = http_get(url, params={"includeCompensation":"true"})
    if not r: return []
    data = r.json()
    postings = data.get("jobs") or data.get("postings") or []
    out = []
    for j in postings:
        title = j.get("title") or ""
        company = org
        # Locations can be nested; flatten strings we care about
        loc = j.get("location") or j.get("locationName") or j.get("employmentLocation") or ""
        if isinstance(loc, dict):
            loc = loc.get("name") or loc.get("shortName") or ""
        urlj = j.get("jobUrl") or j.get("applyUrl") or j.get("url") or ""
        posted = j.get("publishedAt") or j.get("createdAt")
        # Description text may be in "descriptionHtml" or "content"
        desc = j.get("description") or j.get("descriptionHtml") or j.get("content") or ""
        out.append({
            "source": "ashby",
            "company": company,
            "title": title,
            "location": loc,
            "url": urlj,
            "posted_at": posted,
            "description": desc,
            "employment_type": j.get("employmentType"),
            "salary_min": None,
            "salary_max": None,
        })
    return out

FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
}

# ---------------------------
# Normalization, filtering, scoring
# ---------------------------

def normalize_job(j: dict) -> dict:
    title = norm(j.get("title"))
    company = norm(j.get("company"))
    location = norm(j.get("location") or "Unspecified")
    posted_at = parse_date(j.get("posted_at"))
    desc_html = j.get("description") or ""
    # Strip rudimentary HTML tags for scoring
    desc = re.sub(r"<[^>]+>", " ", desc_html)
    text = f"{title}\n{company}\n{location}\n{desc}"
    tokens = tokenize(text)

    # Seniority + mode
    seniority = detect_seniority(title)
    mode = guess_work_mode(location)

    return {
        "job_id": hash_id(j.get("url") or f"{company}|{title}|{location}|{j.get('source')}"),
        "title": title,
        "company": short_company(company) or company,
        "location": location,
        "work_mode": mode,
        "posted_at": posted_at,
        "salary_min": j.get("salary_min"),
        "salary_max": j.get("salary_max"),
        "currency": "USD",
        "employment_type": (j.get("employment_type") or "").lower() or None,
        "seniority": seniority,
        "tech_stack": list(sorted(set(extract_stack(tokens)))),
        "description": desc.strip(),
        "source": j.get("source"),
        "url": j.get("url") or "",
    }

def extract_stack(tokens: List[str]) -> List[str]:
    bag = " ".join(tokens)
    hits = []
    for kw in SERVING_INFRA + CV_KEYWORDS + LLM_KEYWORDS + GENERAL_ML:
        # normalize keyword to tokens for matching
        k = kw.lower()
        if k in bag:
            hits.append(k)
    return hits

def title_match_score(title: str, target_titles: List[str]) -> float:
    # Max of fuzzy ratios against titles + synonyms
    candidates = []
    for t in target_titles:
        candidates.append(t)
        for syn in TITLE_SYNONYMS.get(t, []):
            candidates.append(syn)
    best = 0
    for cand in candidates:
        r = fuzz.token_set_ratio(title, cand)
        if r > best:
            best = r
    return best / 100.0

def skill_overlap_score(tokens: List[str]) -> float:
    bag = " ".join(tokens)
    weights = 0.0
    # Heavier weights for key strengths
    for kw in SERVING_INFRA:
        if kw in bag: weights += 1.2
    for kw in CV_KEYWORDS:
        if kw in bag: weights += 1.0
    for kw in LLM_KEYWORDS:
        if kw in bag: weights += 1.0
    for kw in GENERAL_ML:
        if kw in bag: weights += 0.6
    # Scale to ~0..1 with a soft cap
    return min(1.0, weights / 10.0)

def location_fit_score(loc: str, mode: str, cfg: dict) -> float:
    if mode == "remote" and cfg.get("locations",{}).get("remote_ok", False):
        return 1.0
    if mode in ("hybrid","onsite") and allowed_by_location_and_mode(loc, mode, cfg):
        # hybrid/onsite within St. Louis area
        return 1.0 if mode == "hybrid" else 0.9
    return 0.0

def recency_score(dt: Optional[datetime]) -> float:
    if not dt: return 0.3  # unknown date — small credit
    days = days_ago(dt)
    if days is None: return 0.3
    if days <= 7: return 1.0
    if days >= 45: return 0.0
    # linear decay 7..45
    return max(0.0, 1.0 - (days - 7) / (45 - 7))

def seniority_score(job_seniority: str, cfg: dict) -> float:
    # exact=1, off-by-one=0.6, else 0.2 (but allow filtering separately)
    desired_min = (cfg.get("seniority") or {}).get("min", "junior")
    desired_max = (cfg.get("seniority") or {}).get("max", "senior")
    order = ["junior","mid","senior","lead+"]
    js = order.index(job_seniority if job_seniority in order else "mid")
    mn = order.index(desired_min); mx = order.index(desired_max)
    if mn <= js <= mx: return 1.0
    if js == mn - 1 or js == mx + 1: return 0.6
    return 0.2

def industry_penalty(company: str, cfg: dict) -> float:
    avoid = [a.lower() for a in (cfg.get("filters", {}).get("industries", {}).get("avoid") or [])]
    c = company.lower()
    # simple heuristic: penalize if "bank" in company name or description keywords later (we only have name here)
    if any(a in ["banking","bank"] for a in avoid) and (" bank" in c or c.endswith(" bank") or "bank " in c):
        return -0.5  # strong penalty
    return 0.0

def negative_keyword_excluded(title: str, desc: str, cfg: dict) -> bool:
    negs = [n.lower() for n in (cfg.get("filters", {}).get("negative_keywords") or [])]
    text = f"{title}\n{desc}".lower()
    for n in negs:
        # word boundary match; hyphen/space friendly (e.g., "intern", "unpaid", "commission", "annotator")
        pattern = rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def employment_type_allowed(et: Optional[str], cfg: dict) -> bool:
    allowed = [x.lower() for x in (cfg.get("filters", {}).get("employment_type") or [])]
    if not allowed: return True
    et = (et or "").lower()
    # map common values
    if "full" in et: et = "full-time"
    if et in allowed: return True
    # If unknown, allow
    return et == "" or et is None

def build_why(title_s: float, skill_s: float, mode: str, rec_s: float, title: str, loc: str, src: str) -> str:
    bits = []
    if title_s >= 0.7: bits.append("strong title match")
    elif title_s >= 0.5: bits.append("good title match")
    if skill_s >= 0.7: bits.append("high skill overlap")
    elif skill_s >= 0.5: bits.append("relevant stack")
    bits.append(mode)
    if rec_s >= 0.7: bits.append("fresh posting")
    return f"{'; '.join(bits)}; {src}"

def score_job(j: dict, cfg: dict) -> Tuple[float, str]:
    title = j["title"]
    desc = j["description"]
    tokens = tokenize(f"{title}\n{desc}")
    title_s = title_match_score(title, cfg.get("titles", {}).get("include") or [])
    skill_s = skill_overlap_score(tokens)
    senior_s = seniority_score(j["seniority"], cfg)
    loc_s = location_fit_score(j["location"], j["work_mode"], cfg)
    rec_s = recency_score(j["posted_at"])
    industry_pen = industry_penalty(j["company"], cfg)

    # Weighted blend
    score = (
        28 * title_s +
        26 * skill_s +
        10 * (1.0 if any(k in j["tech_stack"] for k in SERVING_INFRA) else 0.0) +
         8 * (1.0 if any(k in j["tech_stack"] for k in CV_KEYWORDS) else 0.0) +
         8 * (1.0 if any(k in j["tech_stack"] for k in LLM_KEYWORDS) else 0.0) +
         8 * senior_s +
         6 * loc_s +
         4 * rec_s +
         2 * 0.5  # small constant to separate ties
    ) / 100.0
    score = max(0.0, min(1.0, score + industry_pen))
    why = build_why(title_s, skill_s, j["work_mode"], rec_s, title, j["location"], j["source"])
    return score, why

def passes_hard_filters(j: dict, cfg: dict) -> bool:
    # Seniority band
    if not seniority_allowed(
        j["seniority"],
        cfg.get("seniority", {}).get("min","junior"),
        cfg.get("seniority", {}).get("max","senior")
    ):
        DROP_COUNTS["seniority"] += 1
        return False

    # Work mode + locality/US rules
    if not allowed_by_location_and_mode(j["location"], j["work_mode"], cfg, j.get("description",""), j.get("url","")):
        DROP_COUNTS["location"] += 1
        return False

    # Employment type
    if not employment_type_allowed(j.get("employment_type"), cfg):
        DROP_COUNTS["employment"] += 1
        return False

    # Negative keywords
    if negative_keyword_excluded(j["title"], j["description"], cfg):
        DROP_COUNTS["negative"] += 1
        return False

    # Post age
    max_days = int(cfg.get("post_age_days_max", 45))
    if j["posted_at"]:
        if days_ago(j["posted_at"]) is not None and days_ago(j["posted_at"]) > max_days:
            DROP_COUNTS["age"] += 1
            return False

    return True


def dedupe(jobs: List[dict]) -> List[dict]:
    seen = {}
    out = []
    for j in jobs:
        key = (to_lower(j["company"]), to_lower(j["title"]), to_lower(j["location"]), j["source"])
        if key in seen:
            # keep the one with a date or with longer description
            prev = seen[key]
            def richness(x): return (1 if x.get("posted_at") else 0) + len(x.get("description",""))
            if richness(j) > richness(prev):
                seen[key] = j
        else:
            seen[key] = j
    out = list(seen.values())
    return out

def load_seen(seen_path: Optional[str]) -> set[str]:
    if not seen_path or not os.path.exists(seen_path): return set()
    ids = set()
    with open(seen_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row: ids.add(row[0])
    return ids

def save_seen(seen_path: Optional[str], ids: List[str]):
    if not seen_path: return
    os.makedirs(os.path.dirname(seen_path), exist_ok=True)
    with open(seen_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i in ids:
            w.writerow([i])

# ---------------------------
# Main pipeline
# ---------------------------

def collect_jobs(cfg: dict) -> List[dict]:
    allowed_sources = cfg.get("sources_allowed") or ["greenhouse","lever","ashby"]
    jobs: List[dict] = []
    for source in allowed_sources:
        if source not in FETCHERS: 
            continue
        fetch = FETCHERS[source]
        slugs = COMPANY_SOURCES.get(source, [])
        for slug in slugs:
            r = fetch(slug)
            # polite delay
            time.sleep(0.3)
            for j in r:
                jobs.append(j)
    return jobs

def filter_and_rank(all_jobs: List[dict], cfg: dict, seen_ids: set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Normalize
    normed = [normalize_job(j) for j in all_jobs]
    # Dedup
    uniq = dedupe(normed)

    # Pre-filters (work mode & seniority etc.)
    candidates = []
    for j in uniq:
        if j["job_id"] in seen_ids:
            continue
        if not passes_hard_filters(j, cfg):
            continue
        # Score
        score, why = score_job(j, cfg)
        j["relevance_score"] = round(score * 100, 2)
        j["why_it_matches"] = why
        candidates.append(j)

    # Sort
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Top-50
    top_k = int(cfg.get("top_k", 50))
    top50 = candidates[:top_k]

    # Build DataFrames
    cols = ["title","company","location","work_mode","posted_at","salary_min","salary_max","currency",
            "employment_type","seniority","tech_stack","url","source","relevance_score","why_it_matches"]
    df_all = pd.DataFrame(candidates, columns=cols)
    df_top = pd.DataFrame(top50, columns=cols)
    return df_top, df_all

def export_excel(df_top: pd.DataFrame, df_all: pd.DataFrame, cfg: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ✅ Make copies and strip timezones so Excel is happy
    df_top = df_top.copy()
    df_all = df_all.copy()
    for frame in (df_top, df_all):
        if "posted_at" in frame.columns:
            frame["posted_at"] = pd.to_datetime(frame["posted_at"], errors="coerce").dt.tz_localize(None)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Top 50
        df_top_sorted = df_top.sort_values("relevance_score", ascending=False)
        df_top_sorted.to_excel(writer, sheet_name="Top 50", index=False)

        # All Candidates
        df_all_sorted = df_all.sort_values("relevance_score", ascending=False)
        df_all_sorted.to_excel(writer, sheet_name="All Candidates", index=False)

        # Config Snapshot
        cfg_json = json.dumps(cfg, indent=2)
        pd.DataFrame([{"config": cfg_json}]).to_excel(writer, sheet_name="Config Snapshot", index=False)

        # Run Log
        run_log = pd.DataFrame([{
            "generated_at_utc": TODAY.isoformat(),
            "sources": ", ".join(cfg.get("sources_allowed") or []),
            "top_count": len(df_top_sorted),
            "all_count": len(df_all_sorted),
        }])
        run_log.to_excel(writer, sheet_name="Run Log", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out", required=False, help="Output Excel path")
    ap.add_argument("--seen", required=False, help="CSV to track seen job IDs across runs")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    out_path = args.out or (cfg.get("output", {}).get("excel_path") or f"./out/Top50_Mariah_{datetime.now().date()}.xlsx")
    seen_path = args.seen or (cfg.get("run", {}).get("seen_csv_path"))

    print("[1/4] Fetching jobs (ATS) ...")
    all_jobs = collect_jobs(cfg)
    print(f"    collected: {len(all_jobs)} raw postings")

    print("[2/4] Loading seen IDs ...")
    seen_ids = load_seen(seen_path)
    print(f"    seen so far: {len(seen_ids)}")

    print("[3/4] Filtering, scoring, ranking ...")
    df_top, df_all = filter_and_rank(all_jobs, cfg, seen_ids)

    # Auto-relax...
    if len(df_all) < 50 and cfg.get("locations", {}).get("remote_ok", False):
        print("    too few results — relaxing remote strictness and retrying once...")
        cfg_relaxed = json.loads(json.dumps(cfg))
        cfg_relaxed.setdefault("locations", {})["remote_strict"] = False
        df_top, df_all = filter_and_rank(all_jobs, cfg_relaxed, seen_ids)
        print(f"    after relax: kept {len(df_all)}; top {len(df_top)}")

    print(f"[4/4] Exporting Excel → {out_path}")
    export_excel(df_top, df_all, cfg, out_path)


    print(f"    kept: {len(df_all)} candidates; top: {len(df_top)}")

    print(f"[4/4] Exporting Excel → {out_path}")
    export_excel(df_top, df_all, cfg, out_path)

    # Update seen with the batch we surfaced
    if cfg.get("run", {}).get("persist_seen_in_csv", False) and seen_path:
        save_seen(seen_path, list(df_all["url"].apply(lambda u: hash_id(u or ""))))
        print(f"    appended {len(df_all)} IDs to {seen_path}")

    print("Drop reasons:", DROP_COUNTS)
    print("Done ✅")

if __name__ == "__main__":
    main()
