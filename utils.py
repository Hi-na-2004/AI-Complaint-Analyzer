"""Helper utilities: storage, sentiment, urgency, and constants."""

import sqlite3
from datetime import datetime
from pathlib import Path
import re

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from safety_detection import detect_safety_signals

# Complaint categories (must match training labels)
CATEGORIES = [
    "water issue",
    "electricity",
    "fees/finance",
    "harassment",
    "infrastructure",
    "academic issue",
    "other",
]

SENTIMENTS = ["positive", "neutral", "negative"]
URGENCY_LEVELS = ["high", "medium", "low"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "complaints.db"

# Urgency keyword rules (checked on raw lowercased text)
HIGH_URGENCY_KEYWORDS = [
    "harassment", "harassed", "abuse", "abusive", "assault", "threat",
    "threaten", "danger", "dangerous", "emergency", "urgent", "unsafe",
    "safety", "violence", "rape", "molest", "stalk", "bully", "suicide",
    "fire", "flood", "gas leak", "electrocution", "no water", "water cut",
    "power outage", "blackout", "medical", "injury", "hospital",
]

MEDIUM_URGENCY_KEYWORDS = [
    "broken", "leak", "leaking", "damage", "damaged", "not working",
    "failure", "failed", "delay", "delayed", "overcharge", "fee dispute",
    "billing", "refund", "complaint", "unsatisfactory", "poor quality",
    "maintenance", "repair", "outage", "shortage",
]

_CATEGORY_URGENCY_BOOST = {
    "harassment": "high",
    "water issue": "medium",
    "electricity": "medium",
}

_vader = None


def _ensure_vader() -> None:
    global _vader
    if _vader is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _vader = SentimentIntensityAnalyzer()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def init_database() -> None:
    """Create SQLite table for complaints if it does not exist."""
    ensure_data_dir()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_text TEXT NOT NULL,
                category TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                urgency TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def analyze_sentiment(text: str, predicted_category: str | None = None) -> str:
    """Classify sentiment using VADER; safety complaints are always negative."""
    if not text or not str(text).strip():
        return "neutral"

    safety = detect_safety_signals(text)
    if safety.is_critical or predicted_category == "harassment":
        return "negative"

    _ensure_vader()
    scores = _vader.polarity_scores(str(text))
    compound = scores["compound"]

    low = str(text).lower()
    if any(kw in low for kw in SAFETY_CRITICAL_KEYWORDS) and compound < 0.35:
        return "negative"

    if compound >= 0.2:
        return "positive"
    if compound <= -0.2:
        return "negative"
    return "neutral"


HIGH_RISK_PATTERNS = [
    (r"\b(violence|violent|assault|abuse|threat|threaten|unsafe|danger|dangerous)\b", 5),
    (r"\b(electric shock|electrocution|fire|spark|short circuit|gas leak)\b", 5),
    (r"\b(severe leak|major leak|flood|flooding|ceiling collapse|building crack)\b", 5),
    (r"\b(stalking|bullied|bullying|inappropriate behavior|sexual harassment)\b", 5),
]

MODERATE_PATTERNS = [
    (r"\b(repeated|again|every day|every night|for weeks|for months)\b", 1),
    (r"\b(not working|broken|failure|failed|delay|delayed|overcharge|refund pending)\b", 2),
    (r"\b(no water|power cut|outage|leak|unsafe stair|dark corridor)\b", 2),
]

LOW_PATTERNS = [
    (r"\b(suggestion|request|please improve|consider adding|would like)\b", -1),
]

SAFETY_CRITICAL_KEYWORDS = {
    "ragging",
    "harassment",
    "bullying",
    "abuse",
    "unsafe",
    "threat",
    "assault",
    "violence",
    "stalking",
    "discrimination",
    "molestation",
    "attack",
}


def _text_sentiment_score(text: str) -> int:
    _ensure_vader()
    compound = _vader.polarity_scores(str(text))["compound"]
    if compound <= -0.4:
        return 2
    if compound <= -0.2:
        return 1
    if compound >= 0.4:
        return -1
    return 0


def detect_urgency_details(text: str, predicted_category: str) -> dict:
    """Weighted urgency scoring mapped to low/medium/high with reasons."""
    lower = str(text).lower()
    score = 0
    reasons: list[str] = []

    safety = detect_safety_signals(text)
    if safety.is_critical or predicted_category == "harassment":
        score = max(score, 10)
        reasons.append(
            f"Safety override: HIGH urgency ({safety.summary})"
            if safety.matched_all
            else "Safety override: harassment category => HIGH urgency"
        )
        return {"urgency": "high", "score": score, "reasons": reasons, "safety": safety}

    if any(kw in lower for kw in SAFETY_CRITICAL_KEYWORDS):
        score = max(score, 8)
        reasons.append("Safety-critical keyword detected")

    for pattern, weight in HIGH_RISK_PATTERNS:
        if re.search(pattern, lower):
            score += weight
            reasons.append(f"High-risk pattern matched (+{weight})")

    for pattern, weight in MODERATE_PATTERNS:
        if re.search(pattern, lower):
            score += weight
            reasons.append(f"Moderate-risk pattern matched (+{weight})")

    for pattern, weight in LOW_PATTERNS:
        if re.search(pattern, lower):
            score += weight
            reasons.append(f"Low-priority pattern matched ({weight})")

    sentiment_score = _text_sentiment_score(lower)
    score += sentiment_score
    if sentiment_score:
        reasons.append(f"Sentiment adjustment ({sentiment_score:+d})")

    # Category context: harassment can be medium by default, high only when risky language exists.
    if predicted_category == "harassment":
        score += 2
        reasons.append("Category context harassment (+2)")
    elif predicted_category in ("water issue", "electricity", "infrastructure"):
        score += 1
        reasons.append("Category context utility/infrastructure (+1)")

    score = max(0, score)

    if score >= 7:
        urgency = "high"
    elif score >= 3:
        urgency = "medium"
    else:
        urgency = "low"
    return {"urgency": urgency, "score": score, "reasons": reasons}


def detect_urgency(text: str, predicted_category: str) -> tuple[str, int]:
    data = detect_urgency_details(text, predicted_category)
    return data["urgency"], int(data["score"])


def save_complaint(
    complaint_text: str,
    category: str,
    sentiment: str,
    urgency: str,
) -> int:
    """Insert a complaint record; returns new row id."""
    init_database()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO complaints
            (complaint_text, category, sentiment, urgency, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (complaint_text, category, sentiment, urgency, timestamp),
        )
        conn.commit()
        return int(cursor.lastrowid)


def load_all_complaints() -> pd.DataFrame:
    """Load all complaints as a DataFrame."""
    init_database()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, complaint_text, category, sentiment, urgency, created_at
            FROM complaints
            ORDER BY id DESC
            """,
            conn,
        )
    return df


def get_summary_metrics(df: pd.DataFrame) -> dict:
    """Compute dashboard summary counts."""
    if df is None or df.empty:
        return {
            "total": 0,
            "high_urgency": 0,
            "negative_sentiment": 0,
            "top_category": "—",
        }
    high = int((df["urgency"] == "high").sum())
    negative = int((df["sentiment"] == "negative").sum())
    top = df["category"].mode()
    top_cat = top.iloc[0] if len(top) else "—"
    return {
        "total": len(df),
        "high_urgency": high,
        "negative_sentiment": negative,
        "top_category": top_cat,
    }
