"""Text preprocessing for complaint analysis.

Goals:
- Normalize surface forms without over-cleaning.
- Preserve negation ("not", "no") which is critical for meaning.
- Expand common contractions and lemmatize tokens.
"""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOPWORDS: set[str] | None = None
_LEMM: WordNetLemmatizer | None = None


CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "we'd": "we would",
    "they'd": "they would",
}


def _ensure_nltk_data() -> None:
    """Download required NLTK resources if missing."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        _ensure_nltk_data()
        stops = set(stopwords.words("english"))
        # Keep negations; dropping them is a common source of misclassification.
        for neg in ("no", "not", "nor", "never", "none"):
            stops.discard(neg)
        _STOPWORDS = stops
    return _STOPWORDS


def _get_lemmatizer() -> WordNetLemmatizer:
    global _LEMM
    if _LEMM is None:
        _ensure_nltk_data()
        _LEMM = WordNetLemmatizer()
    return _LEMM


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_AMOUNT_RE = re.compile(r"(\$|₹|rs\.?|inr)\s*\d+([,\.]\d+)?|\b\d{2,}\b", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")


def expand_contractions(text: str) -> str:
    t = text
    for c, expanded in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(c)}\b", expanded, t, flags=re.IGNORECASE)
    return t


def normalize_text(text: str) -> str:
    """Lowercase, expand contractions, remove urls/emails, normalize amounts."""
    if not text or not str(text).strip():
        return ""
    t = str(text).strip().lower()
    t = expand_contractions(t)
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _AMOUNT_RE.sub(" amount ", t)
    # Reduce character flooding: "soooo" -> "soo"
    t = _REPEAT_CHAR_RE.sub(r"\1\1", t)
    t = _MULTISPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> list[str]:
    """Tokenize and optionally remove stopwords + lemmatize."""
    _ensure_nltk_data()
    t = normalize_text(text)
    if not t:
        return []

    # Keep punctuation as separators but not as tokens
    t = t.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    t = _MULTISPACE_RE.sub(" ", t).strip()

    tokens = word_tokenize(t)
    if remove_stopwords:
        stops = _get_stopwords()
        tokens = [tok for tok in tokens if tok not in stops]

    # Filter very short tokens but keep 'ai', 'it', 'no'
    tokens = [tok for tok in tokens if (len(tok) > 2 or tok in {"no", "ai"})]

    if lemmatize:
        lemm = _get_lemmatizer()
        tokens = [lemm.lemmatize(tok) for tok in tokens]

    return tokens


def preprocess_for_ml(text: str) -> str:
    """Full pipeline: normalize, tokenize, join for TF-IDF input."""
    from safety_detection import extract_safety_tokens_for_ml

    tokens = tokenize(text, remove_stopwords=True, lemmatize=True)
    safety_tokens = extract_safety_tokens_for_ml(text)
    merged = list(dict.fromkeys(tokens + safety_tokens))  # preserve order, dedupe
    if merged:
        return " ".join(merged)
    return normalize_text(text)
