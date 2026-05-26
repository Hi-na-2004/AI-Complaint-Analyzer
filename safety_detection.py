"""Centralized safety / harassment detection for hybrid prediction pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Prefix and phrase patterns (case-insensitive) — catches harassing, harassed, ragging, etc.
SAFETY_STEM_PATTERNS: list[tuple[str, str]] = [
    (r"\bharass\w*\b", "harass*"),
    (r"\bragg\w*\b", "ragg*"),
    (r"\bbully(?:ing|ied|ies)?\b", "bully*"),
    (r"\babus\w*\b", "abus*"),
    (r"\bthreat\w*\b", "threat*"),
    (r"\bassault\w*\b", "assault*"),
    (r"\bunsafe\b", "unsafe"),
    (r"\bstalk\w*\b", "stalk*"),
    (r"\bviolenc\w*\b", "violenc*"),
    (r"\bdiscriminat\w*\b", "discriminat*"),
    (r"\bmolest\w*\b", "molest*"),
    (r"\battack\w*\b", "attack*"),
    (r"\bintimidat\w*\b", "intimidat*"),
    (r"\bsexual\s+harass\w*\b", "sexual harassment"),
    (r"\bsenior\s+misconduct\b", "senior misconduct"),
    (r"\binappropriate\s+behav\w*\b", "inappropriate behavior"),
]

# Semantic combos common in campus complaints
SAFETY_SEMANTIC_PATTERNS: list[tuple[str, str]] = [
    (r"\bsenior\b.+\b(junior|fresher|student)\b", "senior vs junior/student"),
    (r"\b(junior|fresher)\b.+\bsenior\b", "junior/fresher vs senior"),
    (r"\bharass\w*\b.+\b(student|junior|fresher)\b", "harassment toward student"),
    (r"\bragg\w*\b.+\b(student|junior|fresher)\b", "ragging toward student"),
    (r"\bfeel\s+unsafe\b", "feel unsafe"),
    (r"\bphysical\s+(abuse|assault|attack)\b", "physical harm"),
]


@dataclass
class SafetySignal:
    is_critical: bool = False
    score: int = 0
    matched_stems: list[str] = field(default_factory=list)
    matched_semantic: list[str] = field(default_factory=list)

    @property
    def matched_all(self) -> list[str]:
        return self.matched_stems + self.matched_semantic

    @property
    def summary(self) -> str:
        if not self.matched_all:
            return "No safety keywords detected"
        return ", ".join(self.matched_all)


def detect_safety_signals(text: str) -> SafetySignal:
    """Detect harassment/safety language with stems and semantic patterns."""
    if not text or not str(text).strip():
        return SafetySignal()

    low = str(text).lower()
    signal = SafetySignal()
    score = 0

    for pattern, label in SAFETY_STEM_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE):
            signal.matched_stems.append(label)
            score += 4

    for pattern, label in SAFETY_SEMANTIC_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE | re.DOTALL):
            signal.matched_semantic.append(label)
            score += 3

    # Senior + harassing/ragging without needing both words adjacent
    if re.search(r"\bsenior\b", low) and re.search(r"\b(junior|fresher)\b", low):
        if re.search(r"\b(harass|ragg|bully|abuse|threat|intimid)\w*\b", low):
            if "senior-junior safety context" not in signal.matched_semantic:
                signal.matched_semantic.append("senior-junior safety context")
                score += 5

    signal.score = score
    signal.is_critical = score >= 4 or len(signal.matched_stems) > 0
    return signal


def extract_safety_tokens_for_ml(text: str) -> list[str]:
    """Pull raw safety terms into ML features so classifiers retain critical words."""
    low = str(text).lower()
    tokens: list[str] = []
    for pattern, _ in SAFETY_STEM_PATTERNS:
        for m in re.finditer(pattern, low, re.IGNORECASE):
            tok = m.group(0).strip()
            if tok and tok not in tokens:
                tokens.append(tok)
    return tokens
