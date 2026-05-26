"""Batch evaluation helpers for complaint classifier."""

from __future__ import annotations

import pandas as pd

from model import load_model, predict_category_with_debug
from safety_detection import detect_safety_signals
from utils import analyze_sentiment, detect_urgency_details


def run_sample_checks(samples: list[str]) -> pd.DataFrame:
    """Run end-to-end prediction on sample texts."""
    model = load_model()
    rows = []
    for text in samples:
        pred = predict_category_with_debug(text, model)
        safety = detect_safety_signals(text)
        sentiment = analyze_sentiment(text)
        urgency = detect_urgency_details(text, pred["category"])
        rows.append(
            {
                "text": text,
                "category": pred["category"],
                "confidence": pred["confidence"],
                "sentiment": sentiment,
                "urgency": urgency["urgency"],
                "safety_matches": safety.summary,
            }
        )
    return pd.DataFrame(rows)
