"""Model training, evaluation, and prediction utilities."""

from __future__ import annotations

import pickle
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocess import preprocess_for_ml
from safety_detection import detect_safety_signals
from utils import CATEGORIES, DATA_DIR

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
DATASET_PATH = DATA_DIR / "complaints_dataset.csv"
EVAL_SPLIT_PATH = DATA_DIR / "evaluation_split.csv"

# Below this, non-safety predictions fall back to second-best class (not blindly "other")
DEFAULT_THRESHOLD = 0.18
SECOND_BEST_MARGIN = 0.04


def _build_dataset() -> pd.DataFrame:
    """Create large balanced synthetic dataset (100 per class)."""
    rng = random.Random(42)

    templates = {
        "water issue": [
            "there is {problem} in {location} since {duration}",
            "students reported {problem} at {location} repeatedly",
            "{location} has {problem} and maintenance has not fixed it for {duration}",
            "we are facing {problem} in {location} almost every day",
        ],
        "electricity": [
            "there is {problem} in {location} during {time_period}",
            "{location} keeps having {problem} for {duration}",
            "we noticed {problem} near {location} and this is unsafe",
            "{problem} in {location} is affecting classes for {duration}",
        ],
        "fees/finance": [
            "the {problem} for my account is unresolved for {duration}",
            "finance office has not fixed {problem} despite multiple reminders",
            "i was charged {problem} and need correction urgently",
            "there is {problem} related to {location} payment",
        ],
        "harassment": [
            "i experienced {problem} at {location} and feel unsafe",
            "there was {problem} by a person in {location} for {duration}",
            "please take action on {problem} happening around {location}",
            "{problem} in {location} is creating a threatening environment",
        ],
        "infrastructure": [
            "{location} has {problem} and needs maintenance",
            "there is {problem} in {location} for {duration}",
            "students are affected because of {problem} at {location}",
            "please repair {problem} in {location} soon",
        ],
        "academic issue": [
            "there is {problem} in {location} this semester",
            "our class is facing {problem} for {duration}",
            "{problem} in {location} is affecting academic progress",
            "please resolve {problem} in the academic system",
        ],
        "other": [
            "i have a {problem} regarding {location}",
            "please look into {problem} for {location}",
            "{problem} has caused inconvenience for {duration}",
            "this is a general complaint about {problem} in {location}",
        ],
    }

    problems = {
        "water issue": [
            "no water supply",
            "low water pressure",
            "dirty drinking water",
            "severe water leakage",
            "water tank overflow",
            "foul smell from water",
            "pipeline blockage",
            "intermittent water cut",
            "contaminated tap water",
            "bathroom tap not working",
            "hostel washroom water shortage",
            "drain water backflow",
        ],
        "electricity": [
            "power outage",
            "frequent voltage fluctuation",
            "electric shock from socket",
            "sparks in wiring",
            "classroom fan not working",
            "light failure in corridor",
            "generator backup failure",
            "short circuit near panel",
            "broken switch board",
            "lab equipment not powered",
            "dark stairway due to no lights",
            "sudden blackout at night",
        ],
        "fees/finance": [
            "wrong fee amount",
            "duplicate payment deduction",
            "refund pending",
            "scholarship not credited",
            "late fee charged incorrectly",
            "invoice mismatch",
            "payment receipt missing",
            "hostel fee overcharge",
            "financial aid rejection without reason",
            "exam fee still showing unpaid",
            "double transaction issue",
            "installment plan not updated",
        ],
        "harassment": [
            "verbal abuse",
            "threatening messages",
            "bullying by seniors",
            "ragging by senior students",
            "stalking near hostel",
            "inappropriate behavior by staff",
            "discriminatory comments",
            "sexual harassment complaint",
            "senior misconduct during orientation",
            "intimidation in class",
            "unsafe interaction in corridor",
            "public humiliation",
            "coercive behavior",
            "abusive language in office",
            "physical assault report",
            "violent threatening behavior",
            "senior harassing a junior",
            "some senior is harassing a junior on fifth floor",
            "a senior is ragging freshmen in hostel",
            "senior students bullying juniors after class",
            "threatening behavior by senior toward junior",
            "harassing messages from senior student",
            "junior student harassed in corridor",
            "senior intimidating fresher students",
        ],
        "infrastructure": [
            "broken classroom chairs",
            "damaged ceiling tiles",
            "lift not working",
            "toilet door broken",
            "library air conditioning failure",
            "poor ventilation",
            "potholes on internal road",
            "water seepage in walls",
            "unsafe staircase railing",
            "collapsed false ceiling panel",
            "broken lab benches",
            "campus wifi dead zone",
        ],
        "academic issue": [
            "exam timetable clash",
            "attendance record error",
            "marks not updated",
            "course registration bug",
            "syllabus not completed",
            "no faculty replacement",
            "assignment grading inconsistency",
            "lab slots unavailable",
            "practical exam delay",
            "project review not scheduled",
            "unfair invigilation complaint",
            "portal not allowing submission",
        ],
        "other": [
            "bus schedule confusion",
            "canteen menu complaint",
            "parking permit delay",
            "id card replacement delay",
            "lost and found support needed",
            "guest pass process issue",
            "event management noise concern",
            "sports facility timing request",
            "website login slowdown",
            "mobile app glitch report",
            "request for additional seating",
            "suggestion for cleaner notice board",
        ],
    }

    locations = [
        "hostel block a",
        "hostel block b",
        "main library",
        "science lab",
        "computer lab",
        "admin office",
        "fee counter",
        "auditorium",
        "canteen",
        "main gate",
        "north corridor",
        "classroom 203",
        "classroom 105",
        "engineering block",
        "parking area",
    ]
    durations = [
        "two days",
        "three days",
        "one week",
        "two weeks",
        "several weeks",
        "one month",
        "many days",
    ]
    time_periods = ["evening hours", "exam time", "night", "morning classes"]

    rows: list[dict[str, str]] = []
    samples_per_category = 100

    for category in CATEGORIES:
        cat_templates = templates[category]
        cat_problems = problems[category]
        for i in range(samples_per_category):
            tpl = cat_templates[i % len(cat_templates)]
            phrase = tpl.format(
                problem=cat_problems[i % len(cat_problems)],
                location=locations[(i + rng.randint(0, len(locations) - 1)) % len(locations)],
                duration=durations[(i + rng.randint(0, len(durations) - 1)) % len(durations)],
                time_period=time_periods[(i + rng.randint(0, len(time_periods) - 1)) % len(time_periods)],
            )
            # Add light natural variation
            if i % 3 == 0:
                phrase = f"please help, {phrase}"
            elif i % 3 == 1:
                phrase = f"{phrase}, this keeps repeating."
            else:
                phrase = f"{phrase}. request immediate action."
            rows.append({"complaint_text": phrase, "category": category})

    # Hand-authored realistic harassment/ragging complaints (varied wording)
    harassment_extras = [
        "some senior is harassing a junior on 5th floor",
        "a senior is harassing ragging a student in hostel block",
        "senior students are bullying juniors every night",
        "my friend was threatened and abused by a senior",
        "there is ragging happening on the fifth floor corridor",
        "a senior made inappropriate comments and i feel unsafe",
        "violence and intimidation by seniors against freshers",
        "stalking and harassment near the girls hostel",
        "senior misconduct during induction week needs action",
        "discrimination and verbal abuse in classroom",
        "physical assault reported between senior and junior",
        "senior is harassing junior student please help urgently",
        "bullying and ragging incident on floor five",
        "threatening messages from senior batch students",
        "unsafe environment due to senior harassment",
    ]
    for phrase in harassment_extras:
        rows.append({"complaint_text": phrase, "category": "harassment"})

    df = pd.DataFrame(rows)
    return df


def ensure_dataset(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists() and not force:
        return pd.read_csv(DATASET_PATH)
    df = _build_dataset()
    df.to_csv(DATASET_PATH, index=False)
    return df


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=12000,
    )


def _build_classifier() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42,
    )


def train_model(force: bool = False) -> dict:
    """Train model with train/test split and save artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = ensure_dataset(force=False)
    dataset["processed"] = dataset["complaint_text"].map(preprocess_for_ml)

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["processed"],
        dataset["category"],
        test_size=0.2,
        random_state=42,
        stratify=dataset["category"],
    )
    eval_df = pd.DataFrame({"text": x_test, "actual": y_test})
    eval_df.to_csv(EVAL_SPLIT_PATH, index=False)

    if CLASSIFIER_PATH.exists() and VECTORIZER_PATH.exists() and not force:
        clf, vec = load_model()
        return evaluate_model(clf, vec, x_test, y_test)

    vectorizer = _build_vectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = _build_classifier()
    classifier.fit(x_train_vec, y_train)

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(classifier, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    return evaluate_model(classifier, vectorizer, x_test, y_test)


def load_model():
    """Load classifier and vectorizer (train first if missing)."""
    if not (CLASSIFIER_PATH.exists() and VECTORIZER_PATH.exists()):
        train_model(force=True)
    with open(CLASSIFIER_PATH, "rb") as f:
        classifier = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return classifier, vectorizer


def predict_category_with_debug(
    text: str,
    model_tuple=None,
    confidence_threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Predict category with confidence and explainable override reasons."""
    if model_tuple is None:
        model_tuple = load_model()
    classifier, vectorizer = model_tuple

    processed = preprocess_for_ml(text)
    if not processed.strip():
        return {
            "category": "other",
            "confidence": 0.0,
            "reasons": ["Empty complaint after preprocessing"],
            "probabilities": pd.DataFrame({"category": CATEGORIES, "probability": [0.0] * len(CATEGORIES)}),
        }

    vec = vectorizer.transform([processed])
    proba = classifier.predict_proba(vec)[0]
    classes = list(classifier.classes_)
    top_idx = int(proba.argmax())
    model_label = str(classes[top_idx])
    confidence = float(proba[top_idx])
    reasons = [f"Model top prediction: {model_label} ({confidence * 100:.1f}%)"]

    safety = detect_safety_signals(text)
    if safety.matched_all:
        reasons.append(f"Matched safety signals: {safety.summary}")

    label = model_label

    # PRIORITY 1: Safety override — never classify obvious harassment as "other"
    if safety.is_critical:
        if label != "harassment":
            reasons.append(
                "RULE OVERRIDE: safety-critical language detected => harassment (HIGH priority)"
            )
        label = "harassment"
        confidence = max(confidence, 0.85)
    elif confidence < confidence_threshold:
        # PRIORITY 2: Low-confidence correction — use 2nd best, not random "other"
        order = proba.argsort()[::-1]
        if len(order) > 1:
            second_idx = int(order[1])
            second_label = str(classes[second_idx])
            second_conf = float(proba[second_idx])
            if second_conf >= confidence_threshold - SECOND_BEST_MARGIN:
                reasons.append(
                    f"Low confidence correction: switched {model_label} -> {second_label} "
                    f"({second_conf * 100:.1f}%)"
                )
                label = second_label
                confidence = second_conf
            else:
                reasons.append("Low confidence: kept top class (no strong second candidate)")
        if label not in CATEGORIES:
            label = "other"

    probs_df = (
        pd.DataFrame({"category": classes, "probability": proba})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "category": label,
        "confidence": confidence,
        "reasons": reasons,
        "probabilities": probs_df,
        "safety": safety,
        "model_label": model_label,
        "model_confidence": float(proba[top_idx]),
    }


def predict_category(
    text: str,
    model_tuple=None,
    confidence_threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str, float]:
    """Predict category with anti-bias logic and confidence fallback."""
    result = predict_category_with_debug(text, model_tuple, confidence_threshold)
    return str(result["category"]), float(result["confidence"])


def get_category_probabilities(text: str, model_tuple=None) -> pd.DataFrame:
    """Return ordered probability distribution for all categories."""
    if model_tuple is None:
        model_tuple = load_model()
    classifier, vectorizer = model_tuple

    processed = preprocess_for_ml(text)
    if not processed.strip():
        return pd.DataFrame({"category": CATEGORIES, "probability": [0.0] * len(CATEGORIES)})

    vec = vectorizer.transform([processed])
    proba = classifier.predict_proba(vec)[0]
    classes = list(classifier.classes_)
    return (
        pd.DataFrame({"category": classes, "probability": proba})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model(classifier, vectorizer, x_test, y_test) -> dict:
    """Compute report, confusion matrix, and misclassification insights."""
    x_test_vec = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test_vec)
    y_prob = classifier.predict_proba(x_test_vec)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)

    cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
    cm_df = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)

    pred_conf = y_prob.max(axis=1)
    diag = pd.DataFrame(
        {
            "text": list(x_test),
            "actual": list(y_test),
            "predicted": list(y_pred),
            "confidence": pred_conf,
        }
    )
    misclassified = (
        diag[diag["actual"] != diag["predicted"]]
        .sort_values("confidence", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    samples = diag.sample(min(12, len(diag)), random_state=42).reset_index(drop=True)
    accuracy = float((diag["actual"] == diag["predicted"]).mean())

    return {
        "accuracy": accuracy,
        "classification_report_df": report_df,
        "confusion_matrix_df": cm_df,
        "misclassified_df": misclassified,
        "sample_predictions_df": samples,
    }
