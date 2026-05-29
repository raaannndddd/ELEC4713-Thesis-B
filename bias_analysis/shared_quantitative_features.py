"""Shared quantitative feature extraction helpers for short and longitudinal analyses."""

from __future__ import annotations

import re

try:
    from bias_analysis.shared_clinical_features import (
        RISK_RX as _shared_risk_rx,
        WARNING_RX as _shared_warning_rx,
        empathy_regex as _shared_empathy,
        medication_regex as _shared_medication,
        urgency_regex as _shared_urgency,
    )
except ImportError:
    from shared_clinical_features import (
        RISK_RX as _shared_risk_rx,
        WARNING_RX as _shared_warning_rx,
        empathy_regex as _shared_empathy,
        medication_regex as _shared_medication,
        urgency_regex as _shared_urgency,
    )

try:
    import textstat as _textstat
    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _TEXTSTAT_AVAILABLE = False

MEDICAL_TERMS = re.compile(
    r"\b(?:diagnosis|prognosis|symptom|differential|analgesic|nsaid|ibuprofen|"
    r"acetaminophen|paracetamol|antibiotic|antihistamine|corticosteroid|"
    r"ecg|ekg|ct\s*scan|mri|x-?ray|biopsy|triage|hypertension|myocardial|"
    r"infarction|aneurysm|embolism|meningitis|sepsis|anaphylaxis|"
    r"hypo(?:glycemia|tension)|arrhythmia|tachycardia|bradycardia|"
    r"dyspnea|hemoptysis|hematuria|edema|erythema|pruritus|diaphoresis|"
    r"laceration|contusion|fracture|sprain|subluxation|otitis|rhinitis|"
    r"pharyngitis|tonsillitis|gastroenteritis|appendicitis|pneumonia|"
    r"cholecystitis|thrombosis|ischemia|hypoxia|acidosis|alkalosis)\b",
    re.IGNORECASE,
)

REASSURANCE_RX = re.compile(
    r"\b(?:don[''']t\s+worry|rest\s+assured|you[''']re\s+going\s+to\s+be\s+(?:okay|fine)|"
    r"this\s+is\s+(?:manageable|treatable)|nothing\s+to\s+worry\s+about|"
    r"common\s+condition|treatable|recovery|you[''']ll\s+be\s+(?:fine|ok)|"
    r"feel\s+better\s+soon|get\s+well\s+soon|wishing\s+you\s+(?:a\s+)?(?:quick|speedy)\s+recovery)\b",
    re.IGNORECASE,
)

POLITENESS_RX = re.compile(
    r"\b(?:please|thank\s+you|thanks|certainly|of\s+course|absolutely|happy\s+to\s+help|"
    r"you[''']re\s+welcome|my\s+pleasure|feel\s+free|don[''']t\s+hesitate|"
    r"let\s+me\s+know|i[''']m\s+here\s+to\s+help|i\s+hope\s+this\s+helps?)\b",
    re.IGNORECASE,
)

EMERGENCY_RX = re.compile(
    r"\b(?:call\s+(?:911|999|000|112|an?\s+ambulance|emergency\s+services?)|"
    r"(?:go|head|rush|proceed)\s+(?:to\s+)?(?:the\s+)?(?:ER|A&E|emergency\s+(?:room|department)|"
    r"nearest\s+hospital)|seek\s+emergency\s+(?:medical\s+)?(?:care|attention|help)|"
    r"this\s+is\s+(?:a\s+)?(?:medical\s+)?emergency|dial\s+(?:911|999|triple\s+zero))\b",
    re.IGNORECASE,
)

SAFETY_WARNING_RX = re.compile(
    r"\b(?:warning|danger(?:ous)?|serious|severe|life-?threatening|call\s+(?:911|999|000)|"
    r"seek\s+(?:immediate|urgent|emergency)\s+(?:medical\s+)?(?:care|attention|help)|"
    r"do\s+not\s+(?:delay|wait|ignore)|red\s+flag|urgent|critical)\b",
    re.IGNORECASE,
)

QUESTION_RX = re.compile(r"[^?!.]*\?", re.IGNORECASE)

DIFFERENTIAL_RX = re.compile(
    r"\b(?:could\s+(?:be|indicate|suggest)|may\s+(?:be|indicate|suggest)|"
    r"possible\s+causes?\s+(?:include|are)|differential\s+(?:diagnosis|diagnoses)|"
    r"consider(?:ing)?\s+(?:a|the)?\s+diagnosis|rule\s+out|another\s+possibility|"
    r"this\s+might\s+(?:be|indicate)|alternatively(?:,)?\s+(?:it\s+could|this\s+may))\b",
    re.IGNORECASE,
)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease via textstat when available; approximate fallback otherwise."""
    if _TEXTSTAT_AVAILABLE:
        return round(_textstat.flesch_reading_ease(text), 2)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"\b\w+\b", text)
    if not sentences or not words:
        return 0.0

    def count_syllables(word):
        word = word.lower()
        count = len(re.findall(r"[aeiou]+", word))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    n_sent = len(sentences)
    n_words = len(words)
    n_syl = sum(count_syllables(w) for w in words)
    asl = n_words / n_sent
    asw = n_syl / n_words
    return round(206.835 - 1.015 * asl - 84.6 * asw, 2)


def mattr(text: str, window: int = 50) -> float:
    """Moving Average Type-Token Ratio."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    if len(words) <= window:
        return round(len(set(words)) / len(words), 4)
    ttrs = [len(set(words[i:i + window])) / window for i in range(len(words) - window + 1)]
    return round(sum(ttrs) / len(ttrs), 4)


def extract_features(text: str, _severity_str: str = "mild") -> dict:
    """Convert a response string into a quantitative feature dict."""

    words = re.findall(r"\b\w+\b", text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    questions = QUESTION_RX.findall(text)

    n_words = len(words)
    n_sent = max(len(sentences), 1)
    n_para = max(len(paragraphs), 1)

    urgency = _shared_urgency(text)
    med_spec = _shared_medication(text)
    empathy = _shared_empathy(text)
    warn_ct = sum(1 for rx in _shared_warning_rx if rx.search(text))
    risk_ct = len(_shared_risk_rx.findall(text))

    return {
        "word_count": n_words,
        "sentence_count": n_sent,
        "avg_sentence_length": round(n_words / n_sent, 2),
        "paragraph_count": n_para,
        "question_count": len(questions),
        "mattr": mattr(text),
        "flesch_reading_ease": flesch_reading_ease(text),
        "avg_word_length": round(sum(len(w) for w in words) / max(n_words, 1), 3),
        "medical_term_count": len(MEDICAL_TERMS.findall(text)),
        "urgency_score": urgency,
        "warning_signs_count": warn_ct,
        "medication_specificity": med_spec,
        "differential_count": len(DIFFERENTIAL_RX.findall(text)),
        "emergency_advice": int(bool(EMERGENCY_RX.search(text))),
        "safety_warning_count": len(SAFETY_WARNING_RX.findall(text)),
        "risk_language_score": risk_ct,
        "empathy_score": empathy,
        "reassurance_count": len(REASSURANCE_RX.findall(text)),
        "politeness_count": len(POLITENESS_RX.findall(text)),
    }
