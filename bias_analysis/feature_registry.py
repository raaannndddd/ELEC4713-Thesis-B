from __future__ import annotations

FIELDS = ("group", "label", "scale", "confirmatory", "derived", "min", "max")


def _f(group, label, scale, confirmatory, derived=False, lo=None, hi=None):
    return dict(zip(FIELDS, (group, label, scale, confirmatory, derived, lo, hi)))


FEATURE_REGISTRY = {
    "urgency_score": _f("Clinical Content", "Urgency Score", "ordinal", True, lo=0, hi=3),
    "referral_score": _f("Clinical Content", "Referral Score", "ordinal", True, lo=0, hi=2),
    "medication_specificity": _f("Clinical Content", "Medication Specificity", "ordinal", True, lo=0, hi=2),
    "empathy_score": _f("Empathy / Tone", "Empathy Score", "ordinal", True, lo=0, hi=2),
    "diagnostic_certainty": _f("Clinical Content", "Diagnostic Certainty", "ordinal", True, lo=0, hi=2),
    "triage_appropriateness": _f("Clinical Content", "Triage Appropriateness", "ordinal", False, True, 0, 2),
    "response_length": _f("Clinical Content", "Response Length (words)", "count", True, lo=0),
    "warning_signs_count": _f("Clinical Content", "Warning Signs", "count", True, lo=0),
    "follow_up_count": _f("Clinical Content", "Follow-up Questions", "count", True, lo=0),
    "risk_language_score": _f("Safety Communication", "Risk Language", "count", True, lo=0),
    "word_count": _f("Length & Structure", "Word Count", "count", False, lo=0),
    "sentence_count": _f("Length & Structure", "Sentence Count", "count", False, lo=0),
    "avg_sentence_length": _f("Length & Structure", "Avg Sentence Length", "continuous", False, lo=0),
    "paragraph_count": _f("Length & Structure", "Paragraph Count", "count", False, lo=0),
    "question_count": _f("Length & Structure", "Question Count", "count", False, lo=0),
    "mattr": _f("Linguistic Complexity", "MATTR (Lexical Diversity)", "continuous", False, lo=0, hi=1),
    "flesch_reading_ease": _f("Linguistic Complexity", "Flesch Reading Ease", "continuous", False),
    "avg_word_length": _f("Linguistic Complexity", "Avg Word Length", "continuous", False, lo=0),
    "medical_term_count": _f("Linguistic Complexity", "Medical Term Count", "count", False, lo=0),
    "differential_count": _f("Clinical Content", "Differential Cues", "count", False, lo=0),
    "emergency_advice": _f("Safety Communication", "Emergency Advice", "binary", False, lo=0, hi=1),
    "safety_warning_count": _f("Safety Communication", "Safety Warnings", "count", False, lo=0),
    "reassurance_count": _f("Empathy / Tone", "Reassurance Count", "count", False, lo=0),
    "politeness_count": _f("Empathy / Tone", "Politeness Count", "count", False, lo=0),
}

CLINICAL_FEATURES = [
    "urgency_score", "referral_score", "medication_specificity", "empathy_score",
    "diagnostic_certainty", "triage_appropriateness", "response_length",
    "warning_signs_count", "follow_up_count", "risk_language_score",
]

QUANTITATIVE_FEATURES = [
    "word_count", "sentence_count", "avg_sentence_length", "paragraph_count",
    "question_count", "mattr", "flesch_reading_ease", "avg_word_length",
    "medical_term_count", "urgency_score", "warning_signs_count",
    "medication_specificity", "differential_count", "emergency_advice",
    "safety_warning_count", "risk_language_score", "empathy_score",
    "reassurance_count", "politeness_count",
]

LONGITUDINAL_RULE_BASED_FEATURES = QUANTITATIVE_FEATURES[:]


def feature_names(*, confirmatory_only=False, include_derived=True, registry=FEATURE_REGISTRY):
    return [
        name for name, spec in registry.items()
        if (include_derived or not spec["derived"])
        and (not confirmatory_only or spec["confirmatory"])
    ]


def feature_label_map(names=None):
    names = names or list(FEATURE_REGISTRY)
    return {name: FEATURE_REGISTRY[name]["label"] for name in names}


def feature_meta_tuples(names):
    return {name: (FEATURE_REGISTRY[name]["group"], FEATURE_REGISTRY[name]["label"]) for name in names}


def ordinal_features(names=None):
    names = names or list(FEATURE_REGISTRY)
    return {name for name in names if FEATURE_REGISTRY[name]["scale"] == "ordinal"}


def count_features(names=None):
    names = names or list(FEATURE_REGISTRY)
    return {name for name in names if FEATURE_REGISTRY[name]["scale"] == "count"}


def binary_features(names=None):
    names = names or list(FEATURE_REGISTRY)
    return {name for name in names if FEATURE_REGISTRY[name]["scale"] == "binary"}


def continuous_features(names=None):
    """Return features with truly continuous scales (not count or binary)."""
    names = names or list(FEATURE_REGISTRY)
    return {name for name in names if FEATURE_REGISTRY[name]["scale"] == "continuous"}
