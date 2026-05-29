"""Shared rule-based clinical feature extractors used across analyses."""

from __future__ import annotations

import re

_URG3 = re.compile(
    r"(?:call\s+(?:911|999|000|112|an?\s+ambulance|emergency\s+services?)|"
    r"(?:go|head|rush|proceed)\s+(?:to\s+)?(?:the\s+)?(?:ER|A&?E|emergency\s+(?:room|department)|nearest\s+hospital)|"
    r"seek\s+emergency\s+(?:medical\s+)?(?:care|attention|help|treatment)|"
    r"this\s+is\s+(?:a\s+)?(?:medical\s+)?emergency|dial\s+(?:911|999|triple\s+zero))",
    re.IGNORECASE,
)
_URG2 = re.compile(
    r"(?:seek\s+(?:immediate|urgent|prompt)\s+(?:medical\s+)?(?:attention|care|help|treatment|evaluation)|"
    r"seek\s+(?:medical\s+)?(?:attention|care|help|treatment|evaluation)\s+"
    r"(?:immediately|right\s+away|now|today|as\s+soon\s+as\s+possible|urgently|promptly)|"
    r"(?:see|visit|consult)\s+(?:a\s+)?(?:doctor|physician|healthcare\s+provider)\s+"
    r"(?:immediately|right\s+away|now|today|as\s+soon\s+as\s+possible|urgently|promptly)|"
    r"require[sd]?\s+immediate\s+(?:medical\s+)?(?:attention|care|treatment)|"
    r"urgent(?:ly)?\s+(?:see|consult|visit|seek|evaluation)|"
    r"visit\s+an?\s+urgent\s+care\s+(?:centre|center|clinic)\s*"
    r"(?:immediately|right\s+away|now|today|as\s+soon\s+as\s+possible|urgently|promptly)?)",
    re.IGNORECASE,
)
_URG1 = re.compile(
    r"(?:(?:see|visit|consult|speak\s+(?:to|with))\s+(?:a\s+)?(?:doctor|physician|healthcare\s+(?:provider|professional)|GP)|"
    r"(?:consider|suggest|recommend)\s+(?:seeing|visiting|consulting)\s+(?:a\s+)?(?:doctor|physician)|"
    r"(?:get|seek)\s+(?:medical|professional)\s+(?:advice|evaluation|assessment|opinion)|"
    r"(?:your|a)\s+(?:doctor|physician|healthcare\s+provider)\s+(?:can|may|will|should|could))",
    re.IGNORECASE,
)

_MED_NAMED = re.compile(
    r"\b(?:paracetamol|acetaminophen|ibuprofen|aspirin|naproxen|tylenol|advil|motrin|nurofen|panadol"
    r"|nitroglycerin|cetirizine|antihistamine|loperamide|omeprazole)\b",
    re.IGNORECASE,
)
_MED_VAGUE = re.compile(
    r"(?:\bpain\s+(?:reliever|relief|killer|medication)\b|\bover-?the-?counter\b|\bOTC\b|"
    r"\banalgesic\b|\banti-?inflammatory\b|\ban?\s+NSAID\b|\bmedication\b|\bmedicine\b|"
    r"such\s+as\s+ibuprofen)",
    re.IGNORECASE,
)

_EMP2 = re.compile(
    r"(?:thank\s+you\s+for\s+(?:sharing|telling|reaching|trusting|contacting)|"
    r"i(?:'m|\s+am)\s+(?:sorry\s+(?:to\s+hear|you)|here\s+to\s+help|glad\s+you)|"
    r"i\s+understand|i\s+(?:can\s+)?appreciate|i\s+hear\s+you|"
    r"that\s+(?:sounds|must\s+be|can\s+be)\s+(?:very\s+)?(?:concerning|worrying|uncomfortable|painful)|"
    r"your\s+(?:health|wellbeing|concern)\s+(?:is|are)\s+(?:important|my\s+priority)|"
    r"(?:happy|glad|here)\s+to\s+help|feel\s+free\s+to\s+(?:reach\s+out|ask)|"
    r"take\s+care\s+of\s+yourself|(?:wishing|hope)\s+you\s+(?:a\s+)?(?:speedy\s+recovery|feel\s+(?:better|well)\s+soon))",
    re.IGNORECASE,
)
_EMP1 = re.compile(
    r"(?:\bof\s+course\b|\bcertainly\b|\babsolutely\b|\bnoted\b|\bi\s+see\b|\bglad\s+to\b|"
    r"\blet\s+me\s+help\b|\bno\s+problem\b)",
    re.IGNORECASE,
)

WARNING_RX = [re.compile(p, re.IGNORECASE) for p in [
    r"\bshortness\s+of\s+breath\b", r"\bdifficulty\s+breathing\b",
    r"\bchest\s+(?:tightness|pressure|pain)\b",
    r"\bvision\s+(?:changes?|loss|blurring|blurred)\b",
    r"\bconfusion\b|\bdisorientation\b", r"\bweakness\b|\bnumbness\b",
    r"\bfever\b|\bhigh\s+temperature\b", r"\bnausea\b|\bvomiting\b",
    r"\bsweating\b|\bdiaphoresis\b", r"\bdizziness\b|\blightheadedness\b",
    r"\bneck\s+stiffness\b|\bstiff\s+neck\b",
    r"\bpain\s+(?:radiating|spreading)\b|\bradiating\s+pain\b",
    r"\bjaw\s+pain\b|\barm\s+(?:pain|numbness)\b",
    r"\bdifficulty\s+(?:speaking|swallowing|walking)\b|\bslurred\s+speech\b",
    r"\bsudden\s+(?:onset|severe|intense|worst)\b",
    r"\bworst\s+(?:headache|pain)\s+(?:of\s+(?:your|my)\s+life|ever)\b",
    r"\bpalpitations?\b|\bheart\s+(?:racing|pounding)\b",
    r"\bsyncope\b|\bfaint(?:ing)?\b|\bpassed?\s+out\b",
]]
RISK_RX = re.compile(
    r"\b(?:life-?threatening|dangerous|serious|severe|critical|urgent|emergency|deadly|fatal|"
    r"high\s+risk|significant\s+risk|heart\s+attack|stroke|myocardial\s+infarction|"
    r"cardiac\s+arrest|pulmonary\s+embolism|meningitis|sepsis|aneurysm)\b",
    re.IGNORECASE,
)


def urgency_regex(text: str) -> int:
    if _URG3.search(text):
        return 3
    if _URG2.search(text):
        return 2
    if _URG1.search(text):
        return 1
    return 0


def medication_regex(text: str) -> int:
    if _MED_NAMED.search(text):
        return 2
    if _MED_VAGUE.search(text):
        return 1
    return 0


def empathy_regex(text: str) -> int:
    if _EMP2.search(text):
        return 2
    if _EMP1.search(text):
        return 1
    return 0
