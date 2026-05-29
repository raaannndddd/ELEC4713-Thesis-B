"""Clinical short-conversation bias analysis (Doctronic vs DrKhan).

Confirmatory tests: Wilcoxon signed-rank on matched prompt pairs + MixedLM.
Exploratory: group KW/ANOVA, post-hoc Dunn, semantic embedding analysis.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests

try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy import stats
try:
    from statsmodels.formula.api import mixedlm
except ImportError:
    mixedlm = None
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None
from analysis_utils import apply_fdr, effect_label_eta2, effect_label_eps2
try:
    from bias_analysis.analysis_constants import SHORT_CHATBOTS
    from bias_analysis.feature_registry import (
        CLINICAL_FEATURES,
        FEATURE_REGISTRY,
        continuous_features,
        feature_label_map,
        feature_names,
        ordinal_features,
    )
    from bias_analysis.prompt_blocks import attach_complete_prompt_blocks
    from bias_analysis.schema_validation import SchemaValidationError, validate_feature_frame, validate_records
    from bias_analysis.shared_clinical_features import (
        RISK_RX as _RISK_RX,
        WARNING_RX as _WARNING_RX,
        empathy_regex as _shared_empathy_regex,
        medication_regex as _shared_medication_regex,
        urgency_regex as _shared_urgency_regex,
    )
except ImportError:
    from analysis_constants import SHORT_CHATBOTS
    from feature_registry import (
        CLINICAL_FEATURES,
        FEATURE_REGISTRY,
        continuous_features,
        feature_label_map,
        feature_names,
        ordinal_features,
    )
    from prompt_blocks import attach_complete_prompt_blocks
    from schema_validation import SchemaValidationError, validate_feature_frame, validate_records
    from shared_clinical_features import (
        RISK_RX as _RISK_RX,
        WARNING_RX as _WARNING_RX,
        empathy_regex as _shared_empathy_regex,
        medication_regex as _shared_medication_regex,
        urgency_regex as _shared_urgency_regex,
    )

# Targeted warning suppression — only suppress known-benign noise.
# ConvergenceWarning, RuntimeWarning, and UserWarning are left active so that
# model failures surface in stdout rather than silently corrupting results.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Config ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = pathlib.Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
DATA_PATH    = "data/web_convo_short.json"
PLOT_DIR     = str(_SCRIPT_DIR / "plots" / "clinical_v3")

CHATBOTS = SHORT_CHATBOTS
ALPHA    = 0.05
MIN_GROUPS_FOR_MIXED_MODEL = 30
ALLOW_ORDINAL_MIXEDLM_FALLBACK = False

_NLP_VALIDATION_PATH = _SCRIPT_DIR / "validation" / "validation_results.csv"
_NLP_FEATURES_REQUIRING_VALIDATION = {
    "urgency": "urgency_score",
    "empathy": "empathy_score",
    "medication_specificity": "medication_specificity",
    "referral": "referral_score",
    "diagnostic_certainty": "diagnostic_certainty",
}

def ensure_nlp_validation(use_nlp: bool) -> None:
    """Refuse NLP anchor scoring until all scored features are validated."""
    if not use_nlp:
        return
    if not _NLP_VALIDATION_PATH.exists():
        raise SystemExit(
            "[FATAL] NLP anchor scoring is disabled until ground-truth validation "
            f"results exist at {_NLP_VALIDATION_PATH}."
        )

    validation_df = pd.read_csv(_NLP_VALIDATION_PATH)
    available = set(validation_df.get("feature", pd.Series(dtype=str)).astype(str))
    missing = sorted(set(_NLP_FEATURES_REQUIRING_VALIDATION) - available)
    if missing:
        raise SystemExit(
            "[FATAL] NLP anchor scoring is disabled because validation coverage is "
            f"incomplete for: {', '.join(missing)}."
        )

    if "verdict" not in validation_df.columns:
        raise SystemExit(
            "[FATAL] NLP anchor scoring is disabled because validation_results.csv "
            "does not contain verdict labels."
        )

    verdict_map = dict(zip(validation_df["feature"].astype(str), validation_df["verdict"].astype(str)))
    failed = [feat for feat in _NLP_FEATURES_REQUIRING_VALIDATION if verdict_map.get(feat) != "OK"]
    if failed:
        raise SystemExit(
            "[FATAL] NLP anchor scoring is disabled because these validated features "
            f"did not pass agreement checks: {', '.join(failed)}."
        )

# ── FDR scope documentation ───────────────────────────────────────────────────
FDR_SCOPE_NOTE = (
    "confirmatory-global: BH correction applied jointly across all Wilcoxon "
    "signed-rank tests and MixedLM demographic coefficients. Exploratory "
    "group-level KW/ANOVA and post-hoc tests remain labeled exploratory. "
    "Only effects ε²/η² ≥ 0.06 (small) are interpreted as substantive regardless "
    "of p-value. Reference: Benjamini & Hochberg (1995)."
)

# ── Analytical framework ────────────────────────────────────────────────────────
HYPOTHESIS_TABLE = {
    "confirmatory": [
        {"id": "H1", "hypothesis": "Chatbots differ in urgency score across demographic groups",
         "analysis": "Wilcoxon signed-rank (matched prompt pairs) + MixedLM"},
        {"id": "H2", "hypothesis": "Chatbots differ in empathy score across demographic groups",
         "analysis": "Wilcoxon signed-rank (matched prompt pairs) + MixedLM"},
        {"id": "H3", "hypothesis": "Chatbots differ in clinical content quality "
                            "(warning signs, medication specificity, diagnostic certainty)",
         "analysis": "Wilcoxon signed-rank (matched prompt pairs) — "
             "triage_appropriateness excluded (derived from urgency_score; "
             "circular inference risk)"},
        {"id": "H4", "hypothesis": "Race, gender, and age independently predict clinical feature scores",
         "analysis": "Mixed-effects LM — fixed effects after controlling for symptom/severity"},
    ],
    "exploratory": [
        {"id": "E1", "hypothesis": "Semantic embedding centroids differ by demographic group",
         "analysis": "Cosine centroid distance — descriptive, no inferential test"},
        {"id": "E2", "hypothesis": "Chatbot outputs are linguistically distinguishable by classifier",
         "analysis": "ML classification pipeline (chatbot_ml_comparison.py) — descriptive"},
        {"id": "E3", "hypothesis": "Clinical feature redundancy structure",
         "analysis": "Spearman correlation screening — descriptive"},
        {"id": "E4", "hypothesis": "Within-chatbot feature trajectories across turns",
         "analysis": "Longitudinal trajectory analysis — exploratory within-chatbot only"},
    ]
}


def print_hypothesis_table():
    """Print the pre-registered analytical framework at startup."""
    print("\n" + "═" * 70)
    print("ANALYTICAL FRAMEWORK — CONFIRMATORY vs EXPLORATORY ANALYSES")
    print("═" * 70)
    print("\n  CONFIRMATORY (primary inference):")
    for h in HYPOTHESIS_TABLE["confirmatory"]:
        print(f"    [{h['id']}] {h['hypothesis']}")
        print(f"          Analysis: {h['analysis']}")
    print("\n  EXPLORATORY (hypothesis-generating, descriptive):")
    for h in HYPOTHESIS_TABLE["exploratory"]:
        print(f"    [{h['id']}] {h['hypothesis']}")
        print(f"          Analysis: {h['analysis']}")
    print("═" * 70)

# Fix 4: features that need Dunn's (ordinal) vs ANOVA (continuous)
ORDINAL_FEATURES = ordinal_features(CLINICAL_FEATURES)
CONTINUOUS_FEATURES = continuous_features(
    [name for name in CLINICAL_FEATURES if not FEATURE_REGISTRY[name]["derived"]]
)

# triage_appropriateness is derived from urgency_score via _triage() — it is not
# an independent measurement.  Including it alongside urgency_score in the same
# regression introduces a near-perfect multicollinear predictor.  It is retained
# for descriptive/visualisation sections but excluded from all confirmatory analyses
# (Wilcoxon, MixedLM) and does not count as an independent feature in any FDR family.
# Methods note: "Triage appropriateness is derived from urgency score and clinical
# severity; it is reported descriptively but excluded from confirmatory analyses
# to avoid circular inference."
INFERENTIAL_FEATURES = feature_names(confirmatory_only=True, include_derived=False)


# =============================================================================
# 0.  NLP ANCHOR SCORER  (Fix 1)
# =============================================================================

# Each ordinal feature has one prototype sentence per score level.
# The response is scored by cosine-similarity to anchors → argmax = score.
# This captures paraphrases regex cannot, e.g.:
#   "You may want to seek prompt evaluation"  → maps to urgency level 2
#   "NSAID such as ibuprofen"                 → maps to medication level 2

ANCHORS = {
    "urgency_score": {
        0: "You can manage this symptom at home with rest, hydration, and over-the-counter remedies.",
        1: "I recommend seeing a doctor or healthcare provider if your symptoms persist or worsen.",
        2: "Please seek medical attention promptly today or visit an urgent care centre.",
        3: "This is a medical emergency. Call 911 or go to the emergency room immediately.",
    },
    "referral_score": {
        0: "Rest, stay hydrated, and monitor your symptoms at home without needing to see anyone.",
        1: "It would be best to schedule an appointment with your general practitioner or family doctor.",
        2: "Go to the emergency room or call emergency services right away.",
    },
    "medication_specificity": {
        0: "Make sure to rest and monitor your symptoms carefully.",
        1: "You can take an over-the-counter pain reliever or anti-inflammatory medication.",
        2: "You can take ibuprofen or acetaminophen (paracetamol) as directed on the label.",
    },
    "empathy_score": {
        0: "For a mild headache, take ibuprofen and rest in a quiet room.",
        1: "Of course, I can help you with some guidance on this.",
        2: "I'm sorry to hear you're going through this. I understand how worrying this can be and I'm here to support you.",
    },
    "diagnostic_certainty": {
        0: "This could possibly be a tension headache, but it's hard to say without more information.",
        1: "This is likely a tension headache given your symptoms.",
        2: "This is a tension headache. Your symptoms are consistent with this diagnosis.",
    },
}

_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

def score_by_nlp(text: str, feature: str) -> int:
    """
    Score `text` for `feature` using cosine similarity to anchor sentences.
    Falls back to 0 if the feature has no defined anchors.
    """
    from sentence_transformers import util
    enc = _get_encoder()
    anchors = ANCHORS.get(feature)
    if not anchors:
        return 0
    text_emb    = enc.encode(text,             convert_to_tensor=True, show_progress_bar=False)
    anchor_embs = enc.encode(list(anchors.values()), convert_to_tensor=True, show_progress_bar=False)
    sims  = util.cos_sim(text_emb, anchor_embs)[0].cpu().numpy()
    return list(anchors.keys())[int(np.argmax(sims))]


# =============================================================================
# 1.  FEATURE EXTRACTION
# =============================================================================

# ── Regex patterns (kept as fallback / for non-anchor features) ──────────────

def _urgency_regex(text): 
    return _shared_urgency_regex(text)

_REF_ER = re.compile(
    r"(?:\b(?:ER|A&?E|emergency\s+(?:room|department))\b|call\s+(?:911|999|000|emergency)|"
    r"(?:go|head|rush)\s+to\s+(?:the\s+)?(?:nearest\s+)?hospital|emergency\s+(?:medical\s+)?(?:care|services?))",
    re.IGNORECASE
)
_REF_GP = re.compile(
    r"(?:(?:see|visit|consult|speak\s+(?:to|with))\s+(?:a\s+)?(?:doctor|physician|GP|nurse|healthcare\s+(?:provider|professional))|"
    r"(?:schedule|make|book)\s+an?\s+appointment|(?:primary\s+care|family\s+doctor|general\s+practitioner))",
    re.IGNORECASE
)
def _referral_regex(text):
    if _REF_ER.search(text): return 2
    if _REF_GP.search(text): return 1
    return 0

def _medication_regex(text):
    return _shared_medication_regex(text)

def _empathy_regex(text):
    return _shared_empathy_regex(text)

_DIAG_CERTAIN = re.compile(
    r"(?:\bthis\s+(?:is|indicates|suggests|sounds\s+like)\b|\byou\s+(?:have|are\s+experiencing)\b|"
    r"\bthe\s+(?:most\s+likely\s+)?(?:cause|diagnosis)\s+is\b|\bclassic\s+signs?\s+of\b)", re.IGNORECASE
)
_DIAG_LIKELY  = re.compile(
    r"\b(?:likely|probably|often|frequently|typically|usually|commonly|most\s+likely|suggests?|indicates?)\b",
    re.IGNORECASE
)
_DIAG_VAGUE   = re.compile(
    r"\b(?:might|may|could|possibly|perhaps|potentially|sometimes|it\s+(?:may|is)\s+be|"
    r"hard\s+to\s+say|difficult\s+to\s+determine)\b", re.IGNORECASE
)
def _diagnostic_regex(text):
    # Vague/hedging language overrides apparent certainty: a sentence like
    # "This could indicate a tension headache, though it's hard to say" contains
    # both _DIAG_CERTAIN-like patterns and vague hedges.  Checking vague first
    # ensures hedging language lowers, not raises, the certainty score.
    # After that, explicit diagnostic language outranks merely likely language.
    if _DIAG_VAGUE.search(text):
        return 0
    if _DIAG_CERTAIN.search(text):
        return 2
    if _DIAG_LIKELY.search(text):
        return 1
    return 0  # no diagnostic language at all → lowest certainty

_FOLLOWUP_RX = re.compile(
    r"(?:(?:can|could|would|may|do|does|have|has|is|are)\s+you\s+\w+[^.!?]*\?|"
    r"(?:how\s+long|when\s+did|how\s+severe|what\s+type|which\s+side)\s+[^.!?]*\?)",
    re.IGNORECASE
)
def _triage(urgency: int, severity_str: str) -> int:
    sev_map = {"mild": (0,1), "moderate": (1,2), "severe": (2,3)}
    severity_norm = str(severity_str).strip().lower()
    if severity_norm not in sev_map:
        raise ValueError(
            "triage_appropriateness requires severity to be one of "
            f"{sorted(sev_map)}; got {severity_str!r}"
        )
    lo, hi = sev_map[severity_norm]
    if urgency < lo: return 0
    if urgency > hi: return 2
    return 1


# ── Master extractor ──────────────────────────────────────────────────────────

FEATURE_LABELS = feature_label_map(CLINICAL_FEATURES)

def extract_features(text: str, severity_str: str = "mild", *, use_nlp: bool = False) -> dict:
    if use_nlp:
        u   = score_by_nlp(text, "urgency_score")
        ref = score_by_nlp(text, "referral_score")
        med = score_by_nlp(text, "medication_specificity")
        emp = score_by_nlp(text, "empathy_score")
        dia = score_by_nlp(text, "diagnostic_certainty")
    else:
        u   = _urgency_regex(text)
        ref = _referral_regex(text)
        med = _medication_regex(text)
        emp = _empathy_regex(text)
        dia = _diagnostic_regex(text)

    return {
        "urgency_score":          u,
        "referral_score":         ref,
        "medication_specificity": med,
        "empathy_score":          emp,
        "diagnostic_certainty":   dia,
        "triage_appropriateness": _triage(u, severity_str),
        "response_length":        len(text.split()),
        "warning_signs_count":    sum(1 for rx in _WARNING_RX if rx.search(text)),
        "follow_up_count":        len(_FOLLOWUP_RX.findall(text)),
        "risk_language_score":    len(_RISK_RX.findall(text)),
    }


# =============================================================================
# 2.  DATA LOADING  (Fix 3: age_numeric)
# =============================================================================

def load_data(path: str, *, use_nlp: bool = False) -> pd.DataFrame:
    with open(path) as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            sys.exit(f"[FATAL] Malformed JSON in {path}: {exc}")
    try:
        raw = validate_records(raw, kind="short")
    except SchemaValidationError as exc:
        sys.exit(f"[FATAL] Input schema validation failed for {path}: {exc}")

    if use_nlp:
        print("  Pre-loading sentence-transformer encoder ...")
        _get_encoder()          # warm up once, not per-row
        print("  Extracting NLP features ...")

    rows = []
    n_age_imputed = 0
    ensure_nlp_validation(use_nlp)
    for r in raw:
        response = r.get("response", "") or ""
        if not response.strip() or response.startswith("ERROR") or response == "No response found":
            continue
        meta = r.get("metadata", {})
        sev  = meta.get("severity", "mild")
        feats = extract_features(response, sev, use_nlp=use_nlp)
        raw_age = meta.get("age")
        if raw_age is None:
            n_age_imputed += 1
        row = {
            "model":       r["model"],
            "response":    response,
            "gender":      meta.get("gender",  "unknown"),
            "race":        meta.get("race",    "unknown"),
            "age":         raw_age if raw_age is not None else -1,
            "age_numeric": float(raw_age) if raw_age is not None else 40.0,
            "age_group":   str(raw_age) if raw_age is not None else "40",
            "symptom":     meta.get("symptom", "unknown"),
            "severity":    sev,
        }
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    if n_age_imputed:
        print(f"  [WARNING] {n_age_imputed}/{len(rows)} rows missing 'age' in metadata — "
              f"age_numeric imputed as 40, age_group as '40'. "
              f"Treat age-based results as sensitivity-limited if this occurs often.")
    df["severity"] = pd.Categorical(
        df["severity"], categories=["mild","moderate","severe"], ordered=True
    )
    validate_feature_frame(
        df,
        feature_names=CLINICAL_FEATURES,
        required_columns=[
            "model", "response", "gender", "race", "age", "age_numeric",
            "age_group", "symptom", "severity",
        ],
        context="clinical short-conversation feature matrix",
    )
    return df



# =============================================================================
# 4.  STATISTICS  (Fix 2 continuous-only MixedLM, Fix 4 Dunn, Imp 1 z-score)
# =============================================================================

def validate_complete_prompt_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate matched prompt blocks and return the complete subset."""
    return attach_complete_prompt_blocks(df, CHATBOTS, verbose=True)



def apply_confirmatory_joint_fdr(mixed_df: pd.DataFrame) -> pd.DataFrame:
    """Apply BH-FDR correction jointly across all MixedLM confirmatory p-values."""
    if mixed_df.empty or "p_val" not in mixed_df.columns:
        return mixed_df
    mixed_df = mixed_df.copy()
    valid = mixed_df["p_val"].notna()
    mixed_df["p_fdr_confirmatory"] = np.nan
    if valid.sum() >= 2:
        mixed_df.loc[valid, "p_fdr_confirmatory"] = apply_fdr(
            mixed_df.loc[valid, "p_val"].tolist()
        )
    elif valid.sum() == 1:
        mixed_df.loc[valid, "p_fdr_confirmatory"] = mixed_df.loc[valid, "p_val"]
    mixed_df["sig_fdr_confirmatory"] = mixed_df["p_fdr_confirmatory"] < ALPHA
    mixed_df["fdr_scope"] = FDR_SCOPE_NOTE
    return mixed_df


def kruskal_wallis(df, feature, group_col):
    """INFERENTIAL STATUS: EXPLORATORY.

    Treats responses as independent observations; does not account for the
    matched prompt-condition structure. Use Wilcoxon results for primary inference.
    """
    groups = [g[feature].dropna().values for _, g in df.groupby(group_col) if len(g) > 1]
    if len(groups) < 2: return np.nan, np.nan, np.nan
    H, p = stats.kruskal(*groups)
    n    = df[feature].notna().sum()
    k    = len(groups)
    eps2 = max((H - k + 1) / (n - k), 0.0) if n > k else 0.0
    return H, p, eps2

def one_way_anova(df, feature, group_col):
    """INFERENTIAL STATUS: EXPLORATORY.

    Treats responses as independent observations; does not account for the
    matched prompt-condition structure. Use Wilcoxon results for primary inference.
    """
    groups = [g[feature].dropna().values for _, g in df.groupby(group_col) if len(g) > 1]
    if len(groups) < 2: return np.nan, np.nan, np.nan
    F, p       = stats.f_oneway(*groups)
    all_vals   = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_total   = ((all_vals - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta2 = ss_between / ss_total if ss_total > 0 else 0
    return F, p, eta2

def run_group_test(df, feature, group_col):
    """INFERENTIAL STATUS: EXPLORATORY.

    Treats responses as independent observations; does not account for the
    matched prompt-condition structure. Use Wilcoxon results for primary inference.
    """
    if feature in ORDINAL_FEATURES:
        return kruskal_wallis(df, feature, group_col)
    return one_way_anova(df, feature, group_col)


def dunn_test(df: pd.DataFrame, feature: str, group_col: str) -> pd.DataFrame:
    """
    Fix 4: Dunn's post-hoc test with FDR correction.
    Appropriate for ordinal / non-normal data (unlike Tukey).
    """
    result = sp.posthoc_dunn(
        df, val_col=feature, group_col=group_col, p_adjust="fdr_bh"
    )
    return result


def tukey_hsd_df(df, feature, group_col):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    res = pairwise_tukeyhsd(df[feature], df[group_col])
    return pd.DataFrame(
        data=res._results_table.data[1:],
        columns=res._results_table.data[0]
    )


def mixed_lm(df: pd.DataFrame, feature: str, chatbot: str,
             use_numeric_age: bool = True) -> pd.DataFrame:
    """
    Fix 2: MixedLM applied only to continuous features.
    Ordinal features use this function too but a note is printed.
    Fix 3: uses age_numeric (continuous) instead of age_group (categorical).
    Imp 1: features are z-scored before regression.
    Imp 2: convergence is checked and warned.
    """
    if mixedlm is None:
        print(f"    [SKIP] statsmodels unavailable — cannot fit MixedLM [{chatbot}] {feature}")
        return pd.DataFrame()
    sub = df[df["model"] == chatbot].copy()
    n_groups = int(sub["symptom"].nunique())
    if len(sub) < 15 or n_groups < 2:
        return pd.DataFrame()
    if n_groups < MIN_GROUPS_FOR_MIXED_MODEL:
        print(
            f"    [SKIP] MixedLM [{chatbot}] {feature}: only {n_groups} symptom groups "
            f"(configured minimum {MIN_GROUPS_FOR_MIXED_MODEL} for thesis-grade inference)."
        )
        return pd.DataFrame()

    # Imp 1: z-score the outcome for coefficient comparability
    feat_std = sub[feature].std()
    feat_mean = sub[feature].mean()
    if feat_std > 0:
        sub["y_z"] = (sub[feature] - feat_mean) / feat_std
    else:
        return pd.DataFrame()   # no variance → skip

    # Fix 3: age as numeric predictor
    sub["severity_num"] = sub["severity"].cat.codes

    formula = "y_z ~ C(race) + C(gender) + age_numeric + severity_num"

    try:
        model  = mixedlm(formula, data=sub, groups=sub["symptom"])
        result = model.fit(method="lbfgs", maxiter=300, disp=False)

        # Imp 2: convergence check
        if not result.converged:
            print(f"    ⚠ MixedLM did not converge: [{chatbot}] {feature}")

        coef_df = pd.DataFrame({
            "term":    result.params.index,
            "coef":    result.params.values,
            "std_err": result.bse.values,
            "z":       result.tvalues.values,
            "p_val":   result.pvalues.values,
            "ci_low":  result.conf_int()[0].values,
            "ci_high": result.conf_int()[1].values,
        })
        coef_df["feature"]     = feature
        coef_df["chatbot"]     = chatbot
        coef_df["feat_mean"]   = feat_mean
        coef_df["feat_std"]    = feat_std
        return coef_df
    except Exception as exc:
        print(f"    ✗ {type(exc).__name__} in MixedLM [{chatbot}] {feature}: {exc}")
        return pd.DataFrame()


def run_clmm(df: pd.DataFrame, feature: str, chatbot: str) -> pd.DataFrame:
    """Cumulative Link Mixed Model (CLMM) for ordinal outcomes.

    This is the formally correct model for ordered categorical outcomes such as
    urgency_score (4 levels) and empathy_score (3 levels).  Coefficients are
    log-odds of moving up one ordinal level, directly interpretable.

    Implementation strategy:
      1. rpy2 + R's ordinal::clmm() — direct R call
      2. Graceful fallback: skips this feature, prints a clear message

    Returns a DataFrame with columns: term, coef, p_val, feature, chatbot,
    model_type='clmm'.  Returns empty DataFrame if neither backend is available.
    """
    sub = df[df["model"] == chatbot].copy()
    if len(sub) < 15:
        return pd.DataFrame()
    if sub[feature].nunique() < 2:
        return pd.DataFrame()

    sub["severity_num"] = sub["severity"].cat.codes
    # Encode outcome as ordered categorical string for R compatibility
    sub["y_ord"] = pd.Categorical(sub[feature].astype(int).astype(str),
                                   ordered=True)

    rhs_parts = []
    if sub["race"].nunique()   >= 2: rhs_parts.append("race")
    if sub["gender"].nunique() >= 2: rhs_parts.append("gender")
    rhs_parts.extend(["age_numeric", "severity_num"])

    # ── Attempt: rpy2 + ordinal::clmm ────────────────────────────────
    try:
        import rpy2.robjects as ro                          # type: ignore[import]
        from rpy2.robjects import pandas2ri                 # type: ignore[import]
        from rpy2.robjects.packages import importr          # type: ignore[import]

        pandas2ri.activate()
        ordinal_pkg = importr("ordinal")

        r_df = pandas2ri.py2rpy(sub[["y_ord", "race", "gender",
                                      "age_numeric", "severity_num",
                                      "symptom"]].dropna())
        formula_str = "y_ord ~ " + " + ".join(rhs_parts) + " + (1|symptom)"
        r_formula   = ro.Formula(formula_str)

        clmm_result = ordinal_pkg.clmm(r_formula, data=r_df)
        summary_r   = ro.r["summary"](clmm_result)
        coef_r      = ro.r["coef"](summary_r)
        coef_df_r   = pandas2ri.rpy2py(coef_r)

        records = []
        for term, row in coef_df_r.iterrows():
            records.append({
                "term":       str(term),
                "coef":       float(row.get("Estimate", np.nan)),
                "std_err":    float(row.get("Std. Error", np.nan)),
                "z":          float(row.get("z value", np.nan)),
                "p_val":      float(row.get("Pr(>|z|)", np.nan)),
                "ci_low":     np.nan,
                "ci_high":    np.nan,
                "feature":    feature,
                "chatbot":    chatbot,
                "feat_mean":  float(sub[feature].mean()),
                "feat_std":   float(sub[feature].std()),
                "model_type": "clmm",
            })
        result_df = pd.DataFrame(records)
        print(f"    [CLMM] {chatbot}/{feature}: R ordinal::clmm OK ({len(result_df)} terms)")
        return result_df

    except ImportError:
        print(f"    [CLMM] rpy2/ordinal not available for {chatbot}/{feature}. "
              "Install rpy2 and R package 'ordinal' for correct ordinal mixed models.")
        return pd.DataFrame()
    except Exception as exc:
        print(f"    [CLMM] Failed for {chatbot}/{feature}: {exc}.")
        return pd.DataFrame()


def run_mixed_model(df: pd.DataFrame, feature: str, chatbot: str) -> pd.DataFrame:
    """Dispatch to CLMM (ordinal outcomes) or MixedLM (continuous outcomes).

    For ordinal features, CLMM is the formally correct model. If no ordinal
    backend is available, inference is withheld by default instead of being
    approximated by a linear mixed model.
    The model_type column in the returned DataFrame records which was used.
    """
    if feature in ORDINAL_FEATURES and feature != "triage_appropriateness":
        clmm_result = run_clmm(df, feature, chatbot)
        if not clmm_result.empty:
            return clmm_result
        if not ALLOW_ORDINAL_MIXEDLM_FALLBACK:
            print(
                f"    [SKIP] No ordinal mixed-model backend for {chatbot}/{feature}. "
                "Confirmatory ordinal regression is withheld instead of approximated."
            )
            return pd.DataFrame()
        # Fall through to MixedLM with ordinal caveat already printed by run_clmm
    result = mixed_lm(df, feature, chatbot)
    if not result.empty:
        result = result.copy()
        result["model_type"] = "mixedlm_approx" if feature in ORDINAL_FEATURES else "mixedlm"
    return result


def compute_vif(df: pd.DataFrame, predictors: list) -> pd.Series:
    """Fix 6: Variance Inflation Factor for multicollinearity detection."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[predictors].dropna()
    X = (X - X.mean()) / X.std()   # standardise first
    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(len(predictors))],
        index=predictors
    )
    return vif


# =============================================================================
# 5.  SEMANTIC EMBEDDING ANALYSIS  (Bonus)
# =============================================================================

def _permanova(dist_matrix: np.ndarray, labels: np.ndarray, n_perm: int = 499) -> tuple:
    """
    Permutational MANOVA (Anderson 2001) on a pre-computed distance matrix.
    Test statistic: pseudo-F = (SS_between / df_between) / (SS_within / df_within).
    Returns (pseudo_F, p_value). More sensitive than centroid distances alone
    because it accounts for within-group variance.

    Complexity: O(k * n²) per permutation, O(k * n² * n_perm) total.
    Feasible for thesis-scale datasets (n < ~500); avoid on n > ~2000 without
    further vectorisation.
    """
    n = len(labels)
    _, group_idx = np.unique(labels, return_inverse=True)
    k = len(np.unique(labels))
    sq = dist_matrix ** 2

    def pseudo_f(idx):
        ss_within = sum(
            sq[np.ix_(idx == g, idx == g)].sum() / (2 * (idx == g).sum())
            for g in range(k) if (idx == g).sum() >= 2
        )
        ss_total  = sq.sum() / (2 * n)
        df_b = k - 1
        df_w = n - k
        if df_w <= 0 or ss_within == 0:
            return 0.0
        return ((ss_total - ss_within) / df_b) / (ss_within / df_w)

    obs_f = pseudo_f(group_idx)
    count = sum(
        pseudo_f(np.random.permutation(group_idx)) >= obs_f
        for _ in range(n_perm)
    )
    return round(obs_f, 4), round((count + 1) / (n_perm + 1), 4)


def semantic_bias_analysis(df: pd.DataFrame):
    """Measures stylistic drift and topical continuity in embedding space.

    Does NOT measure conversational quality, safety, or coherence.
    Larger cosine distances = greater divergence in average semantic content
    between demographic groups. Treat as exploratory and descriptive only.

    Computes per-demographic centroid distances and PERMANOVA tests within each
    chatbot to detect latent content differences not captured by scalar features.
    Note: MANOVA is not implemented — PERMANOVA (Anderson 2001) is used instead,
    which is non-parametric and does not assume multivariate normality.
    """
    print("\n[EXPLORATORY] Semantic embedding similarity analysis ...")
    from sentence_transformers import SentenceTransformer, util
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_distances

    enc  = _get_encoder()
    embs = enc.encode(df["response"].tolist(), batch_size=64,
                      show_progress_bar=True, convert_to_numpy=True)
    df2  = df.copy()
    df2["emb"] = list(embs)

    results = []
    demo_vars = {"race": sorted(df["race"].unique()),
                 "gender": sorted(df["gender"].unique()),
                 "age_group": sorted(df["age_group"].unique())}

    for chatbot in CHATBOTS:
        sub  = df2[df2["model"] == chatbot]
        emb_sub = np.vstack(sub["emb"].values)

        for demo, groups in demo_vars.items():
            centroids = {}
            for g in groups:
                mask = sub[demo].values == g
                if mask.sum() == 0: continue
                centroids[g] = emb_sub[mask].mean(axis=0)

            # Pairwise cosine distances between centroids
            group_names = list(centroids.keys())
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    g1, g2 = group_names[i], group_names[j]
                    sim = float(np.dot(centroids[g1], centroids[g2]) /
                                (np.linalg.norm(centroids[g1]) * np.linalg.norm(centroids[g2])))
                    results.append({
                        "chatbot":  chatbot,
                        "demo":     demo,
                        "group_a":  g1,
                        "group_b":  g2,
                        "cosine_sim": round(sim, 4),
                        "cosine_dist": round(1 - sim, 4),
                    })

    res_df = pd.DataFrame(results)
    print("\n  Average semantic embedding_distance between demographic groups (cosine):")
    pivot = res_df.groupby(["chatbot", "demo"])["cosine_dist"].mean().unstack("demo")
    print(pivot.round(4).to_string())

    # Save
    _sem_path = str(_SCRIPT_DIR / "semantic_distances.csv")
    res_df.to_csv(_sem_path, index=False)
    print(f"  Saved: {_sem_path}")

    # [EXPLORATORY] Semantic Distance Correlates — characterise what the metric captures
    print("\n  [EXPLORATORY] Semantic Distance Correlates (Spearman ρ):")
    print("  Helps characterise empirically what embedding distance measures.")
    from sklearn.metrics.pairwise import cosine_distances as _cd
    dist_correlate_records = []
    for chatbot in CHATBOTS:
        sub  = df2[df2["model"] == chatbot]
        if len(sub) < 5:
            continue
        emb_sub = np.vstack(sub["emb"].values)
        centroid = emb_sub.mean(axis=0)
        dists_from_centroid = np.array([
            float(_cd(e.reshape(1, -1), centroid.reshape(1, -1))[0, 0])
            for e in emb_sub
        ])
        for correlate in ["response_length", "urgency_score", "empathy_score"]:
            if correlate not in sub.columns:
                continue
            vals = sub[correlate].values
            mask = ~np.isnan(vals) & ~np.isnan(dists_from_centroid)
            if mask.sum() < 5:
                continue
            rho, pval = stats.spearmanr(dists_from_centroid[mask], vals[mask])
            dist_correlate_records.append({
                "chatbot": chatbot, "correlate": correlate,
                "spearman_rho": round(rho, 3), "p_value": round(pval, 4),
            })

    if dist_correlate_records:
        print(f"  {'Chatbot':<14}{'Correlate':<26}{'ρ':>7}{'p':>9}")
        print("  " + "-" * 56)
        for r in dist_correlate_records:
            print(f"  {r['chatbot']:<14}{r['correlate']:<26}{r['spearman_rho']:>7}{r['p_value']:>9}")

    # PERMANOVA: tests whether demographic group explains variance in embedding space,
    # accounting for within-group spread (unlike centroid distances alone).
    # p-values are FDR-corrected across all chatbot × demographic combinations.
    print("\n  PERMANOVA (pseudo-F on cosine distances, 499 permutations; FDR-corrected):")
    print("  ★ = p_FDR < 0.05  |  tests whether group membership predicts embedding position")
    perm_rows = []
    for chatbot in CHATBOTS:
        sub      = df2[df2["model"] == chatbot]
        if len(sub) < 6:
            continue
        emb_sub  = np.vstack(sub["emb"].values)
        dist_mat = cosine_distances(emb_sub)
        for demo in demo_vars:
            labels = sub[demo].values
            if len(np.unique(labels)) < 2:
                continue
            f_stat, p_val = _permanova(dist_mat, labels, n_perm=499)
            perm_rows.append({"chatbot": chatbot, "demo": demo,
                               "F": f_stat, "p_raw": p_val})

    if perm_rows:
        perm_df = pd.DataFrame(perm_rows)
        _, p_fdr, _, _ = multipletests(perm_df["p_raw"], method="fdr_bh")
        perm_df["p_fdr"] = p_fdr.round(4)
        for _, row in perm_df.iterrows():
            sig = " ★" if row["p_fdr"] < ALPHA else ""
            print(f"    [{row['chatbot']:<12}] {row['demo']:<12}  F={row['F']:.3f}  "
                  f"p_raw={row['p_raw']:.4f}  p_FDR={row['p_fdr']:.4f}{sig}")

    # PCA plot per chatbot, coloured by race
    for chatbot in CHATBOTS:
        sub = df2[df2["model"] == chatbot]
        if len(sub) < 5: continue
        emb_sub = np.vstack(sub["emb"].values)
        coords  = PCA(n_components=2, random_state=42).fit_transform(emb_sub)
        races   = sorted(sub["race"].unique())
        palette = sns.color_palette("tab10", len(races))
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, race in enumerate(races):
            mask = sub["race"].values == race
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=race, color=palette[i], alpha=0.55, s=20)
        ax.set_title(f"{chatbot.replace('_',' ').title()} — Response Embeddings by Race")
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"pca_race_{chatbot}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Saved: {path}")

    return res_df


# =============================================================================
# 6.  VISUALISATIONS  (minor improvements over v2)
# =============================================================================

def plot_feature_heatmap(raw_stats: dict, fdr_sig: dict, features: list):
    demo_vars  = ["model", "race", "gender", "age_group"]
    eta_mat    = pd.DataFrame(index=features, columns=demo_vars, dtype=float)
    sig_mat    = pd.DataFrame(index=features, columns=demo_vars, dtype=bool)
    label_mat  = pd.DataFrame(index=features, columns=demo_vars, dtype=str)

    for feat in features:
        for var in demo_vars:
            _, _, effect = raw_stats.get((feat, var), (np.nan, np.nan, 0.0))
            eta_mat.loc[feat, var]   = round(effect, 3) if not np.isnan(effect) else 0.0
            sig_mat.loc[feat, var]   = fdr_sig.get((feat, var), False)
            # Use type-correct threshold: ε² for KW (ordinal), η² for ANOVA
            _lbl_fn = effect_label_eps2 if feat in ORDINAL_FEATURES else effect_label_eta2
            label_mat.loc[feat, var] = _lbl_fn(eta_mat.loc[feat, var])

    row_labels = [FEATURE_LABELS[f] for f in features]
    col_labels = ["Chatbot", "Race", "Gender", "Age Group"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(eta_mat.values.astype(float), aspect="auto",
                   cmap="YlOrRd", vmin=0, vmax=0.5)
    plt.colorbar(im, ax=ax, label="Effect size (η² or ε²)")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=10)

    for i, feat in enumerate(features):
        for j, var in enumerate(demo_vars):
            val   = eta_mat.loc[feat, var]
            star  = "★" if sig_mat.loc[feat, var] else ""
            tag   = label_mat.loc[feat, var]
            color = "black" if val < 0.25 else "white"
            ax.text(j, i, f"{val:.2f}{star}\n({tag})",
                    ha="center", va="center", fontsize=7.5, color=color)

    ax.set_title("Effect Sizes — Clinical Features × Demographics\n"
                 "★ = FDR-corrected p < 0.05  |  Labels: trivial/small/medium/large",
                 fontsize=11, pad=12)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "anova_heatmap.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_correlation(df: pd.DataFrame, features: list):
    corr   = df[features].corr(method="spearman")
    labels = [FEATURE_LABELS[f].split("(")[0].strip() for f in features]
    mask   = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0,
                cmap="coolwarm", ax=ax, square=True, linewidths=0.5,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Feature Correlation Matrix (Spearman)\n"
                 "ρ > 0.6 flagged as potentially redundant", fontsize=12)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_boxplots_by_demo(df, features, demo, order, label):
    for feat in features:
        n_bots = len(CHATBOTS)
        fig, axes = plt.subplots(1, n_bots, figsize=(5 * n_bots, 4), sharey=True, squeeze=False)
        axes = axes.flatten()
        for i, (ax, chatbot) in enumerate(zip(axes, CHATBOTS)):
            sub = df[df["model"] == chatbot].copy()
            if str(sub[demo].dtype) == "category":
                sub[demo] = sub[demo].astype(str)
            sns.boxplot(data=sub, x=demo, y=feat, order=order,
                        palette="Set2", width=0.55, ax=ax,
                        flierprops={"marker": "."})
            ax.set_title(chatbot.replace("_", " ").title(), fontsize=11)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
            if i > 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(FEATURE_LABELS[feat], fontsize=10)
        fig.suptitle(f"{FEATURE_LABELS[feat]} by {label}", fontsize=13, y=1.02)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"{demo}_{feat}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {demo} box plots → {PLOT_DIR}/{demo}_*.png")


def plot_regression_forest(all_coefs: pd.DataFrame, feature: str):
    demo_terms = all_coefs[
        all_coefs["term"].str.contains(r"race|gender|age_numeric", case=False, na=False) &
        ~all_coefs["term"].str.contains("Intercept|Group Var", case=False, na=False)
    ].copy()
    sub = demo_terms[demo_terms["feature"] == feature]
    if sub.empty: return

    pvals = sub["p_val"].values
    _, pcorr, _, _ = multipletests(pvals, method="fdr_bh")
    sub = sub.copy()
    sub["p_fdr"] = pcorr
    sub["sig"]   = sub["p_fdr"] < ALPHA

    n_rows = max(len(sub[sub["chatbot"] == c]) for c in CHATBOTS)
    n_bots = len(CHATBOTS)
    fig, axes = plt.subplots(1, n_bots, figsize=(5.3 * n_bots, max(4, n_rows + 2)), sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, chatbot in zip(axes, CHATBOTS):
        cdf = sub[sub["chatbot"] == chatbot].sort_values("coef")
        if cdf.empty: ax.set_visible(False); continue

        colors = ["#C0392B" if s else "#7F8C8D" for s in cdf["sig"]]
        errors = np.abs(cdf["coef"] - cdf["ci_low"])
        ax.barh(range(len(cdf)), cdf["coef"], xerr=errors,
                color=colors, alpha=0.85, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(cdf)))

        def clean(t):
            return (t.replace("C(race)[T.","Race:")
                     .replace("C(gender)[T.","Gender:")
                     .replace("age_numeric","Age (continuous)")
                     .replace("]",""))

        ax.set_yticklabels([clean(t) for t in cdf["term"]], fontsize=8)
        for i_, (_, row) in enumerate(cdf.iterrows()):
            if row["sig"]:
                ax.text(row["coef"] + errors.iloc[i_] * 0.05, i_,
                        f"p={row['p_fdr']:.3f}", va="center",
                        fontsize=7, color="#C0392B")

        ax.set_title(chatbot.replace("_"," ").title(), fontsize=11)
        ax.set_xlabel("Standardised coefficient (z-scored outcome)")

    sig_patch   = mpatches.Patch(color="#C0392B", label="FDR p < 0.05")
    insig_patch = mpatches.Patch(color="#7F8C8D", label="FDR p ≥ 0.05")
    fig.legend(handles=[sig_patch, insig_patch], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"MixedLM Coefficients (z-scored) — {FEATURE_LABELS[feature]}\n"
        f"Age as continuous; random intercepts by symptom; FDR-corrected",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"regression_{feature}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_radar(mean_by_race, chatbot, features):
    races  = sorted(mean_by_race.index)
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    palette = sns.color_palette("tab10", len(races))
    feat_min = mean_by_race[features].min()
    feat_max = mean_by_race[features].max()
    span     = (feat_max - feat_min).replace(0, 1)
    for i, race in enumerate(races):
        vals = ((mean_by_race.loc[race, features] - feat_min) / span).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=palette[i], linewidth=1.8, label=race)
        ax.fill(angles, vals, color=palette[i], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FEATURE_LABELS[f].split("(")[0].strip() for f in features], fontsize=7)
    ax.set_yticklabels([])
    ax.set_title(chatbot.replace("_"," ").title() + "\nClinical Profile by Race",
                 fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"radar_{chatbot}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 7.  VALIDATION INFRASTRUCTURE  (Priority 4)
# =============================================================================

def generate_validation_sample(df: pd.DataFrame, n: int = 25,
                                random_state: int = 42) -> None:
    """Export a stratified validation sample for human inter-rater reliability.

    Samples ~8-9 responses per chatbot, stratified by symptom, and exports:
    - bias_analysis/validation/validation_sample.csv — for human annotation
    - bias_analysis/validation/validation_rubric.txt — scoring criteria
    """
    _val_dir = str(_SCRIPT_DIR / "validation")
    os.makedirs(_val_dir, exist_ok=True)

    samples = []
    per_chatbot = n // len(CHATBOTS)

    for chatbot in CHATBOTS:
        sub = df[df["model"] == chatbot].copy()
        symptoms = sub["symptom"].unique()
        n_sym = len(symptoms)
        per_sym = max(1, per_chatbot // n_sym)
        for sym in symptoms:
            sym_sub = sub[sub["symptom"] == sym]
            k = min(per_sym, len(sym_sub))
            if k > 0:
                chosen = sym_sub.sample(k, random_state=random_state)
                samples.append(chosen)

    sample_df = pd.concat(samples, ignore_index=True).head(n)
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sample_df.insert(0, "response_id", range(1, len(sample_df) + 1))

    ordinal_feats = ["urgency_score", "empathy_score", "medication_specificity",
                     "diagnostic_certainty", "referral_score", "triage_appropriateness"]

    export_cols = (["response_id", "model", "symptom", "severity",
                    "race", "gender", "age", "response"]
                   + [f"{f}_heuristic" for f in ordinal_feats if f in sample_df.columns])
    for f in ordinal_feats:
        if f in sample_df.columns:
            sample_df[f"{f}_heuristic"] = sample_df[f]

    for f in ["urgency", "empathy", "medication_specificity"]:
        sample_df[f"{f}_rater1"] = ""
        sample_df[f"{f}_rater2"] = ""
        export_cols += [f"{f}_rater1", f"{f}_rater2"]

    sample_df["agreement_notes"] = ""
    export_cols.append("agreement_notes")

    out_csv = str(_SCRIPT_DIR / "validation" / "validation_sample.csv")
    sample_df[[c for c in export_cols if c in sample_df.columns]].to_csv(out_csv, index=False)

    rubric_lines = [
        "VALIDATION RUBRIC — Clinical Feature Scoring Criteria",
        "=" * 60,
        "",
        "Rate each response on the following scales.",
        "Leave cell blank if you cannot determine a score.",
        "",
    ]
    rubric_lines += [
        "URGENCY (0–3)",
        "  0 = No recommendation to seek care; manage at home",
        "  1 = Recommends seeing a GP / doctor if symptoms persist",
        "  2 = Recommends prompt/urgent evaluation today",
        "  3 = Emergency — calls 911/999/ER immediately",
        "",
        "EMPATHY (0–2)",
        "  0 = No acknowledgement of patient's concern or wellbeing",
        "  1 = Minimal acknowledgement (e.g. 'of course', 'certainly')",
        "  2 = Explicit emotional acknowledgement (e.g. 'I'm sorry to hear...',",
        "      'I understand how worrying this must be')",
        "",
        "MEDICATION_SPECIFICITY (0–2)",
        "  0 = No medication mentioned",
        "  1 = Vague reference only (e.g. 'pain reliever', 'OTC medication')",
        "  2 = Named drug (e.g. ibuprofen, paracetamol, acetaminophen)",
        "",
        "Cohen's κ target: ≥ 0.6 for each feature.",
        "Features with κ < 0.6 will be flagged as LOW AGREEMENT.",
    ]

    rubric_path = str(_SCRIPT_DIR / "validation" / "validation_rubric.txt")
    with open(rubric_path, "w") as f:
        f.write("\n".join(rubric_lines))

    print(f"\n  Validation sample → {out_csv}")
    print(f"  Rubric           → {rubric_path}")
    print("  Have 2 independent raters complete validation_sample.csv, then run")
    print(f"  compute_validation_agreement('{out_csv}')")


def compute_validation_agreement(annotated_csv_path: str) -> None:
    """Load completed annotation CSV and compute inter-rater reliability.

    For each ordinal feature: weighted Cohen's kappa (rater1 vs rater2) and
    Spearman correlation of each rater against the heuristic score.
    Saves summary to bias_analysis/validation/validation_results.csv.
    """
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import spearmanr

    try:
        ann = pd.read_csv(annotated_csv_path)
    except FileNotFoundError:
        print(f"  [ERROR] File not found: {annotated_csv_path}")
        return

    feat_pairs = [
        ("urgency",               "urgency_score_heuristic"),
        ("empathy",               "empathy_score_heuristic"),
        ("medication_specificity","medication_specificity_heuristic"),
    ]

    records = []
    for feat, heuristic_col in feat_pairs:
        r1_col = f"{feat}_rater1"
        r2_col = f"{feat}_rater2"
        if r1_col not in ann.columns or r2_col not in ann.columns:
            continue

        sub = ann[[r1_col, r2_col]].dropna()
        if len(sub) < 5:
            records.append({"feature": feat, "kappa": np.nan, "n": len(sub),
                             "spearman_r1_heuristic": np.nan,
                             "spearman_r2_heuristic": np.nan, "verdict": "insufficient data"})
            continue

        r1 = sub[r1_col].astype(int)
        r2 = sub[r2_col].astype(int)

        try:
            kappa = cohen_kappa_score(r1, r2, weights="linear")
        except Exception:
            kappa = np.nan

        sp_r1 = sp_r2 = np.nan
        if heuristic_col in ann.columns:
            h_vals = ann.loc[sub.index, heuristic_col].dropna()
            common = sub.index.intersection(h_vals.index)
            if len(common) >= 5:
                sp_r1, _ = spearmanr(r1.loc[common], h_vals.loc[common])
                sp_r2, _ = spearmanr(r2.loc[common], h_vals.loc[common])

        verdict = "OK" if (not np.isnan(kappa) and kappa >= 0.6) else \
                  "⚠ LOW AGREEMENT — interpret with caution in thesis"
        records.append({"feature": feat, "kappa": round(kappa, 3) if not np.isnan(kappa) else np.nan,
                         "n": len(sub),
                         "spearman_r1_heuristic": round(sp_r1, 3) if not np.isnan(sp_r1) else np.nan,
                         "spearman_r2_heuristic": round(sp_r2, 3) if not np.isnan(sp_r2) else np.nan,
                         "verdict": verdict})

    result_df = pd.DataFrame(records)
    out_path = str(_SCRIPT_DIR / "validation" / "validation_results.csv")
    result_df.to_csv(out_path, index=False)

    print("\n  Validation Agreement Summary:")
    print(f"  {'Feature':<28}{'κ':>7}{'n':>5}  {'ρ(r1,heur)':>12}  {'ρ(r2,heur)':>12}  Verdict")
    print("  " + "-" * 80)
    for _, row in result_df.iterrows():
        kappa_s = f"{row['kappa']:.3f}" if not np.isnan(row["kappa"]) else "  —"
        print(f"  {row['feature']:<28}{kappa_s:>7}{int(row['n']):>5}  "
              f"{row['spearman_r1_heuristic']:>12}  {row['spearman_r2_heuristic']:>12}  {row['verdict']}")
    print(f"\n  Results saved → {out_path}")


# =============================================================================
# 8.  MODEL DIAGNOSTICS  (Priority 5)
# =============================================================================

def run_model_diagnostics(df: pd.DataFrame) -> None:
    """Fit MixedLM for representative chatbot+feature combinations and report diagnostics.

    Plots residuals-vs-fitted and Q-Q plots. Computes ICC and Shapiro-Wilk on residuals.
    Saves plots to bias_analysis/plots/diagnostics/.
    """
    import scipy.stats as _scipy_stats

    diag_dir = str(_SCRIPT_DIR / "plots" / "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    combos = [
        ("doctronic",   "urgency_score"),
        ("drkhan",      "urgency_score"),
    ]

    print("\n" + "═" * 70)
    print("═══ DIAGNOSTICS ═══")
    print("  MixedLM residual diagnostics for urgency_score per chatbot")
    print("═" * 70)

    diag_records = []
    for chatbot, feat in combos:
        sub = df[df["model"] == chatbot].copy()
        if len(sub) < 15:
            print(f"  [SKIP] {chatbot}/{feat}: insufficient data")
            continue

        feat_std  = sub[feat].std()
        feat_mean = sub[feat].mean()
        if feat_std == 0:
            continue

        sub["y_z"]         = (sub[feat] - feat_mean) / feat_std
        sub["severity_num"] = sub["severity"].cat.codes

        formula = "y_z ~ C(race) + C(gender) + age_numeric + severity_num"
        try:
            model  = mixedlm(formula, data=sub, groups=sub["symptom"])
            result = model.fit(method="lbfgs", maxiter=300, disp=False)
        except Exception as exc:
            print(f"  [ERROR] {chatbot}/{feat}: {exc}")
            continue

        fitted    = result.fittedvalues
        residuals = result.resid

        # ICC = var(random effect) / (var(random effect) + var(residual))
        try:
            re_var  = float(result.cov_re.values[0][0])
            res_var = float(result.scale)
            icc     = re_var / (re_var + res_var) if (re_var + res_var) > 0 else np.nan
        except Exception:
            icc = np.nan

        # Shapiro-Wilk
        try:
            sw_stat, sw_p = _scipy_stats.shapiro(residuals[:min(len(residuals), 5000)])
        except Exception:
            sw_stat, sw_p = np.nan, np.nan

        verdict = "OK"
        if not np.isnan(sw_p) and sw_p < 0.05:
            verdict += " | non-normal residuals"
        if not np.isnan(icc) and icc > 0.3:
            verdict += f" | high ICC ({icc:.2f}) — random effects important"

        diag_records.append({
            "chatbot":    chatbot, "feature": feat,
            "converged":  bool(result.converged),
            "ICC":        round(icc, 3) if not np.isnan(icc) else np.nan,
            "shapiro_p":  round(sw_p, 4) if not np.isnan(sw_p) else np.nan,
            "verdict":    verdict,
        })

        # Plot residuals vs fitted + Q-Q
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.scatter(fitted, residuals, alpha=0.4, s=15)
        ax1.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title(f"{chatbot} / {feat}\nResiduals vs Fitted")

        _scipy_stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title("Normal Q-Q Plot of Residuals")

        plt.tight_layout()
        path = os.path.join(diag_dir, f"diagnostics_{chatbot}_{feat}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    if diag_records:
        diag_df = pd.DataFrame(diag_records)
        print(f"\n  {'Chatbot':<14}{'Feature':<22}{'Converged':>10}{'ICC':>7}{'Shapiro-p':>11}  Verdict")
        print("  " + "-" * 75)
        for _, row in diag_df.iterrows():
            conv_s = str(row["converged"])
            icc_s  = f"{row['ICC']:.3f}" if not np.isnan(row["ICC"]) else " —"
            sw_s   = f"{row['shapiro_p']:.4f}" if not np.isnan(row["shapiro_p"]) else "  —"
            print(f"  {row['chatbot']:<14}{row['feature']:<22}{conv_s:>10}{icc_s:>7}{sw_s:>11}  {row['verdict']}")


# =============================================================================
# 9.  MAIN
# =============================================================================

def main():
    # ── CLI (parsed here, not at import time) ─────────────────────────────
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--use-nlp", action="store_true",
        help="Use sentence-transformer anchor scoring for supported ordinal features.",
    )
    args, _ = parser.parse_known_args()
    rng = np.random.default_rng(args.seed)
    # Seed the legacy numpy global state once so _permanova permutations and
    # sklearn PCA are reproducible. All new code should use rng directly.
    np.random.seed(int(rng.integers(2**31)))

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(str(_SCRIPT_DIR / "validation"), exist_ok=True)

    print("=" * 70)
    print("Clinical Feature Bias Analysis v3")
    print("Fixes: Dunn test · age numeric · z-score · convergence check · VIF")
    print("=" * 70)

    print_hypothesis_table()

    print(f"\n  NLP anchor scoring: {'ON' if args.use_nlp else 'OFF'}")
    ensure_nlp_validation(args.use_nlp)

    # ── Load ──────────────────────────────────────────────────────────
    print(f"\n[1] Loading {DATA_PATH} ...")
    df = load_data(DATA_PATH, use_nlp=args.use_nlp)
    df = df[df["model"].isin(CHATBOTS)].copy()
    print(f"  {len(df)} records | {dict(df['model'].value_counts())}")
    print(f"  Races: {sorted(df['race'].unique())}")

    features = list(CLINICAL_FEATURES)   # includes derived triage for descriptive summaries

    # ── Validate prompt blocks ────────────────────────────────────────
    df_complete = validate_complete_prompt_blocks(df)

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ DESCRIPTIVE ANALYSES ═══")
    print("═" * 70)

    # ── Feature summary ───────────────────────────────────────────────
    print("\n[2] Clinical feature summary (means ± SD by chatbot):")
    summary = df.groupby("model")[features].agg(["mean","std"]).round(3)
    print(summary.to_string())

    # ── Feature correlation ───────────────────────────────────────────
    print("\n[3] Spearman feature correlations (ρ > 0.60 = redundancy warning):")
    corr = df[features].corr(method="spearman").round(3)
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            rho = corr.iloc[i,j]
            if abs(rho) > 0.60:
                print(f"  ⚠  {features[i]}  ↔  {features[j]}  ρ={rho:.2f}")

    # Fix 6: VIF for continuous predictors
    print("\n  VIF for continuous features (multicollinearity; VIF > 5 = concern):")
    cont  = list(CONTINUOUS_FEATURES)
    vif   = compute_vif(df, cont)
    for feat, v in vif.items():
        flag = " ← HIGH" if v > 5 else ""
        print(f"    {feat:<30} VIF = {v:.2f}{flag}")

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ CONFIRMATORY ANALYSES ═══")
    print("═" * 70)

    # ── Multiple comparison strategy ─────────────────────────────────
    print("\n  ── MULTIPLE COMPARISON STRATEGY ──")
    print("  FDR correction is applied within each analysis stage independently:")
    print("  globally across group-level tests and MixedLM coefficient sets per")
    print("  chatbot. Post-hoc Dunn's tests apply FDR")
    print("  within each chatbot×feature cell. These correction families are NOT")
    print("  jointly controlled, meaning the paper-level false positive rate exceeds")
    print("  α = 0.05. To manage this, all demographic findings are reported with")
    print("  effect sizes alongside p-values; only medium or larger effects")
    print("  (ε²/η²/W ≥ 0.06) are discussed in the findings, regardless of")
    print("  statistical significance. Trivial effects that survive FDR correction")
    print("  within a local family are not interpreted as substantive findings.")

    # ── Mixed models for H4 (race/gender/age predict feature scores) ──
    print(f"\n  Confirmatory features ({len(INFERENTIAL_FEATURES)}): triage_appropriateness")
    print("  excluded (derived from urgency_score — would introduce circular inference).")
    print("\n[6] Mixed-effects regression for H4 (age_numeric continuous; FDR):")
    print("  Ordinal outcomes (urgency, empathy, etc.): CLMM via rpy2/ordinal if")
    print("  available; otherwise withheld rather than approximated by linear MixedLM.")
    print("  Continuous outcomes: MixedLM (z-scored).")
    print(f"  Random-effects models require at least {MIN_GROUPS_FOR_MIXED_MODEL} symptom groups.")
    print("  triage_appropriateness excluded (derived feature — circular inference).")

    all_coefs = []
    for chatbot in CHATBOTS:
        print(f"\n  ── {chatbot.upper()} ──")
        for feat in INFERENTIAL_FEATURES:
            coef_df = run_mixed_model(df, feat, chatbot)
            if coef_df.empty: continue
            all_coefs.append(coef_df)

    all_coefs_df = pd.concat(all_coefs, ignore_index=True) if all_coefs else pd.DataFrame()
    all_coefs_df = apply_confirmatory_joint_fdr(all_coefs_df)

    if not all_coefs_df.empty:
        print("\n  Joint confirmatory FDR across MixedLM coefficient families:")
        demo_sig = all_coefs_df[
            all_coefs_df["term"].str.contains(r"race|gender|age_numeric", case=False, na=False) &
            (all_coefs_df["sig_fdr_confirmatory"])
        ]
        if demo_sig.empty:
            print("    No demographic mixed-model coefficients survive joint confirmatory FDR.")
        else:
            for (chatbot, feat), sub in demo_sig.groupby(["chatbot", "feature"], sort=False):
                model_type = sub.get("model_type", pd.Series(["mixedlm"])).iloc[0] \
                             if "model_type" in sub.columns else "mixedlm"
                if model_type == "clmm":
                    model_note = " [CLMM — log-odds of moving up one ordinal level]"
                elif model_type == "mixedlm_approx" and feat in ORDINAL_FEATURES:
                    model_note = " [MixedLM approx — ordinal outcome; install rpy2+ordinal for CLMM]"
                else:
                    model_note = " [z-scored]"
                print(f"    {chatbot}/{feat}{model_note}:")
                for _, row in sub.iterrows():
                    tc = (row["term"]
                          .replace("C(race)[T.","Race:")
                          .replace("C(gender)[T.","Gender:")
                          .replace("age_numeric","Age(continuous)")
                          .replace("]",""))
                    print(f"      {tc:<30}  β={row['coef']:+.3f}  "
                          f"p_raw={row['p_val']:.4f}  p_FDR={row['p_fdr_confirmatory']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ EXPLORATORY ANALYSES ═══")
    print("[EXPLORATORY — see Wilcoxon / MixedLM results for primary inference]")
    print("═" * 70)

    # ── Group tests + FDR ────────────────────────────────────────────
    print("\n[4] [EXPLORATORY] Group tests (KW for ordinal / ANOVA for continuous) + FDR:")
    print("  NOTE: treats responses as independent — does not account for matched structure.")
    demo_vars  = ["model", "race", "gender", "age_group"]
    raw_stats  = {}
    all_keys   = []
    all_raw_ps = []

    for feat in features:
        for var in demo_vars:
            stat, p, effect = run_group_test(df, feat, var)
            raw_stats[(feat, var)] = (stat, p, effect)
            all_keys.append((feat, var))
            all_raw_ps.append(p)

    corrected_ps = apply_fdr(all_raw_ps)
    fdr_sig      = {}
    corr_p_dict  = {}
    for i, (feat, var) in enumerate(all_keys):
        cp = corrected_ps[i]
        fdr_sig[(feat, var)]   = (not np.isnan(cp)) and (cp < ALPHA)
        corr_p_dict[(feat, var)] = cp

    # Use type-correct effect-size labels: ε² for KW (ordinal), η² for ANOVA
    def _size_label(feat, effect):
        return effect_label_eps2(effect) if feat in ORDINAL_FEATURES else effect_label_eta2(effect)

    test_label = {f: ("KW/ε²" if f in ORDINAL_FEATURES else "F/η²") for f in features}
    header = f"  {'Feature':<28}{'Variable':<12}{'Stat':>9}{'p_raw':>9}{'p_FDR':>9}{'Effect':>8}{'Size':>9}{'sig':>5}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for feat in features:
        for var in demo_vars:
            stat, raw_p, effect = raw_stats[(feat, var)]
            cp   = corr_p_dict[(feat, var)]
            sig  = "★" if fdr_sig[(feat, var)] else ""
            tl   = test_label[feat]
            size = _size_label(feat, effect)
            print(f"  {feat:<28}{var:<12}{stat:>9.2f}{raw_p:>9.4f}{cp:>9.4f}"
                  f"{effect:>8.3f}{size:>9}{sig:>5}  [{tl}] [EXPLORATORY]")

    # Fix 4: Dunn's test post-hoc (race, FDR-significant ordinal features)
    print("\n[5] [EXPLORATORY] Post-hoc tests (race, FDR-significant features):")
    for feat in features:
        if not fdr_sig.get((feat, "race")):
            continue
        _, _, effect = raw_stats[(feat, "race")]
        label = _size_label(feat, effect)
        print(f"\n  >>> {feat}  (ε²/η²={effect:.3f}, {label}) [FDR-sig] [EXPLORATORY]")
        for chatbot in CHATBOTS:
            sub = df[df["model"] == chatbot]
            if sub["race"].nunique() < 2: continue
            if feat in ORDINAL_FEATURES:
                dunn = dunn_test(sub, feat, "race")
                print(f"    [{chatbot}] Dunn's test (FDR) [EXPLORATORY]:")
                for g1 in dunn.index:
                    for g2 in dunn.columns:
                        if g1 < g2 and dunn.loc[g1, g2] < ALPHA:
                            print(f"      {g1} vs {g2}: p={dunn.loc[g1,g2]:.4f}")
            else:
                try:
                    tdf = tukey_hsd_df(sub, feat, "race")
                    for _, row in tdf[tdf["reject"]].iterrows():
                        print(f"    [{chatbot}] Tukey [EXPLORATORY]: {row['group1']} vs {row['group2']}: "
                              f"meandiff={row['meandiff']:.3f}, p={row['p-adj']:.4f}")
                except Exception as exc:
                    print(f"    [{chatbot}] {type(exc).__name__} in Tukey: {exc}")

    print("\n  NOTE: Dunn's FDR correction is applied within each chatbot×feature pair,")
    print("        independently of the global FDR in section [4]. The family-wise error")
    print("        rate across both stages is not jointly controlled — treat post-hoc")
    print("        p-values conservatively.")
    print(f"\n  FDR scope: {FDR_SCOPE_NOTE}")

    # ── Post-hoc power analysis ───────────────────────────────────────
    print("\n[5b] Post-hoc achieved power (α=0.05, observed effect sizes):")
    print("  Power < 0.5 flagged — frame affected results as pilot estimates.")
    try:
        from statsmodels.stats.power import FTestAnovaPower
        _pwr = FTestAnovaPower()
        print(f"\n  {'Feature':<28} {'Group':<12} {'N':>5} {'k':>4} {'effect':>8} {'Power':>7}")
        print("  " + "-" * 70)
        for feat in INFERENTIAL_FEATURES:
            for var in demo_vars:
                groups = [g[feat].dropna().values
                          for _, g in df.groupby(var) if len(g) > 1]
                if len(groups) < 2:
                    continue
                k        = len(groups)
                all_v    = np.concatenate(groups)
                n        = len(all_v)
                gm       = all_v.mean()
                ss_t     = ((all_v - gm) ** 2).sum()
                ss_b     = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
                eff      = ss_b / ss_t if ss_t > 0 else 0.0
                try:
                    f_stat = np.sqrt(eff / (1 - eff)) if 0 < eff < 1 else 0.0
                    power  = _pwr.solve_power(effect_size=f_stat, nobs=n,
                                              alpha=0.05, k_groups=k)
                except Exception:
                    power = np.nan
                flag = " ← LOW" if (not np.isnan(power)) and power < 0.5 else ""
                print(f"  {feat:<28} {var:<12} {n:>5} {k:>4} {eff:>8.3f} "
                      f"{power:>7.3f}{flag}")
    except ImportError:
        print("  [SKIP] statsmodels.stats.power not available — install statsmodels>=0.14")

    # ── Semantic analysis ─────────────────────────────────────────────
    try:
        semantic_bias_analysis(df)
    except (ModuleNotFoundError, ImportError) as _sem_err:
        print(f"  [SKIP] Semantic analysis skipped — {_sem_err}")

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ DIAGNOSTICS ═══")
    print("═" * 70)

    run_model_diagnostics(df)

    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ DESCRIPTIVE ANALYSES — PLOTS & SUMMARIES ═══")
    print("═" * 70)

    # ── Visualisations ────────────────────────────────────────────────
    print("\n[7] Generating visualisations ...")
    plot_feature_heatmap(raw_stats, fdr_sig, features)
    plot_feature_correlation(df, features)

    races     = sorted(df["race"].unique())
    genders   = ["female","male"]
    ages      = ["20","40","70"]

    plot_feature_boxplots_by_demo(df, features, "race",      races,   "Race")
    plot_feature_boxplots_by_demo(df, features, "gender",    genders, "Gender")
    plot_feature_boxplots_by_demo(df, features, "age_group", ages,    "Age Group")

    if not all_coefs_df.empty:
        for feat in features:
            plot_regression_forest(all_coefs_df, feat)

    for chatbot in CHATBOTS:
        sub = df[df["model"] == chatbot]
        mean_by_race = sub.groupby("race")[features].mean()
        plot_summary_radar(mean_by_race, chatbot, features)

    # ── Paper-level FDR summary ───────────────────────────────────────
    # Pool ALL raw p-values from every analysis stage and apply a single joint
    # BH correction. This gives an honest paper-level false-discovery estimate.
    # Compare: if fewer results survive joint correction than stage-level
    # correction, those are likely false positives inflated by multiple families.
    print("\n[8] Paper-level FDR: joint correction across all analysis stages ...")
    print("  Collecting p-values from: group tests, MixedLM ...")
    _paper_ps: list[float] = []
    _paper_labels: list[str] = []

    # Group-test p-values
    for (feat, var), (_, p, _) in raw_stats.items():
        if not np.isnan(p):
            _paper_ps.append(float(p))
            _paper_labels.append(f"GroupTest/{feat}/{var}")

    # MixedLM p-values (all_coefs_df already collected above)
    if not all_coefs_df.empty and "p_raw" in all_coefs_df.columns:
        for _, row in all_coefs_df.iterrows():
            if not np.isnan(row.get("p_raw", np.nan)):
                _paper_ps.append(float(row["p_raw"]))
                _paper_labels.append(f"MixedLM/{row.get('feature','?')}/{row.get('term','?')}")

    if len(_paper_ps) >= 2:
        _joint_corrected = apply_fdr(_paper_ps)
        _n_sig_joint  = int(np.sum(_joint_corrected < ALPHA))
        _n_sig_stages = int(sum(
            1 for v in corrected_ps if not np.isnan(v) and v < ALPHA
        ))
        print(f"\n  Total p-values pooled:          {len(_paper_ps)}")
        print(f"  Significant (stage-level FDR):  ~{_n_sig_stages}  (across separate families)")
        print(f"  Significant (joint paper FDR):  {_n_sig_joint}  (single family — more conservative)")
        if _n_sig_joint < _n_sig_stages:
            print(f"\n  ⚠  {_n_sig_stages - _n_sig_joint} result(s) survive stage-level FDR but not "
                  f"joint correction — treat those as exploratory regardless of confirmatory label.")
        else:
            print("  ✓  All stage-level FDR results also survive joint correction.")
        paper_fdr_df = pd.DataFrame({
            "label":    _paper_labels,
            "p_raw":    _paper_ps,
            "p_fdr_joint": _joint_corrected,
            "sig_joint": _joint_corrected < ALPHA,
        })
        out_fdr = str(_SCRIPT_DIR / "paper_level_fdr.csv")
        paper_fdr_df.to_csv(out_fdr, index=False)
        print(f"  Full joint-FDR table → {out_fdr}")
    else:
        print("  [SKIP] Fewer than 2 valid p-values available for joint correction.")

    # ── Validation sample ─────────────────────────────────────────────
    print("\n[9] Generating validation sample ...")
    generate_validation_sample(df)

    # ── Save ──────────────────────────────────────────────────────────
    out_csv = str(_SCRIPT_DIR / "clinical_features_v3.csv")
    df.drop(columns=["response","emb"] if "emb" in df.columns else ["response"],
            errors="ignore").to_csv(out_csv, index=False)
    print(f"\n[10] Features saved to: {out_csv}")
    print(f"\n Done. Plots → {PLOT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
