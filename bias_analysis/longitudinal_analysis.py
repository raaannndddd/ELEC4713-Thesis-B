"""
Exploratory Longitudinal and Trajectory Analysis.

Analyses how Doctronic and DrKhan responses evolve across conversation turns.
Uses multi-turn conversation data (data/web_convo_long.json).

Outputs (bias_analysis/):
  longitudinal_features.csv              — per-turn feature matrix
  longitudinal_trajectory_stats.csv      — MixedLM trajectory slope per feature
  longitudinal_bias_conv_level_tests.csv — KW demographic bias tests (conv-aggregated)
  longitudinal_cross_chatbot_stats.csv   — Mann-Whitney per turn (Doctronic vs DrKhan)
  longitudinal_short_vs_long_stats.csv   — short vs. multi-turn comparisons
  longitudinal_bias_regression.csv       — demographic + trajectory MixedLM regression

Plots in bias_analysis/plots/longitudinal/

Usage (from project root):
  python bias_analysis/longitudinal_analysis.py
  python bias_analysis/longitudinal_analysis.py --no-nlp
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

try:
    from bias_analysis.shared_quantitative_features import extract_features
    from bias_analysis.analysis_utils import apply_fdr, rank_biserial_r, effect_label_r, effect_label_eps2
    from bias_analysis.analysis_constants import LONGITUDINAL_CHATBOTS, LONGITUDINAL_COLORS, LONGITUDINAL_LABELS
except ImportError:
    from shared_quantitative_features import extract_features
    from analysis_utils import apply_fdr, rank_biserial_r, effect_label_r, effect_label_eps2
    from analysis_constants import LONGITUDINAL_CHATBOTS, LONGITUDINAL_COLORS, LONGITUDINAL_LABELS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT   = pathlib.Path(__file__).parent
_ROOT     = _SCRIPT.parent
DATA_PATH  = str(_ROOT / "data" / "web_convo_long.json")
SHORT_PATH = str(_ROOT / "data" / "web_convo_short.json")
OUT_DIR    = _SCRIPT / "plots" / "longitudinal"
FEAT_CSV   = _SCRIPT / "longitudinal_features.csv"
TRAJ_CSV   = _SCRIPT / "longitudinal_trajectory_stats.csv"
BIAS_CSV   = _SCRIPT / "longitudinal_bias_conv_level_tests.csv"
CROSS_CSV  = _SCRIPT / "longitudinal_cross_chatbot_stats.csv"
SVSL_CSV   = _SCRIPT / "longitudinal_short_vs_long_stats.csv"
REGR_CSV   = _SCRIPT / "longitudinal_bias_regression.csv"

ALPHA    = 0.05
MAX_TURN = 16
CHATBOTS = LONGITUDINAL_CHATBOTS
COLORS   = LONGITUDINAL_COLORS
LABELS   = LONGITUDINAL_LABELS
DEMO_VARS = ["race", "gender", "age_group"]

ALL_FEATURES = [
    "word_count", "sentence_count", "avg_sentence_length", "paragraph_count",
    "question_count", "mattr", "flesch_reading_ease", "avg_word_length",
    "medical_term_count", "urgency_score", "warning_signs_count",
    "medication_specificity", "differential_count", "emergency_advice",
    "safety_warning_count", "risk_language_score", "empathy_score",
    "reassurance_count", "politeness_count",
    "semantic_similarity_prev", "semantic_similarity_t1",
]

# Features suitable for linear MixedLM regression (excludes ordinal + NLP)
REGRESSION_FEATURES = [
    "word_count", "sentence_count", "avg_sentence_length", "paragraph_count",
    "question_count", "mattr", "flesch_reading_ease", "avg_word_length",
    "medical_term_count", "differential_count", "emergency_advice",
    "safety_warning_count", "risk_language_score", "reassurance_count",
    "politeness_count", "warning_signs_count",
]

FEATURE_LABELS = {
    "word_count":              "Response Length (words)",
    "sentence_count":          "Sentence Count",
    "avg_sentence_length":     "Avg Sentence Length",
    "paragraph_count":         "Paragraph Count",
    "question_count":          "Follow-up Questions",
    "mattr":                   "Lexical Diversity (MATTR)",
    "flesch_reading_ease":     "Flesch Reading Ease",
    "avg_word_length":         "Avg Word Length",
    "medical_term_count":      "Medical Term Count",
    "urgency_score":           "Urgency Score",
    "warning_signs_count":     "Warning Signs Count",
    "medication_specificity":  "Medication Specificity",
    "differential_count":      "Differential Diagnoses",
    "emergency_advice":        "Emergency Advice",
    "safety_warning_count":    "Safety Warnings",
    "risk_language_score":     "Risk Language Score",
    "empathy_score":           "Empathy Score",
    "reassurance_count":       "Reassurance Count",
    "politeness_count":        "Politeness Markers",
    "semantic_similarity_prev":"Semantic Similarity (prev turn)",
    "semantic_similarity_t1":  "Semantic Similarity (turn 1)",
}


def _age_group(age: int) -> str:
    if age <= 25:  return "Young (20)"
    if age <= 55:  return "Middle (40)"
    return "Older (70)"


def _severity_num(sev: str) -> int:
    return {"mild": 0, "moderate": 1, "severe": 2}.get(str(sev).lower(), 1)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_feature_df(records: list, use_nlp: bool) -> pd.DataFrame:
    """Extract per-turn features from multi-turn conversation records."""
    enc = None
    if use_nlp:
        try:
            from sentence_transformers import SentenceTransformer
            print("  Loading sentence-transformer model …")
            enc = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("  [WARN] sentence-transformers not installed; semantic features → NaN")

    rows = []
    for conv_id, rec in enumerate(records):
        meta  = rec.get("metadata", {})
        model = rec.get("model", "")
        if model not in CHATBOTS:
            continue

        transcript = rec.get("transcript", [])
        age_val = meta.get("age", 40)
        try:
            age_int = int(float(age_val))
        except (TypeError, ValueError):
            age_int = 40

        severity_str = meta.get("severity", "moderate")

        # Extract chatbot turns in order
        chatbot_texts = [
            m["text"] for m in transcript if m.get("role") == "chatbot"
        ]
        if not chatbot_texts:
            continue

        # Embed all chatbot turns at once if NLP enabled
        embs = None
        if enc is not None and chatbot_texts:
            try:
                embs = enc.encode(chatbot_texts, convert_to_numpy=True, show_progress_bar=False)
            except Exception:
                embs = None

        t1_emb = embs[0] if embs is not None else None

        for turn_idx, text in enumerate(chatbot_texts):
            turn_num = turn_idx + 1
            if turn_num > MAX_TURN:
                break
            if not text.strip():
                continue

            feats = extract_features(text, severity_str)

            # Semantic similarity features
            sem_prev = np.nan
            sem_t1   = np.nan
            if embs is not None:
                from sklearn.metrics.pairwise import cosine_similarity
                if turn_idx > 0:
                    sem_prev = float(cosine_similarity(
                        embs[turn_idx].reshape(1, -1),
                        embs[turn_idx - 1].reshape(1, -1)
                    )[0, 0])
                sem_t1 = float(cosine_similarity(
                    embs[turn_idx].reshape(1, -1),
                    t1_emb.reshape(1, -1)
                )[0, 0])

            rows.append({
                "conversation_id": conv_id,
                "model":           model,
                "turn_number":     turn_num,
                "gender":          meta.get("gender", "unknown"),
                "race":            meta.get("race",   "unknown"),
                "age":             age_int,
                "age_group":       _age_group(age_int),
                "severity":        severity_str,
                "severity_num":    _severity_num(severity_str),
                "symptom":         meta.get("symptom", "unknown"),
                **feats,
                "semantic_similarity_prev": sem_prev,
                "semantic_similarity_t1":  sem_t1,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 1: Trajectory (MixedLM slope over turn_number)
# ---------------------------------------------------------------------------

def run_trajectory_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-chatbot × per-feature MixedLM: feature ~ turn_number | conversation_id."""
    rows = []
    for model in CHATBOTS:
        sub = df[df["model"] == model].copy()
        for feat in ALL_FEATURES:
            if feat not in sub.columns or sub[feat].isna().all():
                continue
            sub_f = sub[["conversation_id", "turn_number", feat]].dropna()
            if len(sub_f) < 20:
                continue
            try:
                res = smf.mixedlm(f"{feat} ~ turn_number", sub_f,
                                  groups=sub_f["conversation_id"]).fit(reml=True)
                coef = float(res.params["turn_number"])
                ci   = res.conf_int().loc["turn_number"]
                p    = float(res.pvalues["turn_number"])
                rows.append({
                    "model":          model,
                    "feature":        feat,
                    "coef_turn":      coef,
                    "ci_lower":       float(ci[0]),
                    "ci_upper":       float(ci[1]),
                    "p_raw":          p,
                    "converged":      res.converged,
                    "direction":      "increasing" if coef > 0 else "decreasing",
                    "ordinal_approx": False,
                    "turns_included": "1+",
                    "fdr_scope":      "within_chatbot",
                })
            except Exception as e:
                pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr"] = apply_fdr(result["p_raw"].tolist())
    return result


# ---------------------------------------------------------------------------
# Analysis 2: Demographic bias at conversation level (KW)
# ---------------------------------------------------------------------------

def _kruskal_eps2(stat: float, n: int, k: int) -> float:
    denom = n - k
    return float((stat - k + 1) / denom) if denom > 0 else np.nan


def run_conv_level_bias(df: pd.DataFrame) -> pd.DataFrame:
    """KW tests on conversation-aggregated features across demographic groups."""
    rows = []
    for model in CHATBOTS:
        sub = df[df["model"] == model]
        # Aggregate per conversation
        agg_cols = [f for f in ALL_FEATURES if f in sub.columns]
        conv_agg = sub.groupby("conversation_id")[agg_cols + DEMO_VARS].agg(
            {**{f: "mean" for f in agg_cols}, **{d: "first" for d in DEMO_VARS}}
        ).reset_index()

        n_conv = len(conv_agg)
        for feat in agg_cols:
            for demo in DEMO_VARS:
                groups_vals = [
                    conv_agg.loc[conv_agg[demo] == g, feat].dropna().values
                    for g in conv_agg[demo].dropna().unique()
                ]
                groups_vals = [g for g in groups_vals if len(g) >= 3]
                if len(groups_vals) < 2:
                    continue
                try:
                    stat, p = kruskal(*groups_vals)
                    k_groups = len(groups_vals)
                    n_total  = sum(len(g) for g in groups_vals)
                    eps2 = _kruskal_eps2(stat, n_total, k_groups)
                    rows.append({
                        "model":           model,
                        "feature":         feat,
                        "groupvar":        demo,
                        "n_conversations": n_conv,
                        "analysis_type":   "primary_conv_aggregated",
                        "test_type":       "KW",
                        "stat":            stat,
                        "p_raw":           p,
                        "effect_size":     eps2,
                        "effect_label_str": effect_label_eps2(eps2 if eps2 is not np.nan else np.nan),
                        "fdr_scope":       "global_all_feature_x_variable_x_chatbot",
                        "ordinal_approx":  False,
                    })
                except Exception:
                    pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr"]  = apply_fdr(result["p_raw"].tolist())
        result["sig_fdr"] = result["p_fdr"] < ALPHA
        result = result[["model","feature","groupvar","n_conversations","analysis_type",
                          "p_fdr","sig_fdr","test_type","stat","p_raw","effect_size",
                          "effect_label_str","fdr_scope","ordinal_approx"]]
    return result


# ---------------------------------------------------------------------------
# Analysis 3: Cross-chatbot per-turn Mann-Whitney
# ---------------------------------------------------------------------------

def run_cross_chatbot(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U at each turn comparing Doctronic vs DrKhan."""
    rows = []
    all_turns = sorted(df["turn_number"].unique())
    feats = [f for f in ALL_FEATURES if f in df.columns]

    for turn in all_turns:
        d_sub = df[(df["model"] == "doctronic") & (df["turn_number"] == turn)]
        k_sub = df[(df["model"] == "drkhan")    & (df["turn_number"] == turn)]
        if len(d_sub) < 5 or len(k_sub) < 5:
            continue

        status = "confirmatory_turn1" if turn == 1 else f"exploratory_turn{turn}"

        for feat in feats:
            a = d_sub[feat].dropna().values
            b = k_sub[feat].dropna().values
            if len(a) < 3 or len(b) < 3:
                continue
            try:
                U, p = mannwhitneyu(a, b, alternative="two-sided")
                r = rank_biserial_r(U, len(a), len(b))
                rows.append({
                    "turn_number":      turn,
                    "feature":          feat,
                    "n_doctronic":      len(a),
                    "n_drkhan":         len(b),
                    "mean_doctronic":   float(a.mean()),
                    "mean_drkhan":      float(b.mean()),
                    "U":                U,
                    "p_raw":            p,
                    "effect_r":         r,
                    "inferential_status": status,
                    "fdr_scope":        "global_all_turn_x_feature_tests",
                })
            except Exception:
                pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr"]  = apply_fdr(result["p_raw"].tolist())
        result["sig_fdr"] = result["p_fdr"] < ALPHA
        result = result[["turn_number","feature","n_doctronic","n_drkhan",
                          "mean_doctronic","mean_drkhan","U","p_raw","p_fdr",
                          "effect_r","sig_fdr","inferential_status","fdr_scope"]]
    return result


# ---------------------------------------------------------------------------
# Analysis 4: Short vs Long comparisons
# ---------------------------------------------------------------------------

def run_short_vs_long(df_long: pd.DataFrame, short_path: str) -> pd.DataFrame:
    """Mann-Whitney comparing single-turn (short) vs multi-turn (long) responses."""
    try:
        with open(short_path) as f:
            short_recs = json.load(f)
    except FileNotFoundError:
        print(f"  [WARN] Short data not found at {short_path}; skipping short-vs-long.")
        return pd.DataFrame()

    # Extract features from short-form data
    short_rows = []
    for rec in short_recs:
        if rec.get("model") not in CHATBOTS:
            continue
        response = rec.get("response", "").strip()
        if not response:
            continue
        feats = extract_features(response, rec.get("metadata", {}).get("severity", "moderate"))
        short_rows.append({"model": rec["model"], **feats})
    df_short = pd.DataFrame(short_rows)

    rows = []
    feats = [f for f in ALL_FEATURES if f in df_long.columns and f in df_short.columns
             and not f.startswith("semantic_")]

    for model in CHATBOTS:
        short_m  = df_short[df_short["model"] == model]
        long_m   = df_long[df_long["model"]   == model]
        long_t1  = long_m[long_m["turn_number"] == 1]
        long_all = long_m
        turn_max = long_m["turn_number"].max() if not long_m.empty else 1
        long_last = long_m[long_m["turn_number"] == turn_max]

        comparisons = [
            ("short vs long_turn1",     short_m, long_t1),
            ("short vs long_all",        short_m, long_all),
            (f"long_turn1 vs long_turn{int(turn_max)}", long_t1, long_last),
        ]

        for label, g_a, g_b in comparisons:
            for feat in feats:
                a = g_a[feat].dropna().values if feat in g_a.columns else np.array([])
                b = g_b[feat].dropna().values if feat in g_b.columns else np.array([])
                if len(a) < 3 or len(b) < 3:
                    continue
                try:
                    U, p = mannwhitneyu(a, b, alternative="two-sided")
                    r = rank_biserial_r(U, len(a), len(b))
                    rows.append({
                        "model":      model,
                        "feature":    feat,
                        "comparison": label,
                        "n_a":        len(a),
                        "n_b":        len(b),
                        "mean_a":     float(a.mean()),
                        "mean_b":     float(b.mean()),
                        "U":          U,
                        "p_raw":      p,
                        "effect_r":   r,
                    })
                except Exception:
                    pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr"]  = apply_fdr(result["p_raw"].tolist())
        result["sig_fdr"] = result["p_fdr"] < ALPHA
        result = result[["model","feature","comparison","n_a","n_b",
                          "mean_a","mean_b","U","p_raw","p_fdr","effect_r","sig_fdr"]]
    return result


# ---------------------------------------------------------------------------
# Analysis 5: Demographic + trajectory MixedLM regression
# ---------------------------------------------------------------------------

def run_bias_regression(df: pd.DataFrame) -> pd.DataFrame:
    """MixedLM: feature ~ turn_number + demographics + severity + symptom | conv_id."""
    rows = []
    for model in CHATBOTS:
        sub = df[df["model"] == model].copy()
        sub["age_numeric"] = sub["age"].astype(float)
        for feat in REGRESSION_FEATURES:
            if feat not in sub.columns or sub[feat].isna().all():
                continue
            sub_f = sub[["conversation_id","turn_number","race","gender",
                          "age_numeric","severity_num","symptom", feat]].dropna()
            if len(sub_f) < 20:
                continue
            # Standardise the feature for interpretable coefficients
            feat_mean = float(sub_f[feat].mean())
            feat_std  = float(sub_f[feat].std())
            if feat_std < 1e-9:
                continue
            sub_f = sub_f.copy()
            sub_f[feat] = (sub_f[feat] - feat_mean) / feat_std
            formula = (f"{feat} ~ turn_number + C(race) + C(gender) "
                       f"+ age_numeric + severity_num + C(symptom)")
            try:
                res = smf.mixedlm(formula, sub_f,
                                  groups=sub_f["conversation_id"]).fit(reml=True)
                for term, coef in res.params.items():
                    if term in ("Intercept", "Group Var"):
                        continue
                    try:
                        ci  = res.conf_int().loc[term]
                        p   = float(res.pvalues[term])
                        rows.append({
                            "model":     model,
                            "feature":   feat,
                            "term":      term,
                            "coef":      float(coef),
                            "ci_lower":  float(ci[0]),
                            "ci_upper":  float(ci[1]),
                            "p_raw":     p,
                            "feat_mean": feat_mean,
                            "feat_std":  feat_std,
                            "fdr_scope": "within_model_x_feature_demographic_terms",
                            "ordinal_approx": False,
                        })
                    except Exception:
                        pass
            except Exception:
                pass

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr"] = apply_fdr(result["p_raw"].tolist())
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _savefig(path: pathlib.Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", **kw)
    plt.close()


def plot_survivorship(df: pd.DataFrame):
    """Turn survivorship: n conversations per turn per chatbot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for model in CHATBOTS:
        sub = df[df["model"] == model]
        counts = sub.groupby("turn_number")["conversation_id"].nunique()
        ax.plot(counts.index, counts.values,
                marker="o", markersize=4,
                color=COLORS[model], label=LABELS[model])
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Active conversations (n)")
    ax.set_title("Turn Survivorship — Active Conversations per Turn")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(OUT_DIR / "turn_survivorship.png")


def plot_trajectory(df: pd.DataFrame, feat: str):
    """Mean ± SE per turn for one feature, both chatbots."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for model in CHATBOTS:
        sub = df[(df["model"] == model) & (df[feat].notna())]
        stats = sub.groupby("turn_number")[feat].agg(["mean", "sem", "count"]).reset_index()
        stats = stats[stats["count"] >= 10]
        ax.plot(stats["turn_number"], stats["mean"],
                marker="o", markersize=4,
                color=COLORS[model], label=LABELS[model])
        ax.fill_between(stats["turn_number"],
                        stats["mean"] - stats["sem"],
                        stats["mean"] + stats["sem"],
                        alpha=0.2, color=COLORS[model])
    ax.set_xlabel("Turn number")
    ax.set_ylabel(FEATURE_LABELS.get(feat, feat))
    ax.set_title(f"Response Trajectory: {FEATURE_LABELS.get(feat, feat)}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(OUT_DIR / f"trajectory_{feat}.png")


def plot_trajectory_heatmap(traj_df: pd.DataFrame):
    """Heatmap of standardised trajectory slopes (coef_turn / std → directionality)."""
    if traj_df.empty:
        return
    pivot = traj_df.pivot_table(index="feature", columns="model", values="coef_turn")
    fig, ax = plt.subplots(figsize=(6, max(6, len(pivot) * 0.45)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", center=0,
        cmap="RdBu_r", linewidths=0.5, ax=ax,
        cbar_kws={"label": "Slope (units/turn)"},
        xticklabels=[LABELS.get(c, c) for c in pivot.columns],
    )
    ax.set_title("Trajectory Slopes (MixedLM coef_turn)")
    ax.set_xlabel("")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    _savefig(OUT_DIR / "trajectory_heatmap.png")


def plot_cross_chatbot_heatmap(cross_df: pd.DataFrame):
    """Heatmap of Mann-Whitney effect sizes per turn × feature."""
    if cross_df.empty:
        return
    pivot = cross_df.pivot_table(index="feature", columns="turn_number", values="effect_r")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.7),
                                   max(6, len(pivot) * 0.45)))
    sns.heatmap(
        pivot, annot=False, center=0,
        cmap="RdBu_r", linewidths=0.3, ax=ax,
        cbar_kws={"label": "Rank-biserial r"},
    )
    ax.set_title("Cross-chatbot Divergence per Turn (Mann-Whitney effect r)")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    _savefig(OUT_DIR / "cross_chatbot_divergence_heatmap.png")


def plot_demog_trajectories(df: pd.DataFrame, bias_df: pd.DataFrame):
    """Per-demographic-group trajectory for significant (model, feature, demo) combos."""
    if bias_df.empty:
        return
    sig = bias_df[bias_df["sig_fdr"]]
    DEMO_COLORS = {
        "race":     ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00"],
        "gender":   ["#E41A1C","#377EB8"],
        "age_group":["#377EB8","#FF7F00","#E41A1C"],
    }
    for _, row in sig.iterrows():
        model = row["model"]
        feat  = row["feature"]
        demo  = row["groupvar"]
        sub   = df[(df["model"] == model) & (df[feat].notna())]
        groups = sorted(sub[demo].dropna().unique())
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        cols = DEMO_COLORS.get(demo, plt.cm.tab10.colors)
        for i, grp in enumerate(groups):
            grp_sub  = sub[sub[demo] == grp]
            stats    = grp_sub.groupby("turn_number")[feat].agg(["mean","sem","count"]).reset_index()
            stats    = stats[stats["count"] >= 5]
            col      = cols[i % len(cols)]
            ax.plot(stats["turn_number"], stats["mean"],
                    marker="o", markersize=3, color=col, label=str(grp))
            ax.fill_between(stats["turn_number"],
                            stats["mean"] - stats["sem"],
                            stats["mean"] + stats["sem"],
                            alpha=0.15, color=col)
        ax.set_xlabel("Turn number")
        ax.set_ylabel(FEATURE_LABELS.get(feat, feat))
        ax.set_title(f"{LABELS[model]}: {FEATURE_LABELS.get(feat,feat)} by {demo}")
        ax.legend(title=demo, fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.3)
        fname = f"demog_trajectory_{model}_{feat}_{demo}.png"
        _savefig(OUT_DIR / fname)


def plot_short_vs_long(svsl_df: pd.DataFrame):
    """Bar plots comparing short vs long response features."""
    if svsl_df.empty:
        return
    feats = svsl_df["feature"].unique()
    for feat in feats:
        sub = svsl_df[svsl_df["feature"] == feat]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        comparisons = sub["comparison"].unique()
        x = np.arange(len(comparisons))
        width = 0.35
        for i, model in enumerate(CHATBOTS):
            m_sub = sub[sub["model"] == model]
            means_a = [m_sub[m_sub["comparison"] == c]["mean_a"].values[0]
                       if len(m_sub[m_sub["comparison"] == c]) > 0 else 0
                       for c in comparisons]
            means_b = [m_sub[m_sub["comparison"] == c]["mean_b"].values[0]
                       if len(m_sub[m_sub["comparison"] == c]) > 0 else 0
                       for c in comparisons]
            offset = (i - 0.5) * width
            ax.bar(x + offset - width * 0.25, means_a, width * 0.45,
                   color=COLORS[model], alpha=0.9, label=f"{LABELS[model]} (A)")
            ax.bar(x + offset + width * 0.25, means_b, width * 0.45,
                   color=COLORS[model], alpha=0.45, label=f"{LABELS[model]} (B)")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in comparisons], rotation=20, ha="right")
        ax.set_ylabel(FEATURE_LABELS.get(feat, feat))
        ax.set_title(f"Short vs Long: {FEATURE_LABELS.get(feat, feat)}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _savefig(OUT_DIR / f"short_vs_long_{feat}.png")


def plot_feature_profiles(df: pd.DataFrame):
    """Feature profile bar charts for three turn-tercile groups (Q1/Q2/Q3)."""
    turns = sorted(df["turn_number"].unique())
    n = len(turns)
    terciles = [
        ("q1", turns[:max(1, n//3)]),
        ("q2", turns[max(1, n//3): max(2, 2*n//3)]),
        ("q3", turns[max(2, 2*n//3):]),
    ]
    feats = [f for f in REGRESSION_FEATURES if f in df.columns]

    for q_label, q_turns in terciles:
        sub = df[df["turn_number"].isin(q_turns)]
        fig, axes = plt.subplots(1, len(CHATBOTS), figsize=(14, 5), sharey=False)
        for ax, model in zip(axes, CHATBOTS):
            m_sub = sub[sub["model"] == model]
            means = [m_sub[f].mean() for f in feats]
            norms = [(v - min(means)) / (max(means) - min(means) + 1e-9) for v in means]
            short_labels = [f.replace("_", "\n")[:14] for f in feats]
            ax.barh(range(len(feats)), norms, color=COLORS[model])
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(short_labels, fontsize=7)
            ax.set_xlabel("Normalised mean")
            ax.set_title(f"{LABELS[model]} — {q_label.upper()}")
            ax.grid(axis="x", alpha=0.3)
        plt.suptitle(f"Feature Profile — Turn Tercile {q_label.upper()}", fontweight="bold")
        plt.tight_layout()
        _savefig(OUT_DIR / f"feature_profile_{q_label}.png")


def plot_radar_profiles(df: pd.DataFrame):
    """Radar charts showing feature profiles per turn-tercile group."""
    turns = sorted(df["turn_number"].unique())
    n = len(turns)
    terciles = [
        ("q1", turns[:max(1, n//3)]),
        ("q2", turns[max(1, n//3): max(2, 2*n//3)]),
        ("q3", turns[max(2, 2*n//3):]),
    ]
    radar_feats = ["word_count","question_count","empathy_score","urgency_score",
                   "medical_term_count","risk_language_score","politeness_count",
                   "reassurance_count"]
    radar_feats = [f for f in radar_feats if f in df.columns]
    if len(radar_feats) < 3:
        return

    angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
    angles += angles[:1]

    for q_label, q_turns in terciles:
        sub = df[df["turn_number"].isin(q_turns)]
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw={"projection": "polar"})
        for model in CHATBOTS:
            m_sub = sub[sub["model"] == model]
            vals = [m_sub[f].mean() for f in radar_feats]
            # Min-max normalise across chatbots
            vals_norm = [(v / (max(abs(v), 1e-9))) for v in vals]
            vals_norm += vals_norm[:1]
            ax.plot(angles, vals_norm, color=COLORS[model], linewidth=2,
                    label=LABELS[model])
            ax.fill(angles, vals_norm, color=COLORS[model], alpha=0.1)
        ax.set_thetagrids(np.degrees(angles[:-1]),
                          [f.replace("_", "\n")[:12] for f in radar_feats],
                          fontsize=8)
        ax.set_title(f"Radar Profile — {q_label.upper()}", pad=15, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()
        _savefig(OUT_DIR / f"radar_turns_{q_label}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Longitudinal and trajectory analysis")
    p.add_argument("--no-nlp", action="store_true",
                   help="Skip sentence-transformer semantic features (faster)")
    p.add_argument("--data-path", default=DATA_PATH,
                   help="Path to multi-turn JSON data file")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    use_nlp = not args.no_nlp

    print("=" * 60)
    print("Longitudinal and Trajectory Analysis")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    data_path = args.data_path
    print(f"\nLoading multi-turn data: {data_path}")
    with open(data_path) as f:
        records = json.load(f)
    records = [r for r in records if r.get("model") in CHATBOTS]
    print(f"  {len(records)} conversations (Doctronic + DrKhan)")

    # ── Feature extraction ────────────────────────────────────────
    print("\nExtracting per-turn features …")
    df = build_feature_df(records, use_nlp=use_nlp)
    df = df[df["race"] != "unknown"]
    df = df[df["gender"] != "unknown"]
    print(f"  {len(df)} turn-level observations")
    print(f"  Chatbots: {df['model'].value_counts().to_dict()}")
    print(f"  Turn range: {df['turn_number'].min()} – {df['turn_number'].max()}")

    df.to_csv(FEAT_CSV, index=False)
    print(f"  Saved: {FEAT_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Analysis 1: Trajectory slopes ────────────────────────────
    print("\n[1/5] Trajectory analysis (MixedLM) …")
    traj_df = run_trajectory_analysis(df)
    traj_df.to_csv(TRAJ_CSV, index=False)
    print(f"  {len(traj_df)} rows → {TRAJ_CSV.name}")

    # ── Analysis 2: Demographic bias at conv level ────────────────
    print("\n[2/5] Conversation-level demographic bias (KW) …")
    bias_df = run_conv_level_bias(df)
    bias_df.to_csv(BIAS_CSV, index=False)
    print(f"  {len(bias_df)} rows → {BIAS_CSV.name}")
    n_sig = bias_df["sig_fdr"].sum() if not bias_df.empty else 0
    print(f"  Significant (FDR < {ALPHA}): {n_sig}")

    # ── Analysis 3: Cross-chatbot per-turn ───────────────────────
    print("\n[3/5] Cross-chatbot per-turn comparisons (Mann-Whitney) …")
    cross_df = run_cross_chatbot(df)
    cross_df.to_csv(CROSS_CSV, index=False)
    print(f"  {len(cross_df)} rows → {CROSS_CSV.name}")

    # ── Analysis 4: Short vs Long ────────────────────────────────
    print("\n[4/5] Short vs long comparisons …")
    svsl_df = run_short_vs_long(df, SHORT_PATH)
    if not svsl_df.empty:
        svsl_df.to_csv(SVSL_CSV, index=False)
        print(f"  {len(svsl_df)} rows → {SVSL_CSV.name}")
    else:
        print("  Skipped (short data unavailable).")

    # ── Analysis 5: Bias regression ──────────────────────────────
    print("\n[5/5] Demographic + trajectory regression (MixedLM) …")
    regr_df = run_bias_regression(df)
    if not regr_df.empty:
        regr_df.to_csv(REGR_CSV, index=False)
        print(f"  {len(regr_df)} rows → {REGR_CSV.name}")

    # ── Plots ────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_survivorship(df)
    print("  turn_survivorship.png")

    feats_to_plot = [f for f in ALL_FEATURES if f in df.columns
                     and not df[f].isna().all()]
    for feat in feats_to_plot:
        plot_trajectory(df, feat)
    print(f"  trajectory_*.png ({len(feats_to_plot)} features)")

    plot_trajectory_heatmap(traj_df)
    print("  trajectory_heatmap.png")

    plot_cross_chatbot_heatmap(cross_df)
    print("  cross_chatbot_divergence_heatmap.png")

    plot_demog_trajectories(df, bias_df)
    n_demog = len(list(OUT_DIR.glob("demog_trajectory_*.png")))
    print(f"  demog_trajectory_*.png ({n_demog} plots)")

    if not svsl_df.empty:
        plot_short_vs_long(svsl_df)
        n_svsl = len(list(OUT_DIR.glob("short_vs_long_*.png")))
        print(f"  short_vs_long_*.png ({n_svsl} plots)")

    plot_feature_profiles(df)
    print("  feature_profile_q*.png (3 plots)")

    plot_radar_profiles(df)
    print("  radar_turns_q*.png (3 plots)")

    total = len(list(OUT_DIR.glob("*.png")))
    print(f"\nDone. {total} plots saved to {OUT_DIR}")
    print(f"CSVs saved to {_SCRIPT}")


if __name__ == "__main__":
    main()
