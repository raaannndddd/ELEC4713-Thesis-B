"""Paired quantitative comparison of matched chatbot short-form responses.

Primary statistical test: Wilcoxon signed-rank test on matched prompt pairs
(one observation per chatbot per prompt condition). BH-FDR correction is
applied across all feature comparisons.
"""

import argparse
import json
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from math import pi
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler
from analysis_utils import apply_fdr, effect_label_r, r_from_wilcoxon  # noqa: E402
try:
    from bias_analysis.analysis_constants import SHORT_CHATBOTS, SHORT_COLORS, SHORT_LABELS
    from bias_analysis.feature_registry import QUANTITATIVE_FEATURES, feature_meta_tuples
    from bias_analysis.prompt_blocks import attach_complete_prompt_blocks, make_prompt_key
    from bias_analysis.schema_validation import SchemaValidationError, validate_feature_frame, validate_records
    from bias_analysis.shared_quantitative_features import extract_features, flesch_reading_ease, mattr
except ImportError:
    from analysis_constants import SHORT_CHATBOTS, SHORT_COLORS, SHORT_LABELS
    from feature_registry import QUANTITATIVE_FEATURES, feature_meta_tuples
    from prompt_blocks import attach_complete_prompt_blocks, make_prompt_key
    from schema_validation import SchemaValidationError, validate_feature_frame, validate_records
    from shared_quantitative_features import extract_features, flesch_reading_ease, mattr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/web_conversations.json"
PLOT_DIR   = "bias_analysis/plots/quant"
OUT_CSV    = "bias_analysis/quant_features.csv"
STAT_CSV   = "bias_analysis/quant_paired_stats.csv"
SIM_CSV    = "bias_analysis/quant_semantic_similarity.csv"

os.makedirs(PLOT_DIR, exist_ok=True)

CHATBOTS   = SHORT_CHATBOTS
LABELS     = SHORT_LABELS
ALPHA      = 0.05
COL_COLORS = SHORT_COLORS
FDR_SCOPE_NOTE = (
    "confirmatory-global: BH correction is applied across all paired Wilcoxon "
    "comparisons before any p-value is interpreted."
)

FEATURE_META = feature_meta_tuples(QUANTITATIVE_FEATURES)
ALL_FEATURES = list(QUANTITATIVE_FEATURES)


# =============================================================================
# 2. DATA LOADING & PAIRING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    try:
        raw = validate_records(raw, kind="short")
    except SchemaValidationError as exc:
        raise SystemExit(f"[FATAL] Input schema validation failed for {path}: {exc}") from exc

    rows = []
    for r in raw:
        response = r.get("response", "") or ""
        if not response.strip() or response.startswith("ERROR") or response == "No response found":
            continue
        meta  = r.get("metadata", {})
        feats = extract_features(response, meta.get("severity", "mild"))
        row   = {
            "model":    r["model"],
            "response": response,
            "symptom":  meta.get("symptom",  "unknown"),
            "severity": meta.get("severity", "mild"),
            "gender":   meta.get("gender",   "unknown"),
            "race":     meta.get("race",     "unknown"),
            "age":      meta.get("age",      40),
        }
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    validate_feature_frame(
        df,
        feature_names=ALL_FEATURES,
        required_columns=["model", "response", "symptom", "severity", "gender", "race", "age"],
        context="quantitative short-conversation feature matrix",
    )
    return df


def validate_complete_prompt_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate matched prompt blocks and return the complete subset."""
    return attach_complete_prompt_blocks(df, CHATBOTS, verbose=True)


# =============================================================================
# 3. SEMANTIC SIMILARITY  (sentence-transformers)
# =============================================================================

def compute_semantic_similarity(df_paired: pd.DataFrame) -> pd.DataFrame:
    """
    For each prompt_id compute pairwise cosine similarity
    between chatbot responses.
    """
    print("  Computing sentence embeddings...")
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df_paired["response"].tolist()
    embs  = model.encode(texts, batch_size=32, show_progress_bar=True,
                         convert_to_numpy=True)
    df_paired = df_paired.copy()
    df_paired["embedding"] = list(embs)

    pairs = [("doctronic", "drkhan")]

    sim_rows = []
    for pid, group in df_paired.groupby("prompt_id"):
        emb_map = {row["model"]: row["embedding"]
                   for _, row in group.iterrows()}
        for m1, m2 in pairs:
            if m1 not in emb_map or m2 not in emb_map:
                continue
            v1, v2 = emb_map[m1], emb_map[m2]
            sim = float(np.dot(v1, v2) /
                        (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            meta = group.iloc[0]
            sim_rows.append({
                "prompt_id": pid,
                "symptom":   meta["symptom"],
                "severity":  meta["severity"],
                "pair":      f"{LABELS[m1]} vs {LABELS[m2]}",
                "cosine_sim": round(sim, 4),
                "cosine_dist": round(1 - sim, 4),
            })

    return pd.DataFrame(sim_rows)


# =============================================================================
# 4. PAIRED STATISTICAL TESTS
# =============================================================================

def paired_stats(df_paired: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature run a Wilcoxon signed-rank test on within-prompt
    differences (Doctronic minus DrKhan). BH-FDR applied across all features.

    Returns a tidy DataFrame with stat, p (raw), p (FDR), effect size r.
    """
    comparisons = [
        ("doctronic", "drkhan", "Doctronic vs DrKhan"),
    ]

    records = []
    for feat in ALL_FEATURES:
        for m1, m2, label in comparisons:
            pivot = df_paired.pivot_table(
                index="prompt_id", columns="model", values=feat, aggfunc="first"
            )
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            d = (pivot[m1] - pivot[m2]).dropna()
            if len(d) < 5:
                continue
            try:
                wresult = wilcoxon(d, alternative="two-sided", zero_method="zsplit")
                stat, p = wresult.statistic, wresult.pvalue
                r = r_from_wilcoxon(wresult, len(d))
            except Exception as exc:
                print(
                    f"  [WARNING] {type(exc).__name__} in Wilcoxon for "
                    f"{label} / {feat}: {exc}"
                )
                stat, p, r = np.nan, np.nan, np.nan
            records.append({
                "feature":    feat,
                "comparison": label,
                "chatbot_a":  m1,
                "chatbot_b":  m2,
                "n_pairs":    len(d),
                "mean_a":     pivot[m1].mean(),
                "mean_b":     pivot[m2].mean(),
                "mean_diff":  d.mean(),
                "median_diff": d.median(),
                "stat_W":     stat,
                "p_raw":      p,
                "effect_r":   round(r, 4) if not np.isnan(r) else np.nan,
            })

    results = pd.DataFrame(records)
    results["p_fdr"] = apply_fdr(results["p_raw"].tolist())
    results["sig_fdr"] = results["p_fdr"] < ALPHA
    results["effect_label"] = results["effect_r"].apply(effect_label_r)
    results["fdr_scope"] = FDR_SCOPE_NOTE
    return results


# =============================================================================
# 5. VISUALISATIONS
# =============================================================================

PALETTE = [COL_COLORS[c] for c in CHATBOTS]


def _save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_means_bar(df_paired: pd.DataFrame):
    """Side-by-side bar chart of feature means per chatbot, grouped by category."""
    groups = {}
    for feat, (grp, _) in FEATURE_META.items():
        groups.setdefault(grp, []).append(feat)

    for grp, feats in groups.items():
        nf = len(feats)
        fig, axes = plt.subplots(1, nf, figsize=(3.5 * nf, 5), sharey=False)
        if nf == 1:
            axes = [axes]

        for ax, feat in zip(axes, feats):
            means = df_paired.groupby("model")[feat].mean().reindex(CHATBOTS)
            sems  = df_paired.groupby("model")[feat].sem().reindex(CHATBOTS)
            bar_labels = [LABELS[c] for c in CHATBOTS]
            colors = [COL_COLORS[c] for c in CHATBOTS]
            bars = ax.bar(bar_labels, means.values, color=colors,
                          yerr=sems.values, capsize=4,
                          edgecolor="white", linewidth=0.7)
            ax.set_title(FEATURE_META[feat][1], fontsize=10, fontweight="bold")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            for bar, val in zip(bars, means.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        fig.suptitle(f"Feature Group: {grp}", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        fname = f"bar_{grp.lower().replace(' ', '_').replace('/', '_')}.png"
        _save(fig, fname)


def plot_boxplots(df_paired: pd.DataFrame, features: list, title_prefix: str, fname_prefix: str):
    """Box plots for a set of features, one subplot per feature."""
    n = len(features)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, features):
        data = [df_paired.loc[df_paired["model"] == c, feat].dropna().values
                for c in CHATBOTS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops={"color": "black", "linewidth": 2},
                        flierprops={"marker": ".", "markersize": 4, "alpha": 0.5})
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(CHATBOTS) + 1))
        ax.set_xticklabels([LABELS[c] for c in CHATBOTS], rotation=15, fontsize=9)
        ax.set_title(FEATURE_META[feat][1], fontsize=10)
        ax.set_ylabel("")

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(title_prefix, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, f"{fname_prefix}_boxplots.png")


def plot_radar(df_paired: pd.DataFrame):
    """
    Radar / spider chart showing normalised mean feature scores per chatbot.
    Uses a curated subset of representative features across all 5 categories.
    """
    radar_features = [
        "word_count", "mattr", "flesch_reading_ease",
        "urgency_score", "warning_signs_count", "medication_specificity",
        "safety_warning_count", "empathy_score", "question_count",
    ]
    radar_labels = [
        "Verbosity", "Lexical\nDiversity", "Readability",
        "Urgency", "Warning\nSigns", "Medication\nDetail",
        "Safety\nWarnings", "Empathy", "Questions",
    ]

    means = df_paired.groupby("model")[radar_features].mean().reindex(CHATBOTS)
    scaler  = MinMaxScaler()
    normed  = pd.DataFrame(scaler.fit_transform(means),
                           index=means.index, columns=radar_features)

    N      = len(radar_features)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"polar": True})
    for chatbot, color in COL_COLORS.items():
        values = normed.loc[chatbot, radar_features].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.2, label=LABELS[chatbot])
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("Chatbot Behavioural Profiles\n"
                 "(min-max normalised — relative profiles only, not absolute values)",
                 fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    _save(fig, "radar_chatbot_profiles.png")


def plot_heatmap_stat(stat_df: pd.DataFrame):
    """
    Heatmap of Wilcoxon effect sizes (r) with FDR significance markers.
    """
    comparisons = stat_df["comparison"].unique()
    fig, axes   = plt.subplots(1, len(comparisons), figsize=(7 * len(comparisons), 10),
                               sharey=True)
    axes_list = [axes] if len(comparisons) == 1 else list(axes)
    cmap = plt.cm.RdYlGn

    for ax, comp in zip(axes_list, comparisons):
        sub  = stat_df[stat_df["comparison"] == comp].set_index("feature")
        r_vals  = sub.reindex(ALL_FEATURES)["effect_r"].values.astype(float)
        sig     = sub.reindex(ALL_FEATURES)["sig_fdr"].fillna(False).values

        dir_sign = np.sign(sub.reindex(ALL_FEATURES)["mean_diff"].fillna(0).values)
        r_signed = r_vals * dir_sign

        im = ax.imshow(r_signed.reshape(-1, 1), aspect="auto",
                       cmap="coolwarm_r", vmin=-0.7, vmax=0.7)

        ax.set_yticks(range(len(ALL_FEATURES)))
        ax.set_yticklabels([FEATURE_META[f][1] for f in ALL_FEATURES], fontsize=9)
        ax.set_xticks([])
        ax.set_title(comp, fontsize=11, fontweight="bold")

        for i, (r, s) in enumerate(zip(r_vals, sig)):
            label = f"{r:.2f}{'★' if s else ''}"
            color = "white" if abs(r_vals[i]) > 0.4 else "black"
            ax.text(0, i, label, ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=axes_list[-1], label="Effect size r (signed by direction)",
                 fraction=0.04, pad=0.04)
    fig.suptitle("Paired Wilcoxon Effect Sizes (BH-FDR corrected)\n"
                 "★ = p_FDR < 0.05  |  + = Doctronic > DrKhan  |  − = Doctronic < DrKhan",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "wilcoxon_effect_heatmap.png")


def plot_semantic_similarity(sim_df: pd.DataFrame):
    """Box plot of cosine similarity per pair and per symptom."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    pairs = sim_df["pair"].unique()
    pair_colors = ["#4C72B0"]
    data = [sim_df.loc[sim_df["pair"] == p, "cosine_sim"].values for p in pairs]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 4})
    for patch, color in zip(bp["boxes"], pair_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(pairs, rotation=10, fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Response Similarity Between Chatbots\n(per prompt)", fontsize=12)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)

    ax = axes[1]
    syms = sorted(sim_df["symptom"].unique())
    sns.boxplot(data=sim_df, x="symptom", y="cosine_sim", hue="pair",
                palette=["#4C72B0"], ax=ax,
                width=0.6, flierprops={"marker": "."})
    ax.set_xticklabels(syms, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Response Similarity by Symptom", fontsize=12)
    ax.legend(fontsize=8, title="Pair", title_fontsize=8)

    plt.tight_layout()
    _save(fig, "semantic_similarity.png")


def plot_delta_scores(df_paired: pd.DataFrame):
    """
    Δ distributions for representative features (Doctronic − DrKhan).
    Positive = Doctronic scores higher on that feature.
    """
    pivot = df_paired.pivot_table(
        index="prompt_id", columns="model", values=ALL_FEATURES, aggfunc="first"
    )

    delta_feats = [
        "word_count", "mattr", "flesch_reading_ease",
        "urgency_score", "warning_signs_count",
        "safety_warning_count", "empathy_score",
    ]

    pairs = [("doctronic", "drkhan", "Doctronic − DrKhan")]

    delta_rows = []
    for feat in delta_feats:
        for m1, m2, label in pairs:
            if feat not in pivot.columns:
                continue
            cols = pivot[feat].columns
            if m1 not in cols or m2 not in cols:
                continue
            d = pivot[feat][m1] - pivot[feat][m2]
            for pid, val in d.dropna().items():
                delta_rows.append({
                    "prompt_id":  pid,
                    "feature":    FEATURE_META[feat][1],
                    "comparison": label,
                    "delta":      val,
                })

    delta_df = pd.DataFrame(delta_rows)
    if delta_df.empty:
        return

    feats_ordered = [FEATURE_META[f][1] for f in delta_feats
                     if FEATURE_META[f][1] in delta_df["feature"].values]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 4.5 * len(pairs)), squeeze=False)
    axes = axes.flatten()
    for ax, (_, _, label) in zip(axes, pairs):
        sub = delta_df[delta_df["comparison"] == label]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.violinplot(data=sub, x="feature", y="delta",
                       order=feats_ordered, inner="quartile",
                       palette="coolwarm", ax=ax)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"Δ Score Distribution: {label}\n"
                     "(positive = Doctronic scores higher)", fontsize=12)
        ax.set_ylabel("Δ")
        ax.tick_params(axis="x", rotation=25, labelsize=9)
    plt.tight_layout()
    _save(fig, "delta_scores_violin.png")


def plot_feature_correlation(df_paired: pd.DataFrame):
    """Spearman correlation heatmap across all features with significance stars."""
    from scipy.stats import spearmanr

    data = df_paired[ALL_FEATURES].dropna()
    n = len(ALL_FEATURES)
    labels = [FEATURE_META[f][1] for f in ALL_FEATURES]

    rho_mat = np.ones((n, n))
    p_mat   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                rho_mat[i, j], p_mat[i, j] = 1.0, 0.0
            else:
                r, p = spearmanr(data.iloc[:, i], data.iloc[:, j])
                rho_mat[i, j] = r
                p_mat[i, j]   = p

    rho_df = pd.DataFrame(rho_mat, index=ALL_FEATURES, columns=ALL_FEATURES)
    mask   = np.triu(np.ones((n, n), dtype=bool))

    annot = np.full((n, n), "", dtype=object)
    for i in range(n):
        for j in range(i):
            r, p = rho_mat[i, j], p_mat[i, j]
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            annot[i, j] = f"{r:.2f}{star}"

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(rho_df, mask=mask, annot=annot, fmt="", center=0,
                cmap="coolwarm", ax=ax, square=True, linewidths=0.4,
                xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1,
                cbar_kws={"label": "Spearman ρ"},
                annot_kws={"size": 7})
    ax.set_title(
        f"Spearman Correlation — All Features (n={len(data)})\n"
        "** p<0.01  * p<0.05  (uncorrected)",
        fontsize=12, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    _save(fig, "feature_correlation.png")


def plot_mean_summary_table(df_paired: pd.DataFrame):
    """Colour-coded table image showing mean ± SD for every feature × chatbot."""
    summary = df_paired.groupby("model")[ALL_FEATURES].agg(["mean", "std"]).round(3)

    table_data = {}
    for feat in ALL_FEATURES:
        row = {}
        for cb in CHATBOTS:
            m = summary.loc[cb, (feat, "mean")]
            s = summary.loc[cb, (feat, "std")]
            row[LABELS[cb]] = f"{m:.2f} ± {s:.2f}"
        table_data[FEATURE_META[feat][1]] = row
    table_df = pd.DataFrame(table_data).T

    fig, ax = plt.subplots(figsize=(10, len(ALL_FEATURES) * 0.55 + 2))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)

    for j, cb in enumerate(CHATBOTS):
        tbl[(0, j)].set_facecolor(COL_COLORS[cb])
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Feature Summary: Mean ± SD by Chatbot", fontsize=13,
                 fontweight="bold", pad=14)
    plt.tight_layout()
    _save(fig, "summary_table.png")


def plot_per_symptom_profiles(df_paired: pd.DataFrame):
    """Chatbot means broken down by symptom for core features."""
    core = ["word_count", "urgency_score", "empathy_score", "safety_warning_count"]
    symptoms = sorted(df_paired["symptom"].unique())
    n_sym = len(symptoms)

    for feat in core:
        fig, axes = plt.subplots(1, n_sym, figsize=(4 * n_sym, 4.5), sharey=True)
        if n_sym == 1:
            axes = [axes]
        for ax, sym in zip(axes, symptoms):
            sub    = df_paired[df_paired["symptom"] == sym]
            means  = sub.groupby("model")[feat].mean().reindex(CHATBOTS)
            sems   = sub.groupby("model")[feat].sem().reindex(CHATBOTS)
            ax.bar([LABELS[c] for c in CHATBOTS], means.values,
                   color=[COL_COLORS[c] for c in CHATBOTS],
                   yerr=sems.values, capsize=4,
                   edgecolor="white", linewidth=0.6)
            ax.set_title(sym.replace("_", " ").title(), fontsize=10)
            ax.tick_params(axis="x", rotation=20, labelsize=8)
        axes[0].set_ylabel(FEATURE_META[feat][1], fontsize=11)
        fig.suptitle(f"{FEATURE_META[feat][1]} by Symptom", fontsize=13,
                     fontweight="bold")
        plt.tight_layout()
        _save(fig, f"symptom_profile_{feat}.png")


# =============================================================================
# 6. SUMMARY REPORT
# =============================================================================

def print_summary(df_paired: pd.DataFrame, stat_df: pd.DataFrame,
                  sim_df: pd.DataFrame | None):
    """Print a human-readable summary of key findings."""
    print("\n" + "=" * 70)
    print("SUMMARY REPORT — Chatbot Quantitative Comparison")
    print("=" * 70)

    print(f"\n  Paired prompt conditions: {df_paired['prompt_id'].nunique()}")
    print(f"  Responses: {dict(df_paired['model'].value_counts())}")

    print("\n── Feature Means (µ) & Standard Deviations (SD) ──")
    for feat in ALL_FEATURES:
        line = f"  {FEATURE_META[feat][1]:<30}"
        for cb in CHATBOTS:
            sub = df_paired[df_paired["model"] == cb][feat]
            line += f"  {LABELS[cb]}={sub.mean():.2f}(±{sub.std():.2f})"
        print(line)

    print("\n── Statistically Significant Paired Differences (BH-FDR p < 0.05) ──")
    sig = stat_df[stat_df["sig_fdr"] == True].copy()
    if sig.empty:
        print("  No significant differences after BH-FDR correction.")
    else:
        for _, row in sig.sort_values("effect_r", ascending=False).iterrows():
            print(f"  {row['comparison']:<30}  {FEATURE_META[row['feature']][1]:<25}"
                  f"  r={row['effect_r']:.3f} ({row['effect_label']})  "
                  f"p_FDR={row['p_fdr']:.4f}  "
                  f"Δmean={row['mean_diff']:+.3f}")

    if sim_df is not None and not sim_df.empty:
        print("\n── Average Cosine Similarity Between Chatbots ──")
        avg = sim_df.groupby("pair")["cosine_sim"].mean()
        for pair, s in avg.items():
            print(f"  {pair:<35}  sim={s:.4f}  dist={1-s:.4f}")

    print("\n" + "=" * 70)


# =============================================================================
# 7. CHATBOT × SYMPTOM INTERACTION TEST
# =============================================================================

def test_chatbot_symptom_interaction(df_paired: pd.DataFrame) -> pd.DataFrame:
    """
    Two-way OLS: feature ~ C(model) * C(symptom).
    A significant interaction means the chatbot effect is symptom-specific.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    records = []
    for feat in ALL_FEATURES:
        try:
            fit   = ols(f'Q("{feat}") ~ C(model) * C(symptom)', data=df_paired).fit()
            table = anova_lm(fit, typ=2)
            key   = [k for k in table.index if "model" in k.lower() and "symptom" in k.lower()]
            if not key:
                continue
            p = float(table.loc[key[0], "PR(>F)"])
            records.append({"feature": feat, "p_interaction": round(p, 6)})
        except Exception as exc:
            print(
                f"  [WARNING] {type(exc).__name__} in chatbot×symptom OLS "
                f"for {feat}: {exc}"
            )

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result["p_fdr"]   = np.round(apply_fdr(result["p_interaction"].tolist()), 6)
    result["sig_fdr"] = result["p_fdr"] < ALPHA
    return result.sort_values("p_fdr")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Paired quantitative chatbot comparison (Doctronic vs DrKhan).")
    p.add_argument("--no-nlp", action="store_true",
                   help="Skip sentence-transformer embeddings.")
    p.add_argument("--data-path", type=str, default=DATA_PATH, metavar="PATH",
                   help=f"Path to conversations JSON file (default: {DATA_PATH}).")
    return p.parse_args()


def main():
    args    = parse_args()
    use_nlp = not args.no_nlp

    print("=" * 70)
    print("Chatbot Quantitative Comparison  [Doctronic vs DrKhan — Paired Design]")
    print("Primary test: Wilcoxon signed-rank with BH-FDR correction")
    if use_nlp:
        print("Semantic similarity: ON")
    else:
        print("Semantic similarity: OFF  (pass --no-nlp to disable)")
    print("=" * 70)

    # ── Load & extract ────────────────────────────────────────
    print(f"\n[1] Loading {args.data_path} ...")
    df = load_data(args.data_path)
    df = df[df["model"].isin(CHATBOTS)].copy()
    print(f"  {len(df)} valid responses | {dict(df['model'].value_counts())}")

    # ── Validate prompt blocks ────────────────────────────────
    df_complete = validate_complete_prompt_blocks(df)

    # ── Build paired dataset ─────────────────────────────────
    print("\n[2] Building paired dataset (complete pairs only)...")
    df_paired = attach_complete_prompt_blocks(df_complete, CHATBOTS, verbose=False)
    n_prompts = df_paired["prompt_id"].nunique()
    print(f"  {len(df_paired)} responses across {n_prompts} paired prompt conditions")

    # ── Feature CSV ───────────────────────────────────────────
    save_cols = ["prompt_id", "prompt_key", "model", "symptom", "severity",
                 "gender", "race", "age"] + ALL_FEATURES
    df_paired[save_cols].to_csv(OUT_CSV, index=False)
    print(f"\n[3] Feature matrix saved → {OUT_CSV}")

    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ CONFIRMATORY ANALYSIS ═══")
    print("═" * 70)

    # ── Paired Wilcoxon signed-rank (PRIMARY) ─────────────────
    print("\n[4] Running paired Wilcoxon signed-rank tests with BH-FDR correction...")
    stat_df = paired_stats(df_paired)
    stat_df.to_csv(STAT_CSV, index=False)
    print(f"  Saved → {STAT_CSV}")
    sig_count = int(stat_df["sig_fdr"].sum())
    print(f"  Significant results after BH-FDR: {sig_count} / {len(stat_df)}")
    print(f"  FDR scope: {FDR_SCOPE_NOTE}")

    print(f"\n  {'Feature':<30}{'W stat':>10}{'p_raw':>10}{'p_FDR':>10}{'r':>8}{'Effect':>10}{'sig':>5}")
    print("  " + "-" * 76)
    for _, row in stat_df.iterrows():
        sig = "★" if row["sig_fdr"] else ""
        print(f"  {row['feature']:<30}{row['stat_W']:>10.1f}"
              f"{row['p_raw']:>10.4f}{row['p_fdr']:>10.4f}"
              f"{row['effect_r']:>8.3f}{row['effect_label']:>10}{sig:>5}")

    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ EXPLORATORY ANALYSES ═══")
    print("═" * 70)

    # ── Chatbot × symptom interaction ─────────────────────────
    print("\n[4b] [EXPLORATORY] Testing chatbot × symptom interaction (two-way OLS, FDR)...")
    interaction_df = test_chatbot_symptom_interaction(df_paired)
    if not interaction_df.empty:
        sig_int = interaction_df[interaction_df["sig_fdr"]]
        if sig_int.empty:
            print("  No significant chatbot × symptom interactions after FDR correction.")
        else:
            print("  Features with significant chatbot × symptom interaction:")
            for _, row in sig_int.iterrows():
                print(f"    {FEATURE_META[row['feature']][1]:<30}  p_FDR={row['p_fdr']:.4f}  [EXPLORATORY]")

    # ── Semantic similarity ───────────────────────────────────
    sim_df = None
    if use_nlp:
        print("\n[5] Computing semantic similarity (sentence-transformers)...")
        sim_df = compute_semantic_similarity(df_paired)
        sim_df.to_csv(SIM_CSV, index=False)
        print(f"  Saved → {SIM_CSV}")
    else:
        print("\n[5] Skipping semantic similarity (--no-nlp)")

    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("═══ DESCRIPTIVE ANALYSES ═══")
    print("═" * 70)

    # ── Visualisations ────────────────────────────────────────
    print("\n[6] Generating visualisations...")
    plot_feature_means_bar(df_paired)
    plot_boxplots(df_paired, ALL_FEATURES,
                  "All Feature Distributions by Chatbot", "all_features")
    plot_radar(df_paired)
    plot_heatmap_stat(stat_df)
    plot_delta_scores(df_paired)
    plot_feature_correlation(df_paired)
    plot_mean_summary_table(df_paired)
    plot_per_symptom_profiles(df_paired)
    if sim_df is not None and not sim_df.empty:
        plot_semantic_similarity(sim_df)

    # ── Summary report ────────────────────────────────────────
    print_summary(df_paired, stat_df, sim_df)

    print(f"\n✅ Done.  Plots → {PLOT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
