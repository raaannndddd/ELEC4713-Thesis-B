"""
chatbot_ml_comparison.py
ML-based comparison of Doctronic and DrKhan chatbot responses.

Assumptions:
- Text = all chatbot turns concatenated per conversation
- Symptom + severity are covariates (appended as binary columns to TF-IDF features)
- Stratified train/test split on chatbot x symptom x severity
- CPU-only (M2 Mac); sentence-transformers will auto-use MPS if available
- SHAP excluded to keep dependencies minimal

The goal is descriptive: quantify how distinguishable chatbot responses are once
direct identity markers are removed. Downstream F1 scores indicate residual
separability, not clinical quality by themselves. Generic phrase-level
boilerplate stripping is disabled because the hand-written phrase list was not
externally validated.
"""

import argparse
import json
import os
import pathlib
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Targeted suppression only — ConvergenceWarning from sklearn classifiers is
# an important signal that a model failed to converge and must NOT be suppressed.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
DATA_PATH = str(_PROJECT_DIR / "data" / "web_convo_short.json")
PLOT_DIR = "bias_analysis/plots"
RANDOM_STATE = 42
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Text cleaning  — remove identity-leaking markers
# ---------------------------------------------------------------------------

# Tier 1: chatbot name references (non-clinical, directly identify the source)
_NAME_RX = re.compile(
    r"\b(dr\.?\s*khan|doctronic)\b",
    re.IGNORECASE,
)

def clean_text(text: str) -> str:
    """
    Remove identity-leaking markers from a chatbot response.

    Removes explicit chatbot name references only.

    Replacements use a single space so surrounding words are not accidentally
    concatenated into new n-grams.
    """
    text = _NAME_RX.sub(" ", text)
    # Collapse multiple whitespace created by removals
    text = re.sub(r"  +", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# 1. Load & extract data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    with open(path) as f:
        convs = json.load(f)

    rows = []
    for c in convs:
        # Older records store chatbot turns in a transcript list; newer records
        # store the full chatbot response directly under "response".
        chatbot_turns = [t["text"] for t in c.get("transcript", []) if t["role"] == "chatbot"]
        raw = " ".join(chatbot_turns) if chatbot_turns else c.get("response", "")
        if not raw.strip():
            continue
        text = clean_text(raw)
        meta = c.get("metadata", {})
        rows.append({
            "model":    c["model"],
            "text":     text,
            "symptom":  meta.get("symptom", "unknown"),
            "severity": meta.get("severity", "unknown"),
        })
    return pd.DataFrame(rows)


def add_covariate_matrix(X_tfidf, df: pd.DataFrame):
    """One-hot encode symptom+severity and append to sparse TF-IDF matrix."""
    cov = pd.get_dummies(df[["symptom", "severity"]], dtype=float)
    cov_sparse = csr_matrix(cov.values)
    return hstack([X_tfidf, cov_sparse])


# ---------------------------------------------------------------------------
# 2. TF-IDF pipeline
# ---------------------------------------------------------------------------

def build_tfidf_features(df: pd.DataFrame, min_df: int = 1, extra_stop_words: list | None = None):
    """
    Build the TF-IDF feature matrix.

    Parameters
    ----------
    min_df : int
        Minimum document frequency threshold.  Terms appearing in fewer than
        this many documents are discarded.  Raising this (e.g. to 5) drops
        chatbot-specific rare n-grams that are unlikely to generalise.
        Default = 1 (no filtering beyond the vectoriser's own max_features).
    extra_stop_words : list | None
        Optional additional stop words to remove on top of regex cleaning.
    """
    stop_words = list(extra_stop_words) if extra_stop_words else None
    vec = TfidfVectorizer(
        max_features=10_000,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=min_df,
        stop_words=stop_words,
    )
    X = vec.fit_transform(df["text"])
    X = add_covariate_matrix(X, df)
    return X, vec


# ---------------------------------------------------------------------------
# 3. Sentence embeddings
# ---------------------------------------------------------------------------

def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    print("  Encoding sentence embeddings (this may take a few minutes)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(df["text"].tolist(), batch_size=32, show_progress_bar=True)
    return embs.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Classifiers
# ---------------------------------------------------------------------------

# Inner classifiers use n_jobs=1 — cross_val_score already parallelises across folds
# with n_jobs=-1, so keeping n_jobs=-1 on inner estimators causes nested parallelism.
CLASSIFIERS_TFIDF = [
    ("LR + TF-IDF",  LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ("SVM + TF-IDF", LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ("RF + TF-IDF",  RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=RANDOM_STATE)),
]

CLASSIFIERS_EMB = [
    ("LR + Emb",  LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ("SVM + Emb", LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ("KNN + Emb", KNeighborsClassifier(n_neighbors=7, n_jobs=1)),
]


def evaluate_classifiers(X, y, classifiers: list, cv: int = 5) -> tuple:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fold_scores = {}
    for name, clf in classifiers:
        scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
        print(f"    {name:25s}  F1-macro = {scores.mean():.3f} ± {scores.std():.3f}")
        results.append({"model": name, "f1_mean": scores.mean(), "f1_std": scores.std()})
        fold_scores[name] = scores
    return pd.DataFrame(results).sort_values("f1_mean", ascending=False), fold_scores


def test_classifier_differences(fold_scores: dict):
    """
    Wilcoxon signed-rank test comparing each classifier against the best on per-fold F1.
    With only 5 folds the minimum achievable p-value is ~0.063, so treat this as
    descriptive ranking support rather than null-hypothesis significance testing.
    """
    from scipy.stats import wilcoxon as _wilcoxon
    best_name = max(fold_scores, key=lambda n: fold_scores[n].mean())
    best_scores = fold_scores[best_name]
    print(f"\n--- Wilcoxon signed-rank vs best ({best_name}) [n_folds=5, min p≈0.063] ---")
    for name, scores in sorted(fold_scores.items(), key=lambda kv: kv[1].mean(), reverse=True):
        if name == best_name:
            continue
        d = best_scores - scores
        if np.all(d == 0):
            print(f"    {name:25s}  identical fold scores")
            continue
        try:
            _res = _wilcoxon(d, alternative="two-sided", zero_method="zsplit")
            stat, p = _res.statistic, _res.pvalue
            print(f"    {name:25s}  W={stat:.1f}  p={p:.4f}  [descriptive only]")
        except Exception as exc:
            print(
                f"    {name:25s}  test inconclusive "
                f"({type(exc).__name__}: {exc})"
            )


# ---------------------------------------------------------------------------
# 5. Visualisations
# ---------------------------------------------------------------------------

def plot_model_comparison(results_df: pd.DataFrame, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#4C72B0" if "TF-IDF" in r else "#DD8452" for r in results_df["model"]]
    ax.barh(results_df["model"], results_df["f1_mean"], xerr=results_df["f1_std"],
            color=colors, capsize=4, edgecolor="white")
    ax.axvline(1/2, color="grey", linestyle="--", linewidth=1, label="Chance (2-class)")
    ax.set_xlabel("CV F1-macro")
    title = "Chatbot Classifier Comparison"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "model_comparison_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(clf, X_test, y_test, labels, label_names=None, title="Best Model"):
    cm = confusion_matrix(y_test, clf.predict(X_test), labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names if label_names is not None else labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_matrix_best.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pca(embs: np.ndarray, labels, label_names):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(embs)
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("tab10", n_colors=len(label_names))
    for i, name in enumerate(label_names):
        mask = np.array(labels) == i
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=0.5, s=18,
                   color=palette[i])
    ax.set_title("PCA of Sentence Embeddings")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "pca_chatbot_embeddings.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_umap(embs: np.ndarray, labels, label_names):
    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed, skipping UMAP plot.")
        return
    print("  Fitting UMAP (may take ~1 min)...")
    reducer = UMAP(n_components=2, random_state=RANDOM_STATE, n_jobs=1)
    coords = reducer.fit_transform(embs)
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("tab10", n_colors=len(label_names))
    for i, name in enumerate(label_names):
        mask = np.array(labels) == i
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=0.5, s=18,
                   color=palette[i])
    ax.set_title("UMAP of Sentence Embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "umap_chatbot_embeddings.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Top vocabulary per chatbot (TF-IDF interpretability)
# ---------------------------------------------------------------------------

def print_top_vocab(vec: TfidfVectorizer, clf, label_names: list, top_n: int = 15):
    if not hasattr(clf, "coef_"):
        print("  Classifier has no coef_ — skipping vocabulary analysis.")
        return
    features = vec.get_feature_names_out()
    # coef_ shape: (n_classes, n_features) for multi-class, or (1, n_features) for binary
    coef = clf.coef_
    # Only covers TF-IDF features (first len(features) columns; rest are covariates)
    n_tfidf = len(features)
    if coef.shape[1] > n_tfidf:
        coef = coef[:, :n_tfidf]

    print("\n--- Top discriminative vocabulary per chatbot ---")
    # Binary LR: coef_ shape (1, n_features) — positive = class[1], negative = class[0]
    if coef.shape[0] == 1:
        c = coef[0]
        top_idx_pos = np.argsort(c)[::-1][:top_n]
        top_idx_neg = np.argsort(c)[:top_n]
        print(f"\n  {label_names[1]} (positive coefficients):")
        print("  " + ", ".join(features[top_idx_pos]))
        print(f"\n  {label_names[0]} (negative coefficients):")
        print("  " + ", ".join(features[top_idx_neg]))
    else:
        for i, name in enumerate(label_names):
            c = coef[i]
            top_idx = np.argsort(c)[::-1][:top_n]
            top_words = features[top_idx]
            print(f"\n  {name}:")
            print("  " + ", ".join(top_words))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ML-based chatbot comparison with configurable text debiasing."
    )
    p.add_argument(
        "--min-df", type=int, default=1, metavar="N",
        help="Minimum document frequency for TF-IDF terms (default: 1). "
             "Raise to e.g. 5 to drop rare chatbot-specific n-grams.",
    )
    p.add_argument(
        "--data-path", type=str, default=DATA_PATH, metavar="PATH",
        help=f"Path to web_conversations JSON file (default: {DATA_PATH}).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Chatbot ML Comparison")
    print("=" * 60)

    # Summarise active cleaning options
    cleaning_notes = ["regex chatbot-name removal only"]
    if args.min_df > 1:
        cleaning_notes.append(f"min_df={args.min_df}")
    print("\nActive text cleaning:")
    for note in cleaning_notes:
        print(f"  • {note}")

    # --- Load ---
    print("\n[1] Loading data...")
    df = load_data(args.data_path)
    try:
        from bias_analysis.analysis_constants import SHORT_CHATBOTS
    except ImportError:
        from analysis_constants import SHORT_CHATBOTS
    df = df[df["model"].isin(SHORT_CHATBOTS)].copy()
    df = df[(df["symptom"] != "unknown") & (df["severity"] != "unknown")].copy()
    print(f"  {len(df)} valid conversations: {dict(df['model'].value_counts())}")

    # Audit chatbot × symptom × severity coverage before appending covariates.
    # Leakage occurs when one chatbot has a (symptom, severity) combo that another
    # chatbot lacks, because the one-hot pattern then partially identifies the label.
    # Symmetric gaps (all chatbots missing the same combos) produce all-zero columns
    # and don't distinguish chatbots.
    combo_sets = {
        model: set(zip(sub["symptom"], sub["severity"]))
        for model, sub in df.groupby("model")
    }
    all_combos = set().union(*combo_sets.values())
    asymmetric = [
        combo for combo in all_combos
        if not all(combo in s for s in combo_sets.values())
    ]
    if asymmetric:
        raise SystemExit(
            "[FATAL] Covariate leakage risk detected: chatbot × symptom × severity "
            f"coverage has {len(asymmetric)} asymmetric missing combo(s) — some chatbots "
            "have data for a combo that others lack, so covariates would partially reveal "
            "the label. Aborting instead of reporting leaked F1 scores."
        )
    else:
        n_combos = len(all_combos)
        print(
            f"  Symmetric chatbot coverage confirmed ({n_combos} shared symptom×severity combos). "
            "Covariate leakage is reduced but not eliminated by design."
        )

    le = LabelEncoder()
    y = le.fit_transform(df["model"])
    label_names = list(le.classes_)
    print(f"  Classes: {label_names}")

    # Stratify key: chatbot + symptom + severity
    strat_key = df["model"] + "|" + df["symptom"] + "|" + df["severity"]

    # --- TF-IDF features ---
    print("\n[2] Building TF-IDF features (with symptom/severity covariates)...")
    X_tfidf, vec = build_tfidf_features(df, min_df=args.min_df)
    print(f"  Feature matrix: {X_tfidf.shape}")

    # --- Sentence embeddings ---
    print("\n[3] Building sentence embeddings...")
    embs = build_embeddings(df)
    print(f"  Embedding matrix: {embs.shape}")

    # Append covariates to embeddings too
    cov_df = pd.get_dummies(df[["symptom", "severity"]], dtype=float)
    X_emb = np.hstack([embs, cov_df.values])
    print(f"  Embedding+cov matrix: {X_emb.shape}")

    # --- Cross-validation ---
    print("\n[4] Cross-validating classifiers (5-fold, stratified by chatbot×symptom×severity)...")
    print("  TF-IDF models:")
    results_tfidf, fold_scores_tfidf = evaluate_classifiers(X_tfidf, y, CLASSIFIERS_TFIDF)
    print("  Embedding models:")
    results_emb, fold_scores_emb = evaluate_classifiers(X_emb, y, CLASSIFIERS_EMB)

    all_results = pd.concat([results_tfidf, results_emb], ignore_index=True).sort_values("f1_mean", ascending=False)
    print("\n--- Model Ranking ---")
    print(all_results.to_string(index=False))
    test_classifier_differences({**fold_scores_tfidf, **fold_scores_emb})

    # Build a plot subtitle that reflects the active cleaning mode
    cleaning_tag = "regex-only"
    parts = []
    if args.min_df > 1:
        parts.append(f"min_df={args.min_df}")
    if parts:
        cleaning_tag = "regex + " + " ".join(parts)
    plot_model_comparison(all_results, subtitle=f"Cleaning: {cleaning_tag}")

    # --- Best model: full train/test evaluation ---
    print("\n[5] Full train/test evaluation of best model...")
    X_train_tf, X_test_tf, y_train, y_test, idx_train, idx_test = train_test_split(
        X_tfidf, y, df.index, test_size=0.2, stratify=strat_key, random_state=RANDOM_STATE
    )
    X_train_emb, X_test_emb = X_emb[df.index.isin(idx_train)], X_emb[df.index.isin(idx_test)]

    best_name = all_results.iloc[0]["model"]
    print(f"  Best model: {best_name}")

    # Pick the actual clf + feature set
    all_clfs = dict(CLASSIFIERS_TFIDF + CLASSIFIERS_EMB)
    best_clf = all_clfs[best_name]
    X_train_best = X_train_tf if "TF-IDF" in best_name else X_train_emb
    X_test_best  = X_test_tf  if "TF-IDF" in best_name else X_test_emb

    best_clf.fit(X_train_best, y_train)
    y_pred = best_clf.predict(X_test_best)
    print("\n" + classification_report(y_test, y_pred, target_names=label_names))

    plot_confusion_matrix(best_clf, X_test_best, y_test,
                          labels=list(range(len(label_names))),
                          label_names=label_names,
                          title=best_name)

    # --- Top vocabulary (from best LR or SVM on TF-IDF) ---
    print("\n[6] Top vocabulary analysis (using LR + TF-IDF)...")
    lr_tfidf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    lr_tfidf.fit(X_train_tf, y_train)
    print_top_vocab(vec, lr_tfidf, label_names)

    # --- Visualisations ---
    print("\n[7] Generating embedding plots...")
    plot_pca(embs, y, label_names)
    plot_umap(embs, y, label_names)

    print("\nDone. Plots saved to:", PLOT_DIR)


if __name__ == "__main__":
    main()
