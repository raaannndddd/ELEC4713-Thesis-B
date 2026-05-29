"""
Standalone PERMANOVA + PERMDISP analysis on chatbot response embeddings.

Tests whether demographic group membership (race, gender, age_group) explains
significant variance in response embedding space, within each chatbot.

PERMANOVA (Anderson 2001): pseudo-F on cosine distance matrix, 999 permutations.
PERMDISP (Anderson 2006): tests homogeneity of multivariate dispersions —
  necessary companion to PERMANOVA because a significant PERMANOVA can reflect
  either a difference in centroids *or* a difference in spread.

Results saved to: bias_analysis/permanova_results.csv
"""

import json
import pathlib
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

_ROOT = pathlib.Path(__file__).parent.parent
_DATA = _ROOT / "data" / "web_convo_short.json"
_OUT  = pathlib.Path(__file__).parent / "permanova_results.csv"

N_PERM    = 999
ALPHA     = 0.05
CHATBOTS  = ["doctronic", "drkhan"]
DEMO_VARS = ["race", "gender", "age_group"]

np.random.seed(42)


# ---------------------------------------------------------------------------
# Distance-based test helpers
# ---------------------------------------------------------------------------

def _pseudo_f(sq_dist: np.ndarray, group_idx: np.ndarray) -> float:
    """Compute PERMANOVA pseudo-F from a squared distance matrix."""
    n = len(group_idx)
    k = len(np.unique(group_idx))
    ss_within = sum(
        sq_dist[np.ix_(group_idx == g, group_idx == g)].sum() / (2 * (group_idx == g).sum())
        for g in range(k)
        if (group_idx == g).sum() >= 2
    )
    ss_total = sq_dist.sum() / (2 * n)
    df_b, df_w = k - 1, n - k
    if df_w <= 0 or ss_within == 0:
        return 0.0
    return ((ss_total - ss_within) / df_b) / (ss_within / df_w)


def permanova(dist_mat: np.ndarray, labels: np.ndarray, n_perm: int = N_PERM):
    """
    Returns (pseudo_F, R2, p_value).
    R² = SS_between / SS_total (proportion of variance explained by groups).
    """
    _, group_idx = np.unique(labels, return_inverse=True)
    sq = dist_mat ** 2
    n  = len(labels)
    k  = len(np.unique(labels))

    ss_within = sum(
        sq[np.ix_(group_idx == g, group_idx == g)].sum() / (2 * (group_idx == g).sum())
        for g in range(k)
        if (group_idx == g).sum() >= 2
    )
    ss_total = sq.sum() / (2 * n)
    ss_between = ss_total - ss_within
    df_b, df_w = k - 1, n - k
    obs_f = (ss_between / df_b) / (ss_within / df_w) if (df_w > 0 and ss_within > 0) else 0.0
    r2 = ss_between / ss_total if ss_total > 0 else 0.0

    count = sum(
        _pseudo_f(sq, np.random.permutation(group_idx)) >= obs_f
        for _ in range(n_perm)
    )
    p = (count + 1) / (n_perm + 1)
    return round(obs_f, 4), round(r2, 4), round(p, 4)


def permdisp(dist_mat: np.ndarray, labels: np.ndarray, n_perm: int = N_PERM):
    """
    Permutational test of multivariate dispersions (Anderson 2006 / betadisper).

    Distance of each point to its group centroid is computed in the
    double-centred (Gower) matrix G via the formula:
        d(i, c_g) = sqrt(G_ii  -  2 * mean_j_in_g(G_ij)  +  mean_jl_in_g(G_jl))

    A one-way permutation F-test on these scalar distances tests whether
    groups differ in spread, not just location.

    Returns (F_obs, p_value).
    """
    from scipy.stats import f_oneway

    _, group_idx = np.unique(labels, return_inverse=True)
    k = len(np.unique(group_idx))
    n = len(labels)

    # Double-centre the squared distance matrix (Gower 1966)
    A = -0.5 * dist_mat ** 2
    row_mean   = A.mean(axis=1, keepdims=True)
    col_mean   = A.mean(axis=0, keepdims=True)
    grand_mean = A.mean()
    G = A - row_mean - col_mean + grand_mean

    diag_G = np.diag(G)  # shape (n,)

    def dist_to_centroid(idx):
        d2c = np.zeros(n)
        for g in range(k):
            mask = idx == g
            n_g  = mask.sum()
            if n_g == 0:
                continue
            row_sum_g  = G[np.ix_(mask, mask)].sum(axis=1)
            total_g    = G[np.ix_(mask, mask)].sum()
            d2c[mask]  = np.sqrt(np.maximum(
                diag_G[mask] - 2.0 * row_sum_g / n_g + total_g / n_g**2,
                0.0
            ))
        return d2c

    obs_d  = dist_to_centroid(group_idx)
    groups = [obs_d[group_idx == g] for g in range(k) if (group_idx == g).sum() >= 2]
    if len(groups) < 2:
        return 0.0, 1.0
    obs_f = f_oneway(*groups).statistic

    count = 0
    for _ in range(n_perm):
        perm_idx    = np.random.permutation(group_idx)
        perm_d      = dist_to_centroid(perm_idx)
        perm_groups = [perm_d[perm_idx == g] for g in range(k) if (perm_idx == g).sum() >= 2]
        if len(perm_groups) >= 2 and f_oneway(*perm_groups).statistic >= obs_f:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return round(float(obs_f), 4), round(p, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading responses from JSON ...")
    with open(_DATA) as f:
        records = json.load(f)

    rows = []
    for r in records:
        meta = r.get("metadata", {})
        response = r.get("response", "").strip()
        if not response:
            continue
        age = meta.get("age", None)
        try:
            age_num = int(float(age))
        except (TypeError, ValueError):
            age_num = None
        if age_num is None:
            continue
        rows.append({
            "model":    r["model"],
            "response": response,
            "race":     meta.get("race", "unknown"),
            "gender":   meta.get("gender", "unknown"),
            "age":      age_num,
            "symptom":  meta.get("symptom", "unknown"),
            "severity": meta.get("severity", "unknown"),
        })

    df = pd.DataFrame(rows)
    df = df[df["race"] != "unknown"]
    df = df[df["gender"] != "unknown"]
    df = df[df["model"].isin(CHATBOTS)]

    def age_label(a):
        if a <= 25:   return "Young (20)"
        if a <= 55:   return "Middle (40)"
        return "Older (70)"

    df["age_group"] = df["age"].apply(age_label)

    print(f"  {len(df)} valid responses loaded.")
    print("  By chatbot:", df["model"].value_counts().to_dict())

    print("\nEncoding sentence embeddings (all-MiniLM-L6-v2) ...")
    enc  = SentenceTransformer("all-MiniLM-L6-v2")
    embs = enc.encode(df["response"].tolist(), batch_size=64,
                      show_progress_bar=True, convert_to_numpy=True)
    df["_emb"] = list(embs)

    rows_out = []

    for chatbot in CHATBOTS:
        sub = df[df["model"] == chatbot].reset_index(drop=True)
        sub_embs = np.vstack(sub["_emb"].values)
        dist_mat = cosine_distances(sub_embs)
        n = len(sub)
        print(f"\n{'='*60}")
        print(f"Chatbot: {chatbot}  (n={n})")
        print(f"{'='*60}")

        for demo in DEMO_VARS:
            labels = sub[demo].values
            groups = np.unique(labels)
            counts = {g: (labels == g).sum() for g in groups}
            if len(groups) < 2:
                print(f"  {demo:12s}  — skipped (only 1 group)")
                continue

            print(f"\n  {demo}  groups={list(groups)}")
            print(f"  group sizes: {counts}")

            F, R2, p = permanova(dist_mat, labels)
            print(f"  PERMANOVA  pseudo-F={F:.3f}  R²={R2:.4f}  p={p:.4f}")

            Fd, pd_ = permdisp(dist_mat, labels)
            print(f"  PERMDISP   F={Fd:.3f}  p={pd_:.4f}")

            rows_out.append({
                "chatbot":      chatbot,
                "demographic":  demo,
                "n":            n,
                "n_groups":     len(groups),
                "permanova_F":  F,
                "permanova_R2": R2,
                "permanova_p":  p,
                "permdisp_F":   Fd,
                "permdisp_p":   pd_,
            })

    res = pd.DataFrame(rows_out)
    if len(res):
        _, p_fdr, _, _ = multipletests(res["permanova_p"], method="fdr_bh")
        res["permanova_p_fdr"] = p_fdr.round(4)
        res["permanova_sig_fdr"] = res["permanova_p_fdr"] < ALPHA

        _, pd_fdr, _, _ = multipletests(res["permdisp_p"], method="fdr_bh")
        res["permdisp_p_fdr"] = pd_fdr.round(4)
        res["permdisp_sig_fdr"] = res["permdisp_p_fdr"] < ALPHA

    print(f"\n\n{'='*80}")
    print("FINAL RESULTS (BH-FDR corrected, α = 0.05)")
    print(f"{'='*80}")
    print(f"{'Chatbot':<14}{'Demo':<12}{'n':>5}  {'PERMANOVA':^32}  {'PERMDISP':^24}")
    print(f"{'':14}{'':12}{'':>5}  {'pseudo-F':>9} {'R²':>7} {'p_raw':>7} {'p_FDR':>7} {'sig':>4}  {'F':>7} {'p_FDR':>7} {'sig':>4}")
    print("-" * 90)
    for _, row in res.iterrows():
        sig_perm = "★" if row["permanova_sig_fdr"] else " "
        sig_disp = "★" if row["permdisp_sig_fdr"] else " "
        print(
            f"{row['chatbot']:<14}{row['demographic']:<12}{int(row['n']):>5}  "
            f"{row['permanova_F']:>9.3f} {row['permanova_R2']:>7.4f} "
            f"{row['permanova_p']:>7.4f} {row['permanova_p_fdr']:>7.4f} {sig_perm:>4}  "
            f"{row['permdisp_F']:>7.3f} {row['permdisp_p_fdr']:>7.4f} {sig_disp:>4}"
        )

    res.to_csv(_OUT, index=False)
    print(f"\nSaved: {_OUT}")


if __name__ == "__main__":
    main()
