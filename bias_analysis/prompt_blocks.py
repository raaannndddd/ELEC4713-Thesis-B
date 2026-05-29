"""Shared helpers for matched prompt-block construction and validation."""

from __future__ import annotations

import pandas as pd


def make_prompt_key(row: pd.Series) -> tuple:
    """Canonical key identifying a shared prompt condition across chatbots."""
    return (row["symptom"], row["severity"], row["gender"], row["race"], row["age"])


def attach_complete_prompt_blocks(
    df: pd.DataFrame,
    chatbots: list[str],
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Attach prompt keys/ids and keep only conditions with one row per chatbot."""
    df = df.copy()
    required = ["model", "symptom", "severity", "gender", "race", "age"]
    if df[required].isna().any().any():
        bad = df.columns[df.isna().any()].intersection(required).tolist()
        raise ValueError(
            "attach_complete_prompt_blocks requires non-null prompt-defining "
            f"columns; found NaN in: {bad}"
        )
    df["prompt_key"] = df.apply(make_prompt_key, axis=1)

    duplicate_counts = df.groupby(["prompt_key", "model"]).size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if not duplicates.empty:
        sample = duplicates.head(5).to_dict()
        raise ValueError(
            "Duplicate rows detected for prompt_key/model pairs; matched analyses "
            f"require unique cells. Example counts: {sample}"
        )

    model_sets = df.groupby("prompt_key")["model"].apply(set)
    required_set = set(chatbots)
    complete_keys = model_sets[model_sets.apply(required_set.issubset)].index
    incomplete_keys = model_sets[~model_sets.apply(required_set.issubset)].index

    df_complete = df[df["prompt_key"].isin(complete_keys)].copy()
    key_to_id = {key: i for i, key in enumerate(sorted(complete_keys))}
    df_complete["prompt_id"] = df_complete["prompt_key"].map(key_to_id)
    if df_complete["prompt_id"].isna().any() or df_complete["model"].isna().any():
        raise ValueError(
            "attach_complete_prompt_blocks produced null prompt_id/model values, "
            "which would break matched analyses."
        )

    if verbose:
        print("\n  [Prompt Block Validation]")
        print(f"    Total prompt conditions : {len(model_sets)}")
        print(f"    Complete pairs          : {len(complete_keys)}")
        print(f"    Incomplete (excluded)   : {len(incomplete_keys)}")
        if len(incomplete_keys) > 0:
            print(
                f"    WARNING: {len(incomplete_keys)} prompt conditions excluded "
                "due to missing chatbot responses"
            )

    return df_complete.reset_index(drop=True)
