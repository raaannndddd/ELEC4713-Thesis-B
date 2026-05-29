from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

try:
    from bias_analysis.feature_registry import FEATURE_REGISTRY
except ImportError:
    from feature_registry import FEATURE_REGISTRY


class SchemaValidationError(ValueError):
    pass


def _as(kind, obj, ctx):
    if not isinstance(obj, kind):
        raise SchemaValidationError(f"{ctx} must be {kind.__name__}, got {type(obj).__name__}")
    return obj


def validate_records(raw: object, *, kind: str) -> list[dict]:
    rows = _as(list, raw, f"{kind} dataset")
    for i, row in enumerate(rows):
        row = _as(dict, row, f"{kind} record {i}")
        if "model" not in row:
            raise SchemaValidationError(f"{kind} record {i} missing 'model'")
        if "metadata" in row:
            meta = _as(dict, row["metadata"], f"{kind} record {i}.metadata")
            if kind == "long" and "age" in meta:
                try:
                    int(meta["age"])
                except (TypeError, ValueError) as exc:
                    raise SchemaValidationError(f"long record {i}.metadata.age must be int-like") from exc
        if kind == "long" and "transcript" in row:
            for j, turn in enumerate(_as(list, row["transcript"], f"long record {i}.transcript")):
                _as(dict, turn, f"long record {i}.transcript[{j}]")
    return rows


def validate_feature_frame(
    df: pd.DataFrame,
    *,
    feature_names: Iterable[str],
    required_columns: Iterable[str],
    context: str,
) -> None:
    required_columns = list(required_columns)
    feature_names = list(feature_names)
    missing = [c for c in [*required_columns, *feature_names] if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"{context} missing columns: {missing}")
    if df[required_columns].isna().any().any():
        bad = df.columns[df.isna().any()].intersection(required_columns).tolist()
        raise SchemaValidationError(f"{context} has NaN in required columns: {bad}")
    if df[list(feature_names)].isna().any().any():
        bad = df.columns[df.isna().any()].intersection(feature_names).tolist()
        raise SchemaValidationError(f"{context} has NaN in: {bad}")
    for name in feature_names:
        spec = FEATURE_REGISTRY.get(name)
        if spec is None:
            raise SchemaValidationError(f"{context} unknown feature '{name}'")
        lo, hi = spec["min"], spec["max"]
        if lo is not None and (df[name] < lo).any():
            raise SchemaValidationError(f"{context} has values below {lo} for '{name}'")
        if hi is not None and (df[name] > hi).any():
            raise SchemaValidationError(f"{context} has values above {hi} for '{name}'")
