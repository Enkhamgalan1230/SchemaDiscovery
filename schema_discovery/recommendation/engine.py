from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .schema import SchemaRecommendations, TableRecommendations, to_dict
from .scorers.pk import recommend_primary_key
from .scorers.composite import recommend_composite_keys
from .scorers.fk import recommend_foreign_keys
from .scorers.datatype import recommend_datatypes
from .scorers.index import recommend_indexes


def _get_df(out: dict[str, Any], key: str) -> pd.DataFrame:
    """
    Safe getter for DataFrames inside `out`.
    Returns an empty DataFrame if:
      - key is missing
      - value is None
      - value is not a DataFrame
    """
    v = out.get(key)
    return v if isinstance(v, pd.DataFrame) else pd.DataFrame()


def build_schema_recommendations(
    *,
    dfs: dict[str, pd.DataFrame],
    out: dict[str, Any],
    normalisation_policy: Optional[dict[str, Any]] = None,
    top_n_alternatives: int = 3,
) -> dict[str, Any]:
    """
    Orchestrator. This does not rediscover anything.
    It ranks + packages recommendations using pipeline outputs in `out`.
    """

    # Never use `df or pd.DataFrame()` -> DataFrames cannot be used in boolean context.
    profiles_df = _get_df(out, "profiles_df")
    ucc_df = _get_df(out, "ucc_df")
    composite_ucc_df = _get_df(out, "composite_ucc_df")
    edges_df = _get_df(out, "edges_df")
    rel_missing_df = _get_df(out, "rel_missing_df")
    default_fk_df = _get_df(out, "default_fk_df")

    # Dict-like stage outputs are fine to use with `or {}`.
    surrogate_keys_out: dict[str, Any] = out.get("surrogate_keys", {}) or {}
    dup_out: dict[str, pd.DataFrame] = out.get("duplicates", {}) or {}
    profile_extras: dict[str, Any] = out.get("profile_extras", {}) or {}

    norm_policy = normalisation_policy or {
        "null_tokens_mapped": ["", "null", "none", "n/a", "na"],
        "trim_whitespace": True,
        "casefold_strings": True,
        "remove_punctuation": False,
    }

    tables: list[TableRecommendations] = []
    for table_name, df in dfs.items():
        row_count = int(len(df))

        pk_rec = recommend_primary_key(
            table_name=table_name,
            profiles_df=profiles_df,
            ucc_df=ucc_df,
            composite_ucc_df=composite_ucc_df,
            edges_df=edges_df,
            surrogate_keys_out=surrogate_keys_out,
            dup_out=dup_out,
            top_n_alternatives=top_n_alternatives,
        )

        composite_recs = recommend_composite_keys(
            table_name=table_name,
            composite_ucc_df=composite_ucc_df,
            top_n_alternatives=top_n_alternatives,
        )

        fk_recs = recommend_foreign_keys(
            table_name=table_name,
            edges_df=edges_df,
            rel_missing_df=rel_missing_df,
            default_fk_df=default_fk_df,
            top_n_alternatives=top_n_alternatives,
        )

        dtype_recs = recommend_datatypes(
            table_name=table_name,
            df=df,  # add this
            profiles_df=profiles_df,
            profile_extras=profile_extras,
            top_n_alternatives=top_n_alternatives,
        )

        idx_recs = recommend_indexes(
            table_name=table_name,
            pk_rec=pk_rec,
            fk_recs=fk_recs,
            composite_recs=composite_recs,
            profiles_df=profiles_df,
        )

        table_bundle = TableRecommendations(
            table_name=table_name,
            row_count=row_count,
            structure={
                "primary_key": pk_rec,
                "composite_keys": composite_recs,
                "foreign_keys": fk_recs,
                "surrogate_key": None,  # optional later, if you want a separate rec
            },
            constraints={
                "recommended": [],
                "blocked": [],
            },
            datatypes={"columns": dtype_recs},
            performance=idx_recs,
            notes=[],
        )
        tables.append(table_bundle)

    root = SchemaRecommendations(
        meta={
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "pipeline_version": "0.1.0",
            "source": "csv_upload",
        },
        normalisation_policy=norm_policy,
        tables=tables,
        global_warnings=[],
    )

    return to_dict(root)
