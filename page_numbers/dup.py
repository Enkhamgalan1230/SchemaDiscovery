from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from schema_discovery.quality.duplicates import (
    DuplicateChecksConfig,
    run_duplicate_checks,
)

# -----------------------------
# Helpers
# -----------------------------
NESTED_COLS_DEFAULT = {
    "example_rows",
    "example_edges",
    "example_subset_rows",
    "example_key_to_distinct_count",
}

def _df_signature(dfs: dict[str, pd.DataFrame]) -> str:
    """Cheap signature for cache invalidation."""
    h = hashlib.sha256()
    for name in sorted(dfs.keys()):
        df = dfs[name]
        h.update(name.encode("utf-8"))
        h.update(str(df.shape[0]).encode("utf-8"))
        h.update(str(df.shape[1]).encode("utf-8"))
        sample = df.head(30).to_csv(index=False).encode("utf-8", errors="ignore")
        h.update(hashlib.sha256(sample).digest())
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _load_dfs(files) -> dict[str, pd.DataFrame]:
    return {f.name.rsplit(".", 1)[0]: pd.read_csv(f) for f in files}


def _stringify_nested_for_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit's dataframe renderer shows nested objects as [object Object].
    This converts list/dict cells into pretty JSON strings for display.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    out = df.copy()
    for col in out.columns:
        # Only stringify columns that actually contain nested objects
        if out[col].apply(lambda x: isinstance(x, (list, dict))).any():
            out[col] = out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False, default=str)
                if isinstance(x, (list, dict))
                else x
            )
    return out


def _download_df(df_raw: pd.DataFrame, base: str) -> None:
    """
    CSV: nested columns are stringified to avoid unreadable "[{'a':1}]" blobs
    JSON: preserves nested objects cleanly (records of dicts/lists)
    """
    c1, c2 = st.columns(2)

    df_for_csv = _stringify_nested_for_table(df_raw)

    with c1:
        st.download_button(
            f"Download {base}.csv",
            df_for_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"{base}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        records = df_raw.to_dict(orient="records") if isinstance(df_raw, pd.DataFrame) else []
        st.download_button(
            f"Download {base}.json",
            json.dumps(records, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
        )


@st.cache_data(show_spinner=False)
def _run_dups_cached(
    data_sig: str,
    cfg_dict: dict[str, Any],
    dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Cached duplicate checks.
    cfg_dict is hashable-friendly and avoids caching issues with dataclass instances.
    """
    cfg = DuplicateChecksConfig(**cfg_dict)
    out = run_duplicate_checks(dfs, cfg=cfg)

    # Guarantee all values are DataFrames
    for k, v in list(out.items()):
        if v is None or not isinstance(v, pd.DataFrame):
            out[k] = pd.DataFrame()
    return out


def _out_to_jsonable(out: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Convert the module output (dict of DataFrames) into a JSON-friendly dict
    so you can show the "full JSON result" on the page.
    """
    items: dict[str, Any] = {}
    for k, df in out.items():
        if isinstance(df, pd.DataFrame):
            items[k] = df.to_dict(orient="records")
        else:
            items[k] = []
    summary = {k: len(v) for k, v in items.items()}
    return {"summary": summary, "items": items}


# -----------------------------
# Page
# -----------------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Duplicate Detection")
    st.caption(
        "Runs duplicate and overlap diagnostics: key duplicates, natural key duplicates, cross table overlaps, "
        "identifier conflicts, and duplicate rows."
    )

    # -----------------------------
    # Data
    # -----------------------------
    with st.container(border=True):
        st.subheader("Data")
        use_existing = st.toggle(
            "Use tables already uploaded in Data Schematics page (recommended)",
            value=False,
        )

        dfs: dict[str, pd.DataFrame] | None = None

        if use_existing and isinstance(st.session_state.get("dfs_uploaded"), dict) and st.session_state["dfs_uploaded"]:
            dfs = st.session_state["dfs_uploaded"]
            st.success(f"Using {len(dfs)} table(s) from session.")
        else:
            files = st.file_uploader("Drop CSV files here", type=["csv"], accept_multiple_files=True)
            if not files:
                st.info("Upload CSVs to continue, or enable reuse from the Table Relation page.")
                st.stop()
            dfs = _load_dfs(files)
            st.session_state["dfs_uploaded"] = dfs
            st.success(f"Loaded {len(dfs)} table(s).")

        preview_table = st.selectbox("Preview table", options=list(dfs.keys()))
        st.dataframe(dfs[preview_table].head(10), width="stretch")

    data_sig = _df_signature(dfs)
    st.divider()

    # -----------------------------
    # Parameters
    # -----------------------------
    with st.expander("Duplicate detection config", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            key_min_non_null_rows = st.slider("key_min_non_null_rows", 1, 500, 20, 1)
            key_min_unique_ratio = st.slider("key_min_unique_ratio", 0.0, 1.0, 0.98, 0.01)
            key_id_like_only = st.toggle("key_id_like_only", value=True)

        with c2:
            overlap_min_ratio_smaller_set = st.slider("overlap_min_ratio_smaller_set", 0.0, 1.0, 0.10, 0.01)
            max_examples = st.slider("max_examples", 1, 50, 10, 1)

        with c3:
            natural_key_min_unique_ratio = st.slider("natural_key_min_unique_ratio", 0.0, 1.0, 0.70, 0.01)
            signature_unique_ratio_max = st.slider("signature_unique_ratio_max", 0.0, 1.0, 0.95, 0.01)
            signature_min_cols = st.slider("signature_min_cols", 1, 5, 2, 1)

        st.caption(
            "Subset key duplicates and relationship table rules are supported, but keep them simple for now. "
            "Add them later when you have real datasets."
        )

    cfg_dict = dict(
        key_min_non_null_rows=int(key_min_non_null_rows),
        key_min_unique_ratio=float(key_min_unique_ratio),
        key_id_like_only=bool(key_id_like_only),
        natural_key_min_unique_ratio=float(natural_key_min_unique_ratio),
        overlap_min_ratio_smaller_set=float(overlap_min_ratio_smaller_set),
        max_examples=int(max_examples),
        signature_unique_ratio_max=float(signature_unique_ratio_max),
        signature_min_cols=int(signature_min_cols),

        # keep advanced parts empty by default
        relational_duplicate_subset_keys={},
        relationship_tables={},
    )

    # -----------------------------
    # Run
    # -----------------------------
    cL, cR = st.columns([1, 1])
    with cL:
        if st.button("Run duplicate checks", type="primary", use_container_width=True):
            st.session_state.pop("dup_out", None)

    with cR:
        if st.button("Clear outputs", use_container_width=True):
            st.session_state.pop("dup_out", None)
            st.toast("Cleared duplicate outputs.", icon="🧹")

    if st.session_state.get("dup_out") is None:
        with st.spinner("Running duplicate checks..."):
            out = _run_dups_cached(data_sig=data_sig, cfg_dict=cfg_dict, dfs=dfs)
            st.session_state["dup_out"] = out
            st.session_state["dup_out_source"] = "Duplicate Detection page"
    else:
        out = st.session_state["dup_out"]

    # -----------------------------
    # Overview
    # -----------------------------
    st.subheader("Overview")
    keys = list(out.keys())
    counts = {k: (0 if out[k] is None else int(len(out[k]))) for k in keys}
    st.json(counts)

    # -----------------------------
    # Full JSON result (module-like output)
    # -----------------------------
    with st.expander("Full JSON result (all sections)", expanded=False):
        jsonable = _out_to_jsonable(out)
        st.json(jsonable)
        st.download_button(
            "Download full_result.json",
            data=json.dumps(jsonable, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="duplicates_full_result.json",
            mime="application/json",
            use_container_width=True,
            type="primary",
        )

    st.divider()

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        [
            "A1 Key duplicates",
            "A2 Natural key duplicates",
            "B1 Overlap same name",
            "B2 Overlap any name",
            "C ID reuse within table",
            "D Same signature multiple IDs",
            "D Summary",
            "E1 Exact duplicate rows",
            "E2 Subset duplicates",
            "F Relationship table issues",
        ]
    )

    def show_df(tab, df_key: str, download_base: str, caption: str) -> None:
        with tab:
            df_raw = out.get(df_key)
            if not isinstance(df_raw, pd.DataFrame):
                df_raw = pd.DataFrame()

            st.caption(caption)
            if df_raw.empty:
                st.info("No issues found.")
                return

            # FIX: make nested objects readable in the table
            df_show = _stringify_nested_for_table(df_raw)
            st.dataframe(df_show, width="stretch", height=420)

            # Also show nested content as real JSON for a few rows (so you can inspect properly)
            nested_cols = [c for c in df_raw.columns if df_raw[c].apply(lambda x: isinstance(x, (list, dict))).any()]
            if nested_cols:
                with st.expander("Expanded nested fields (first 5 rows)", expanded=False):
                    for i in range(min(5, len(df_raw))):
                        st.markdown(f"**Row {i+1}**")
                        st.json({c: df_raw.iloc[i][c] for c in nested_cols})

            _download_df(df_raw, download_base)

    show_df(
        tab1,
        "duplicate_keys",
        "duplicate_keys",
        "Duplicate values inside key-like columns (PK-like columns that should be unique).",
    )
    show_df(
        tab2,
        "duplicate_natural_keys",
        "duplicate_natural_keys",
        "Duplicate values inside natural-key columns (email, phone, username), based on name patterns.",
    )
    show_df(
        tab3,
        "cross_table_overlap_same_name",
        "cross_table_overlap_same_name",
        "Cross table overlap when the same column name appears across tables.",
    )
    show_df(
        tab4,
        "cross_table_overlap_id_or_natural_any_name",
        "cross_table_overlap_id_or_natural_any_name",
        "Cross table overlap for ID-like and natural-key columns even if column names differ.",
    )
    show_df(
        tab5,
        "within_table_id_reuse",
        "within_table_id_reuse",
        "Checks if the same identifier values appear in multiple ID-like columns within the same table.",
    )
    show_df(
        tab6,
        "same_signature_multiple_ids",
        "same_signature_multiple_ids",
        "Collapsed report: same natural key signature mapping to multiple IDs.",
    )
    show_df(
        tab7,
        "same_signature_multiple_ids_summary",
        "same_signature_multiple_ids_summary",
        "Per table summary of signature -> multiple IDs issues.",
    )
    show_df(
        tab8,
        "exact_duplicate_rows",
        "exact_duplicate_rows",
        "Exact row duplicates across all columns.",
    )
    show_df(
        tab9,
        "duplicate_rows_on_subsets",
        "duplicate_rows_on_subsets",
        "Duplicate rows on configured subset keys (business key duplicates). Empty unless configured.",
    )
    show_df(
        tab10,
        "relationship_table_issues",
        "relationship_table_issues",
        "Issues in mapping tables based on configured expected cardinality. Empty unless configured.",
    )

    st.divider()
    if st.button("Clear Streamlit cache and rerun", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("dup_out", None)
        st.toast("Cleared Streamlit cache.", icon="🧼")
        st.rerun()
