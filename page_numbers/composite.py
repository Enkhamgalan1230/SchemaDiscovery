from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from schema_discovery.pipeline.run import run_schema_discovery, PipelineConfig
from schema_discovery.candidates.ucc_composite import CompositeUCCConfig, discover_ucc_composite


# ============================================================
# Helpers
# ============================================================
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


def _get_discovery_from_session() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """
    Returns:
      edges_df, profiles_df, ucc_df, composite_ucc_df, source
    """
    # 1) Preferred shared slot
    eo = st.session_state.get("edges_out")
    if isinstance(eo, dict):
        e = eo.get("edges_df")
        p = eo.get("profiles_df")
        u = eo.get("ucc_df")
        c = eo.get("composite_ucc_df")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                (c if isinstance(c, pd.DataFrame) else pd.DataFrame()),
                eo.get("source") or "Edges cache",
            )

    # 2) Full pipeline output from other pages
    out = st.session_state.get("schema_out")
    if isinstance(out, dict):
        e = out.get("edges_df")
        p = out.get("profiles_df")
        u = out.get("ucc_df")
        c = out.get("composite_ucc_df")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                (c if isinstance(c, pd.DataFrame) else pd.DataFrame()),
                "Full Pipeline (Table Relation page)",
            )

    # 3) Profiler components output
    prof_out = st.session_state.get("prof_out")
    if isinstance(prof_out, dict):
        e = prof_out.get("edges_df")
        p = prof_out.get("profiles_df")
        u = prof_out.get("ucc_df")
        c = prof_out.get("composite_ucc_df")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                (c if isinstance(c, pd.DataFrame) else pd.DataFrame()),
                "Extras (Components page)",
            )

    return None, None, None, None, None


@st.cache_data(show_spinner=False)
def _load_dfs(files) -> dict[str, pd.DataFrame]:
    return {f.name.rsplit(".", 1)[0]: pd.read_csv(f) for f in files}


@st.cache_data(show_spinner=False)
def _run_discovery_inputs(
    data_sig: str,
    dfs: dict[str, pd.DataFrame],
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = run_schema_discovery(dfs, config=cfg)

    edges_df = out.get("edges_df")
    if edges_df is None:
        edges_df = pd.DataFrame()

    profiles_df = out.get("profiles_df")
    if profiles_df is None:
        profiles_df = pd.DataFrame()

    ucc_df = out.get("ucc_df")
    if ucc_df is None:
        ucc_df = pd.DataFrame()

    return edges_df, profiles_df, ucc_df


@st.cache_data(show_spinner=False)
def _run_composite_ucc(
    data_sig: str,
    dfs: dict[str, pd.DataFrame],
    profiles_df: pd.DataFrame,
    ucc_unary_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cfg: CompositeUCCConfig,
) -> pd.DataFrame:
    return discover_ucc_composite(
        dfs=dfs,
        profiles_df=profiles_df,
        ucc_unary_df=ucc_unary_df,
        edges_df=edges_df,
        cfg=cfg,
    )


def _download_df(df: pd.DataFrame, base: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"Download {base}.csv",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    with c2:
        st.download_button(
            f"Download {base}.json",
            df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
            type="primary",
        )


def _status_dot(k: int, n_rows_used: int, n_rows_total: int) -> str:
    """
    Green -> strong (uses most rows)
    Yellow -> partial (lots of rows dropped due to null handling when require_no_nulls=False)
    """
    if n_rows_total <= 0:
        return "🟡"
    coverage = n_rows_used / max(n_rows_total, 1)
    if coverage >= 0.98:
        return "🟢"
    if coverage >= 0.80:
        return "🟡"
    return "🟠"


def _combo_duplicates_preview(
    df: pd.DataFrame,
    cols: list[str],
    max_rows: int = 25,
) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    sub = df[cols].copy()
    dup_mask = sub.duplicated(keep=False)
    if not dup_mask.any():
        return pd.DataFrame()
    return df.loc[dup_mask, cols].head(int(max_rows)).copy()


# ============================================================
# Page
# ============================================================
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Composite Key Discovery")
    st.caption(
        "Finds minimal composite unique column combinations (candidate composite keys). "
        "Uses profiler gates to keep the search cheap and avoids measure columns (floats) to prevent fake keys."
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
    # Inputs (profiles, ucc unary, edges)
    # -----------------------------
    with st.container(border=True):
        st.subheader("Inputs")
        st.caption("This page needs profiles_df and unary ucc_df. edges_df is optional but helps prioritise FK-like columns.")

        use_cached = st.toggle(
            "Reuse discovery outputs from Table Relation page (if available)",
            value=False,
        )

        edges_df: pd.DataFrame | None = None
        profiles_df: pd.DataFrame | None = None
        ucc_df: pd.DataFrame | None = None
        composite_cached: pd.DataFrame | None = None

        if use_cached:
            e, p, u, c, source = _get_discovery_from_session()
            if isinstance(p, pd.DataFrame) and not p.empty:
                edges_df, profiles_df, ucc_df, composite_cached = e, p, u, c
                st.success(f"Reusing discovery outputs from: {source}")
            else:
                st.warning("No cached discovery outputs found. Falling back to running discovery here.")
                edges_df = profiles_df = ucc_df = composite_cached = None

        if profiles_df is None or profiles_df.empty or ucc_df is None:
            with st.expander("Schema discovery config", expanded=False):
                st.caption("Only used if we need to compute profiles/ucc/edges on this page.")

                cA, cB, cC = st.columns(3)
                with cA:
                    ind_min_distinct_coverage = st.slider("IND min_distinct_coverage", 0.50, 1.00, 0.90, 0.01)
                    min_edge_score = st.slider("min_edge_score", 0.50, 1.00, 0.90, 0.01)
                with cB:
                    ind_min_non_null_rows = st.slider("IND min_non_null_rows", 1, 500, 20, 1)
                    ind_min_distinct_child = st.slider("IND min_distinct_child", 1, 200, 10, 1)
                with cC:
                    dtype_selected = st.multiselect(
                        "IND dtype families",
                        options=["int", "string", "float", "datetime", "bool", "other"],
                        default=["int", "string"],
                    )
                    profile_mode = st.selectbox("Profile mode", options=["basic", "enhanced"], index=1)

                cfg = PipelineConfig(
                    profile_mode=profile_mode,
                    ind_min_distinct_child=int(ind_min_distinct_child),
                    ind_min_non_null_rows=int(ind_min_non_null_rows),
                    ind_min_distinct_coverage=float(ind_min_distinct_coverage),
                    ind_dtype_families=tuple(dtype_selected),
                    min_edge_score=float(min_edge_score),
                    rel_missing_enabled=False,
                    duplicates_enabled=False,
                    default_fk_enabled=False,
                    surrogate_keys_enabled=False,
                    composite_ucc_enabled=False,  # we compute composite on this page
                )

            if st.button("Run schema discovery to get inputs", type="primary", use_container_width=True):
                with st.spinner("Running schema discovery..."):
                    edges_df, profiles_df, ucc_df = _run_discovery_inputs(data_sig, dfs, cfg)
                    st.session_state["ck_edges_df"] = edges_df
                    st.session_state["ck_profiles_df"] = profiles_df
                    st.session_state["ck_ucc_df"] = ucc_df

            edges_df = st.session_state.get("ck_edges_df") if edges_df is None else edges_df
            profiles_df = st.session_state.get("ck_profiles_df") if profiles_df is None else profiles_df
            ucc_df = st.session_state.get("ck_ucc_df") if ucc_df is None else ucc_df

        if not isinstance(edges_df, pd.DataFrame):
            edges_df = pd.DataFrame()
        if not isinstance(ucc_df, pd.DataFrame):
            ucc_df = pd.DataFrame()
        if not isinstance(profiles_df, pd.DataFrame) or profiles_df.empty:
            st.warning("profiles_df is required to run composite key discovery.")
            st.stop()

        a, b, c = st.columns(3)
        with a:
            st.metric("Profiles rows", int(len(profiles_df)))
        with b:
            st.metric("Unary UCC candidates", int(len(ucc_df)))
        with c:
            st.metric("Relationships", int(len(edges_df)))

    st.divider()

    # -----------------------------
    # Config + Run
    # -----------------------------
    st.subheader("Run composite discovery")

    with st.expander("Composite config", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            max_k = st.slider("max_k", 2, 5, 3, 1)
            max_cols_per_table = st.slider("max_cols_per_table", 2, 60, 12, 1)
        with c2:
            null_ratio_max = st.slider("null_ratio_max", 0.0, 0.50, 0.01, 0.01)
            min_distinct = st.slider("min_distinct", 1, 200, 10, 1)
        with c3:
            max_avg_len_string = st.slider("max_avg_len_string", 5, 200, 40, 5)
            require_no_nulls = st.toggle("require_no_nulls_in_key", value=True)

        exclude_float_cols = st.toggle("exclude_float_cols (recommended)", value=True)

    cfg_ck = CompositeUCCConfig(
        max_k=int(max_k),
        max_cols_per_table=int(max_cols_per_table),
        null_ratio_max=float(null_ratio_max),
        min_distinct=int(min_distinct),
        max_avg_len_string=int(max_avg_len_string),
        require_no_nulls_in_key=bool(require_no_nulls),
        exclude_float_cols=bool(exclude_float_cols),
    )

    # Run control
    run_clicked = st.button("Run composite discovery", type="primary", use_container_width=True)
    if run_clicked:
        st.session_state.pop("ck_out_df", None)

    # Use cached composite if available and config unchanged? Keep it simple -> always compute when needed.
    if st.session_state.get("ck_out_df") is None:
        with st.spinner("Searching composite keys..."):
            out_df = _run_composite_ucc(
                data_sig=data_sig,
                dfs=dfs,
                profiles_df=profiles_df,
                ucc_unary_df=ucc_df,
                edges_df=edges_df,
                cfg=cfg_ck,
            )
            st.session_state["ck_out_df"] = out_df
    else:
        out_df = st.session_state["ck_out_df"]

    st.subheader("Results")

    if out_df is None or out_df.empty:
        st.info("No composite keys found under the current gates. Try increasing max_cols_per_table or relaxing null_ratio_max/min_distinct.")
        st.stop()

    # Add useful computed fields for display
    show = out_df.copy()
    show["k"] = pd.to_numeric(show["k"], errors="coerce").fillna(0).astype(int)

    # Table row counts for coverage indicator
    table_rows = {t: int(len(dfs[t])) for t in dfs.keys()}
    show["table_rows"] = show["table_name"].map(table_rows).fillna(0).astype(int)
    show["coverage"] = (show["n_rows_used"].astype(float) / show["table_rows"].replace(0, pd.NA).astype(float)).fillna(0.0)
    show.insert(
        0,
        "status",
        [
            _status_dot(int(r["k"]), int(r["n_rows_used"]), int(r["table_rows"]))
            for _, r in show.iterrows()
        ],
    )

    # Summary
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Composite candidates", int(len(show)))
    with m2:
        st.metric("Tables with candidates", int(show["table_name"].nunique()))
    with m3:
        st.metric("Max k found", int(show["k"].max() if len(show) else 0))

    st.divider()

    tab1, tab2 = st.tabs(["Table", "Inspect one key" ])

    with tab1:
        f1, f2, f3, f4 = st.columns([1.2, 1.6, 1.4, 1.4])
        with f1:
            table_filter = st.selectbox("Table", options=["all"] + sorted(show["table_name"].unique().tolist()), index=0)
        with f2:
            k_filter = st.selectbox("k", options=["all"] + sorted(show["k"].unique().tolist()), index=0)
        with f3:
            min_cov = st.slider("Min coverage", 0.0, 1.0, 0.0, 0.01)
        with f4:
            top_n = st.number_input("Show top N", min_value=10, max_value=5000, value=100, step=10)

        filtered = show.copy()
        if table_filter != "all":
            filtered = filtered[filtered["table_name"] == table_filter]
        if k_filter != "all":
            filtered = filtered[filtered["k"] == int(k_filter)]
        filtered = filtered[filtered["coverage"] >= float(min_cov)]

        # Expand columns list to display nicely
        filtered = filtered.copy()
        filtered["columns_str"] = filtered["columns"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        view_cols = ["status", "table_name", "k", "n_rows_used", "table_rows", "coverage", "columns_str"]
        view_cols = [c for c in view_cols if c in filtered.columns]
        st.dataframe(filtered[view_cols].head(int(top_n)), width="stretch", height=360)

        _download_df(filtered.drop(columns=["columns_str"], errors="ignore"), base="ucc_composite")

        st.caption("Legend -> 🟢 uses almost all rows, 🟡 partial coverage, 🟠 low coverage")

    with tab2:
        st.caption("Pick a single candidate key and inspect it on the raw table.")

        tmp = show.copy()
        tmp["label"] = tmp["table_name"].astype(str) + " -> (" + tmp["columns"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x)) + ")"

        picked = st.selectbox("Composite key", options=tmp["label"].tolist(), index=0)
        row = tmp[tmp["label"] == picked].iloc[0].to_dict()

        table = str(row["table_name"])
        cols = row["columns"] if isinstance(row["columns"], list) else []
        df = dfs.get(table)

        cA, cB, cC = st.columns([2.2, 1.2, 1.6])
        with cA:
            st.markdown("**Key**")
            st.code(picked, language="text")
        with cB:
            st.markdown("**k**")
            st.write(int(row.get("k", 0)))
        with cC:
            cov = float(row.get("coverage", 0.0))
            st.markdown("**Coverage**")
            st.write(f"{cov:.3f}")

        st.divider()

        if not isinstance(df, pd.DataFrame) or df.empty:
            st.warning("Table not available in dfs.")
            st.stop()

        # Show a quick uniqueness check and any duplicates (should be none for returned candidates)
        sub = df[cols].copy() if all(c in df.columns for c in cols) else pd.DataFrame()
        if sub.empty:
            st.warning("Some columns are missing from the table.")
            st.stop()

        dup_count = int(sub.duplicated(keep=False).sum())
        null_rows = int(sub.isna().any(axis=1).sum())

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Duplicate rows (in key cols)", dup_count)
        with s2:
            st.metric("Rows with any null (key cols)", null_rows)
        with s3:
            st.metric("Table rows", int(len(df)))

        if dup_count > 0:
            st.warning("Duplicates exist for this combo. This should not happen for returned candidates. Check normalisation or columns typing.")
            st.dataframe(_combo_duplicates_preview(df, cols, max_rows=25), width="stretch")

        # Show sample key rows for sanity
        st.markdown("**Sample rows (key columns)**")
        st.dataframe(df[cols].head(25), width="stretch")


    st.divider()
    cL, cR = st.columns(2)
    with cL:
        if st.button("Clear composite outputs", type="primary", use_container_width=True):
            st.session_state.pop("ck_out_df", None)
            st.toast("Cleared composite outputs.", icon="🧹")
    with cR:
        if st.button("Clear Streamlit cache and rerun", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("ck_out_df", None)
            st.toast("Cleared Streamlit cache.", icon="🧼")
            st.rerun()
