from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from schema_discovery.pipeline.run import run_schema_discovery, PipelineConfig
from schema_discovery.quality.relational_missing import (
    RelMissingConfig,
    run_rel_missing_for_edge,
    rel_missing_summary_to_df,
)

# -----------------------------
# Helpers
# -----------------------------
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

def _get_edges_from_session() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    # 0) Shared edges slot (recommended)
    eo = st.session_state.get("edges_out")
    if isinstance(eo, dict):
        e = eo.get("edges_df")
        p = eo.get("profiles_df")
        if isinstance(e, pd.DataFrame) and not e.empty:
            return e, (p if isinstance(p, pd.DataFrame) else None), eo.get("source") or "Edges cache"

    # 1) Full pipeline output (table_relation.py)
    out = st.session_state.get("schema_out")
    if isinstance(out, dict):
        e = out.get("edges_df")
        p = out.get("profiles_df")
        if isinstance(e, pd.DataFrame) and not e.empty:
            return e, (p if isinstance(p, pd.DataFrame) else None), "Full Pipeline (Table Relation page)"

    # 2) Extras components output (extras.py)
    prof_out = st.session_state.get("prof_out")
    if isinstance(prof_out, dict):
        e = prof_out.get("edges_df")
        p = prof_out.get("profiles_df")
        if isinstance(e, pd.DataFrame) and not e.empty:
            return e, (p if isinstance(p, pd.DataFrame) else None), "Extras (Components page)"

    return None, None, None



@st.cache_data(show_spinner=False)
def _load_dfs(files) -> dict[str, pd.DataFrame]:
    return {f.name.rsplit(".", 1)[0]: pd.read_csv(f) for f in files}


@st.cache_data(show_spinner=False)
def _run_edges(data_sig: str, dfs: dict[str, pd.DataFrame], cfg: PipelineConfig):
    out = run_schema_discovery(dfs, config=cfg)

    edges_df = out.get("edges_df")
    if edges_df is None:
        edges_df = pd.DataFrame()

    profiles_df = out.get("profiles_df")
    if profiles_df is None:
        profiles_df = pd.DataFrame()

    return edges_df, profiles_df

@st.cache_data(show_spinner=False)
def _run_rel_missing_all(
    data_sig: str,
    edges_records: tuple[tuple[tuple[str, Any], ...], ...],
    dfs: dict[str, pd.DataFrame],
    profiles_df: pd.DataFrame,
    cfg_rm: RelMissingConfig,
    sample_n: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Cached relational missing run.
    edges_records is a hashable representation of edges to avoid caching issues.
    """
    # reconstruct list[dict] from hashable tuples
    edges_list: list[dict[str, Any]] = []
    for rec in edges_records:
        edges_list.append({k: v for k, v in rec})

    items: list[dict[str, Any]] = []
    for edge in edges_list:
        items.append(
            run_rel_missing_for_edge(
                dfs=dfs,
                edge_row=edge,
                profiles_df=profiles_df,
                cfg=cfg_rm,
                sample_n=sample_n,
            )
        )

    summary_df = rel_missing_summary_to_df(items)
    return items, summary_df


def _to_hashable_edges(edges_df: pd.DataFrame) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Convert edges_df rows into a hashable structure for caching."""
    if edges_df is None or edges_df.empty:
        return tuple()
    records = edges_df.to_dict(orient="records")
    out = []
    for r in records:
        # sort keys for stable hashing
        out.append(tuple(sorted(r.items(), key=lambda x: x[0])))
    return tuple(out)


def _download_df(df: pd.DataFrame, base: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"Download {base}.csv",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base}.csv",
            mime="text/csv",
            use_container_width=True,  # download_button does not yet support width consistently
        )
    with c2:
        st.download_button(
            f"Download {base}.json",
            df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
        )


# -----------------------------
# Page
# -----------------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Relational Missingness")
    st.caption(
        "Checks discovered relationships (FK -> PK) for unreferenced foreign keys and parent keys, "
        "then assigns an explainable status and severity."
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
    # Where edges come from
    # -----------------------------
    with st.container(border=True):
        st.subheader("Relationships to check (edges)")

        use_existing_edges = st.toggle(
            "Use relationships already discovered in Table Relation page (if available)",
            value=False,
            help="If you previously ran Schema Discovery on the Table Relation page, reuse its edges_df.",
        )

        edges_df = None
        profiles_df = None

        if use_existing_edges:
            e, p, source = _get_edges_from_session()
            if isinstance(e, pd.DataFrame) and not e.empty:
                edges_df, profiles_df = e, p
                st.success(f"Reusing {len(edges_df)} relationship(s) from: {source}")
            else:
                st.warning("No cached relationships found in session. Falling back to running discovery here.")
                edges_df = None
                profiles_df = None

        if edges_df is None:

            with st.expander("Schema discovery config", expanded=False):
                st.caption("Discovery settings (only used if we need to run schema discovery on this page).")

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
                    profile_mode = st.selectbox("Profile mode", options=["basic", "enhanced"], index=0)

                cfg = PipelineConfig(
                    profile_mode=profile_mode,  # basic is enough for rel-missing, enhanced optional
                    ind_min_distinct_child=int(ind_min_distinct_child),
                    ind_min_non_null_rows=int(ind_min_non_null_rows),
                    ind_min_distinct_coverage=float(ind_min_distinct_coverage),
                    ind_dtype_families=tuple(dtype_selected),
                    min_edge_score=float(min_edge_score),
                    rel_missing_enabled=False,  # we run rel-missing ourselves here
                    duplicates_enabled=False,
                )

            if st.button("Run schema discovery to get edges", type="primary", use_container_width=True):
                with st.spinner("Running schema discovery..."):
                    edges_df, profiles_df = _run_edges(data_sig, dfs, cfg)
                    st.session_state["rm_edges_df"] = edges_df
                    st.session_state["rm_profiles_df"] = profiles_df

            # also allow cached local result on page
            if edges_df is None:
                edges_df = st.session_state.get("rm_edges_df")
            if profiles_df is None:
                profiles_df = st.session_state.get("rm_profiles_df")


            if not isinstance(edges_df, pd.DataFrame):
                edges_df = None
            if not isinstance(profiles_df, pd.DataFrame):
                profiles_df = None


        if edges_df is None or edges_df.empty:
            st.warning("No relationships available to check.")
            st.stop()

        st.dataframe(edges_df, width="stretch", height=260)

    st.divider()

    # -----------------------------
    # Run checks
    # -----------------------------
    st.subheader("Run relational missingness checks")

    # -----------------------------
    # Rel-missing config
    # -----------------------------
    with st.expander("Relational missingness config", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            orphan_warn = st.slider("orphan_warn_ratio", 0.0, 0.20, 0.01, 0.001)
            orphan_fail = st.slider("orphan_fail_ratio", 0.0, 0.50, 0.05, 0.005)
        with c2:
            missing_warn = st.slider("missing_warn_ratio", 0.0, 0.50, 0.05, 0.005)
            missing_fail = st.slider("missing_fail_ratio", 0.0, 1.00, 0.20, 0.01)
        with c3:
            required_multiplier = st.slider("required_multiplier", 0.5, 5.0, 1.0, 0.1)
            sample_n = st.slider("sample_n (examples)", 1, 50, 10, 1)

        st.caption("Sentinel detection is built-in (unknown, n/a, -1, 9999 etc).")

    cfg_rm = RelMissingConfig(
        orphan_warn_ratio=float(orphan_warn),
        orphan_fail_ratio=float(orphan_fail),
        missing_warn_ratio=float(missing_warn),
        missing_fail_ratio=float(missing_fail),
        required_multiplier=float(required_multiplier),
    )

    # Let user choose subset of edges to run (important for big graphs)
    edge_labels = [
        f"{r['fk_table']}.{r['fk_column']} -> {r['pk_table']}.{r['pk_column']}"
        for r in edges_df.to_dict(orient="records")
    ]
    picked = st.multiselect(
        "Select relationships to check",
        options=edge_labels,
        default=edge_labels,
        help="If you have many edges, select a subset to run faster.",
    )

    # Filter edges_df to selected
    if picked:
        pick_set = set(picked)
        edges_sel = edges_df[
            edges_df.apply(lambda r: f"{r['fk_table']}.{r['fk_column']} -> {r['pk_table']}.{r['pk_column']}" in pick_set, axis=1)
        ].copy()
    else:
        edges_sel = edges_df.iloc[0:0].copy()

    if edges_sel.empty:
        st.info("Select at least one relationship to run checks.")
        st.stop()

    if profiles_df is None:
        profiles_df = pd.DataFrame()

    edges_records = _to_hashable_edges(edges_sel)

    if st.button("Run checks", type="primary", use_container_width=True):
        st.session_state.pop("rm_out", None)

    if st.session_state.get("rm_out") is None:
        with st.spinner("Checking relationships..."):
            items, summary_df = _run_rel_missing_all(
                data_sig=data_sig,
                edges_records=edges_records,
                dfs=dfs,
                profiles_df=profiles_df,
                cfg_rm=cfg_rm,
                sample_n=int(sample_n),
            )
            st.session_state["rm_out"] = (items, summary_df)
    else:
        items, summary_df = st.session_state["rm_out"]

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Relationship details")
    st.caption("Expand each relationship to see examples, row indices (for orphan FKs), and recommended actions.")

    for item in items:
        edge_str = item.get("edge", "edge")
        cls = item.get("classification", {}) or {}
        sev = cls.get("severity", "unknown")
        status = cls.get("status", "unknown")

        with st.expander(f"{edge_str}  |  status={status}  severity={sev}", expanded=False):
            checks = item.get("checks", {}) or {}
            orphan = (checks.get("orphan_fk") or {})
            missing = (checks.get("missing_children") or {})

            cA, cB, cC = st.columns(3)
            with cA:
                st.markdown("**Orphan FK**")
                st.write(
                    {
                        "orphan_fk_rows": orphan.get("orphan_fk_rows"),
                        "orphan_fk_ratio": orphan.get("orphan_fk_ratio"),
                        "fk_null_ratio": orphan.get("fk_null_ratio"),
                    }
                )
                st.write("Examples:", orphan.get("orphan_fk_examples", []))
                idx = orphan.get("orphan_child_row_indices", [])
                if idx:
                    st.write("Child row indices (sample):", idx)

            with cB:
                st.markdown("**Missing children**")
                st.write(
                    {
                        "parents_missing_children": missing.get("parents_missing_children"),
                        "parents_missing_children_ratio": missing.get("parents_missing_children_ratio"),
                    }
                )
                st.write("Parent key examples:", missing.get("parent_pk_examples", []))

            with cC:
                st.markdown("**Classification**")
                st.write(
                    {
                        "status": cls.get("status"),
                        "severity": cls.get("severity"),
                    }
                )
                st.write("Likely causes:", cls.get("likely_causes", []))
                st.write("Notes:", cls.get("notes", []))
                st.write("Recommended actions:", cls.get("recommended_actions", []))

            with st.expander("Raw JSON for this relationship"):
                st.json(item)

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        if st.button("Clear rel-missing outputs", use_container_width=True):
            st.session_state.pop("rm_out", None)
            st.toast("Cleared relational missingness outputs.", icon="🧹")
    with cR:
        if st.button("Clear Streamlit cache and rerun", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("rm_out", None)
            st.toast("Cleared Streamlit cache.", icon="🧼")
            st.rerun()
