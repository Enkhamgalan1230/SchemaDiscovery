from __future__ import annotations

import hashlib
from typing import Any, Optional

import pandas as pd
import streamlit as st

from schema_discovery.pipeline.run import run_schema_discovery, PipelineConfig
from schema_discovery.quality.default_fk import (
    DefaultFKConfig,
    run_default_fk_for_edges,
    default_fk_summary_to_df,
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
def _run_edges(data_sig: str, dfs: dict[str, pd.DataFrame], cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = run_schema_discovery(dfs, config=cfg)

    edges_df = out.get("edges_df")
    if edges_df is None:
        edges_df = pd.DataFrame()

    profiles_df = out.get("profiles_df")
    if profiles_df is None:
        profiles_df = pd.DataFrame()

    return edges_df, profiles_df


def _to_hashable_edges(edges_df: pd.DataFrame) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Convert edges_df rows into a hashable structure for caching."""
    if edges_df is None or edges_df.empty:
        return tuple()
    records = edges_df.to_dict(orient="records")
    out = []
    for r in records:
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
            use_container_width=True,
            type = "primary"
        )
    with c2:
        st.download_button(
            f"Download {base}.json",
            df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
            type = "primary"
        )


@st.cache_data(show_spinner=False)
def _run_default_fk_selected(
    data_sig: str,
    edges_records: tuple[tuple[tuple[str, Any], ...], ...],
    dfs: dict[str, pd.DataFrame],
    cfg_fk: DefaultFKConfig,
    sample_n: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    # reconstruct list[dict] from hashable tuples
    edges_list: list[dict[str, Any]] = []
    for rec in edges_records:
        edges_list.append({k: v for k, v in rec})

    edges_df = pd.DataFrame(edges_list)
    items = run_default_fk_for_edges(dfs=dfs, edges_df=edges_df, cfg=cfg_fk, sample_n=int(sample_n))
    summary_df = default_fk_summary_to_df(items)
    return items, summary_df


def _edge_label(r: pd.Series) -> str:
    return f"{r['fk_table']}.{r['fk_column']} -> {r['pk_table']}.{r['pk_column']}"


# -----------------------------
# Page
# -----------------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Default FK Values")
    st.caption(
        "Detects suspicious foreign key values that behave like placeholders "
        "(e.g., 0, -1, 'unknown', 'n/a') or default buckets."
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
                    profile_mode=profile_mode,
                    ind_min_distinct_child=int(ind_min_distinct_child),
                    ind_min_non_null_rows=int(ind_min_non_null_rows),
                    ind_min_distinct_coverage=float(ind_min_distinct_coverage),
                    ind_dtype_families=tuple(dtype_selected),
                    min_edge_score=float(min_edge_score),
                    rel_missing_enabled=False,
                    duplicates_enabled=False,
                )

            if st.button("Run schema discovery to get edges", type="primary", use_container_width=True):
                with st.spinner("Running schema discovery..."):
                    edges_df, profiles_df = _run_edges(data_sig, dfs, cfg)
                    st.session_state["dfk_edges_df"] = edges_df
                    st.session_state["dfk_profiles_df"] = profiles_df

            if edges_df is None:
                edges_df = st.session_state.get("dfk_edges_df")
            if profiles_df is None:
                profiles_df = st.session_state.get("dfk_profiles_df")

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
    st.subheader("Run default FK checks")

    # -----------------------------
    # Default FK config
    # -----------------------------
    with st.expander("Default FK detection config", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            top_k = st.slider("top_k (most frequent FK values to inspect)", 1, 20, 3, 1)
            min_ratio = st.slider("min_ratio (freq threshold)", 0.0, 1.0, 0.20, 0.01)
            strong_ratio = st.slider("strong_ratio (dominant bucket)", 0.0, 1.0, 0.50, 0.01)

        with c2:
            small_domain_max_distinct = st.slider("small_domain_max_distinct", 1, 50, 5, 1)
            min_non_null_rows = st.slider("min_non_null_rows", 1, 2000, 20, 1)
            sample_n = st.slider("sample_n (examples)", 1, 50, 10, 1)

        with c3:
            enable_pairing = st.toggle("enable_pairing_variance_check", value=True)
            partner_id_like_only = st.toggle("partner_id_like_only", value=True)
            min_partner_overall_distinct = st.slider("min_partner_overall_distinct", 1, 500, 20, 1)
            min_partner_distinct_for_flag = st.slider("min_partner_distinct_for_flag", 1, 200, 10, 1)
            min_partner_distinct_ratio = st.slider("min_partner_distinct_ratio", 0.0, 1.0, 0.20, 0.01)

        st.caption("Sentinel lists are built-in (unknown, n/a, null, -1, 0, 9999 etc).")

    cfg_fk = DefaultFKConfig(
        top_k=int(top_k),
        min_ratio=float(min_ratio),
        strong_ratio=float(strong_ratio),
        small_domain_max_distinct=int(small_domain_max_distinct),
        min_non_null_rows=int(min_non_null_rows),
        enable_pairing_variance_check=bool(enable_pairing),
        partner_id_like_only=bool(partner_id_like_only),
        min_partner_overall_distinct=int(min_partner_overall_distinct),
        min_partner_distinct_for_flag=int(min_partner_distinct_for_flag),
        min_partner_distinct_ratio=float(min_partner_distinct_ratio),
    )

    edge_labels = [_edge_label(r) for _, r in edges_df.iterrows()]
    picked = st.multiselect(
        "Select relationships to check",
        options=edge_labels,
        default=edge_labels,
        help="If you have many edges, select a subset to run faster.",
    )

    if picked:
        pick_set = set(picked)
        edges_sel = edges_df[edges_df.apply(lambda r: _edge_label(r) in pick_set, axis=1)].copy()
    else:
        edges_sel = edges_df.iloc[0:0].copy()

    if edges_sel.empty:
        st.info("Select at least one relationship to run checks.")
        st.stop()

    edges_records = _to_hashable_edges(edges_sel)

    # reset output when user clicks
    if st.button("Run checks", type="primary", use_container_width=True):
        st.session_state.pop("dfk_out", None)

    if st.session_state.get("dfk_out") is None:
        with st.spinner("Checking default FK candidates..."):
            items, summary_df = _run_default_fk_selected(
                data_sig=data_sig,
                edges_records=edges_records,
                dfs=dfs,
                cfg_fk=cfg_fk,
                sample_n=int(sample_n),
            )
            st.session_state["dfk_out"] = (items, summary_df)
    else:
        items, summary_df = st.session_state["dfk_out"]

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Summary")
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        st.dataframe(summary_df, width="stretch", height=260)
        _download_df(summary_df, base="default_fk_summary")
    else:
        st.info("No default FK candidates detected in the selected relationships.")

    st.divider()

    st.subheader("Relationship details")
    st.caption("Expand each relationship to see flagged values, reasons, and evidence (pairing variance).")

    for item in items:
        rel = item.get("relationship", "relationship")
        detected = bool(item.get("default_detected", False))
        fk_rows = item.get("fk_non_null_rows", None)
        fk_distinct = item.get("fk_distinct", None)

        # Keep expander title clean: relationship only
        with st.expander(rel, expanded=False):

            # -----------------------------
            # Header block (clean + scannable)
            # -----------------------------
            h1, h2, h3 = st.columns([3, 1.5, 1.5])

            with h1:
                st.markdown("**Relationship**")
                st.code(rel, language="text")

            with h2:
                st.markdown("**Status**")
                if detected:
                    st.warning("Defaults detected", icon="⚠️")
                else:
                    st.success("No defaults", icon="✅")

            with h3:
                st.markdown("**FK stats**")
                st.write(
                    {
                        "non_null_rows": fk_rows,
                        "distinct_values": fk_distinct,
                    }
                )

            st.divider()

            # -----------------------------
            # Candidates + Evidence
            # -----------------------------
            cands = item.get("candidates", []) or []

            cA, cB = st.columns([2, 1])

            with cA:
                st.markdown("**Candidates**")
                if cands:
                    rows = []
                    for c in cands:
                        rows.append(
                            {
                                "value": c.get("value"),
                                "count": c.get("count"),
                                "ratio": c.get("ratio"),
                                "in_parent_pk_domain": c.get("in_parent_pk_domain"),
                                "reasons": ", ".join([str(x) for x in (c.get("reasons") or [])]),
                            }
                        )
                    df_cands = pd.DataFrame(rows)

                    # Nice ordering if present
                    preferred = ["ratio", "count", "value", "in_parent_pk_domain", "reasons"]
                    cols = [c for c in preferred if c in df_cands.columns] + [c for c in df_cands.columns if c not in preferred]
                    df_cands = df_cands[cols]

                    st.dataframe(df_cands, width="stretch", hide_index=True)
                else:
                    st.write("No candidates flagged.")

            with cB:
                st.markdown("**Top evidence (if present)**")

                # default to first candidate which should be the strongest one
                top = cands[0] if cands and isinstance(cands[0], dict) else None
                ev = (top.get("evidence") or {}) if top else {}

                if ev:
                    st.write(
                        {
                            "best_partner_column": ev.get("best_partner_column"),
                            "distinct_partners_under_value": ev.get("distinct_partners_under_value"),
                            "partner_distinct_ratio": ev.get("partner_distinct_ratio"),
                        }
                    )
                    ex = ev.get("partner_examples", [])
                    if ex:
                        st.write("partner_examples:", ex)
                else:
                    st.write("No pairing variance evidence for top candidate.")

            # -----------------------------
            # Raw JSON
            # -----------------------------
            with st.expander("Raw JSON for this relationship"):
                st.json(item)


    st.divider()
    cL, cR = st.columns(2)
    with cL:
        if st.button("Clear default-FK outputs", type = "primary", use_container_width=True):
            st.session_state.pop("dfk_out", None)
            st.toast("Cleared default FK outputs.", icon="🧹")
    with cR:
        if st.button("Clear Streamlit cache and rerun", type = "primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("dfk_out", None)
            st.toast("Cleared Streamlit cache.", icon="🧼")
            st.rerun()
