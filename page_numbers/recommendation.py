from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
import streamlit as st

from schema_discovery.pipeline.run import run_schema_discovery, PipelineConfig

from schema_discovery.recommendation.engine import build_schema_recommendations


# =============================================================================
# Helpers: config + conversion + safe calling
# =============================================================================
def _make_pipeline_config(**kwargs):
    """Create PipelineConfig, dropping any kwargs not supported by current PipelineConfig version."""
    sig = inspect.signature(PipelineConfig)
    allowed = set(sig.parameters.keys())
    clean = {k: v for k, v in kwargs.items() if k in allowed}
    return PipelineConfig(**clean)


def _to_plain(obj: Any) -> Any:
    """Convert dataclass/pydantic objects to plain dicts for JSON and downstream processing."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # pydantic v1
        return obj.dict()
    return obj


def _json_sanitize(x: Any) -> Any:
    """
    Ensure Streamlit/PyArrow-safe types only:
    dict, list, str, int, float, bool, None
    """
    if is_dataclass(x):
        return asdict(x)
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()

    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_sanitize(v) for v in x]
    return x


def _safe_df(x: Any) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()


def _table_picklist(dfs: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(list(dfs.keys()))


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


@st.cache_data(show_spinner=False)
def _run_discovery_minimal(data_sig: str, dfs: dict[str, pd.DataFrame], cfg: PipelineConfig) -> dict[str, Any]:
    # data_sig is intentionally unused but included for cache-keying
    return run_schema_discovery(dfs, config=cfg)


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


def _download_json(obj: Any, base: str) -> None:
    blob = json.dumps(obj, indent=2, default=str).encode("utf-8")
    st.download_button(
        f"Download {base}.json",
        blob,
        file_name=f"{base}.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )


def _get_discovery_from_session() -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict | None,
    dict | None,
    str | None,
]:
    """
    Returns:
      edges_df, profiles_df, ucc_df, composite_ucc_df, profile_extras, duplicates_out, source
    """
    eo = st.session_state.get("edges_out")
    if isinstance(eo, dict):
        e = eo.get("edges_df")
        p = eo.get("profiles_df")
        u = eo.get("ucc_df")
        cu = eo.get("composite_ucc_df")
        px = eo.get("profile_extras")
        du = eo.get("duplicates")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                (cu if isinstance(cu, pd.DataFrame) else pd.DataFrame()),
                (px if isinstance(px, dict) else {}),
                (du if isinstance(du, dict) else {}),
                eo.get("source") or "Edges cache",
            )

    out = st.session_state.get("schema_out")
    if isinstance(out, dict):
        e = out.get("edges_df")
        p = out.get("profiles_df")
        u = out.get("ucc_df")
        cu = out.get("composite_ucc_df")
        px = out.get("profile_extras")
        du = out.get("duplicates")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                (cu if isinstance(cu, pd.DataFrame) else pd.DataFrame()),
                (px if isinstance(px, dict) else {}),
                (du if isinstance(du, dict) else {}),
                "Full Pipeline (Table Relation page)",
            )

    prof_out = st.session_state.get("prof_out")
    if isinstance(prof_out, dict):
        e = prof_out.get("edges_df")
        p = prof_out.get("profiles_df")
        u = prof_out.get("ucc_df")
        cu = prof_out.get("composite_ucc_df")
        px = prof_out.get("profile_extras")
        du = prof_out.get("duplicates")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                (cu if isinstance(cu, pd.DataFrame) else pd.DataFrame()),
                (px if isinstance(px, dict) else {}),
                (du if isinstance(du, dict) else {}),
                "Extras (Components page)",
            )

    return None, None, None, None, None, None, None


# =============================================================================
# Helpers: recommendation output parsing
# =============================================================================
def _get_table_blocks(out_obj: Any) -> list[dict[str, Any]]:
    """
    Engine output shape -> {"tables": [ {table_name, structure:{...}, datatypes:{...}, performance:{...}} ]}
    """
    if not isinstance(out_obj, dict):
        return []
    tables = out_obj.get("tables")
    if isinstance(tables, list):
        return [t for t in tables if isinstance(t, dict)]
    return []


def _is_engine(out_obj: Any) -> bool:
    return isinstance(out_obj, dict) and isinstance(out_obj.get("tables"), list)


def _norm_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def _norm_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _pick_block(blocks: list[dict[str, Any]], table_name: str) -> dict[str, Any] | None:
    return next((b for b in blocks if b.get("table_name") == table_name), None)


def _df_from_records(records: Any) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        return pd.DataFrame()
    records = _json_sanitize(records)
    try:
        return pd.json_normalize(records)
    except Exception:
        return pd.DataFrame(records)


def _section_json(block: dict[str, Any], *, is_engine: bool, section: str) -> Any:
    """
    section in {"pk","fk","ck","dt","idx"}
    Returns raw section object as stored in engine output.
    """
    if not is_engine:
        return {}

    if section == "pk":
        return _norm_dict(_norm_dict(block.get("structure")).get("primary_key"))
    if section == "fk":
        return _norm_list(_norm_dict(block.get("structure")).get("foreign_keys"))
    if section == "ck":
        return _norm_list(_norm_dict(block.get("structure")).get("composite_keys"))
    if section == "dt":
        return _norm_list(_norm_dict(block.get("datatypes")).get("columns"))
    if section == "idx":
        perf = _norm_dict(block.get("performance"))
        return {
            "indexes_recommended": _norm_list(perf.get("indexes_recommended")),
            "indexes_avoid": _norm_list(perf.get("indexes_avoid")),
        }
    return {}


def _compact_pk_rows(blocks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        tn = b.get("table_name")
        pk = _section_json(b, is_engine=True, section="pk")
        payload = _norm_dict(_norm_dict(pk).get("payload"))
        cols = _norm_list(payload.get("columns"))
        rows.append(
            {
                "table": tn,
                "pk_columns": ", ".join([str(x) for x in cols]),
                "status": _norm_dict(pk).get("status"),
                "confidence": _norm_dict(pk).get("confidence"),
                "reasons": ", ".join([str(x) for x in _norm_list(_norm_dict(pk).get("reason_codes"))]),
            }
        )
    return pd.DataFrame(rows)


def _compact_fk_rows(blocks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        tn = b.get("table_name")
        fks = _section_json(b, is_engine=True, section="fk")
        if not isinstance(fks, list):
            continue
        for fk in _json_sanitize(fks):
            fk = _norm_dict(fk)
            payload = _norm_dict(fk.get("payload"))
            refs = _norm_dict(payload.get("references"))
            rows.append(
                {
                    "table": tn,
                    "fk_columns": ", ".join([str(x) for x in _norm_list(payload.get("columns"))]),
                    "ref_table": refs.get("table"),
                    "ref_columns": ", ".join([str(x) for x in _norm_list(refs.get("columns"))]),
                    "status": fk.get("status"),
                    "confidence": fk.get("confidence"),
                    "reasons": ", ".join([str(x) for x in _norm_list(fk.get("reason_codes"))]),
                    "blockers": ", ".join(
                        [str(x.get("code")) for x in _norm_list(fk.get("blockers")) if isinstance(x, dict)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _compact_ck_rows(blocks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        tn = b.get("table_name")
        cks = _section_json(b, is_engine=True, section="ck")
        if not isinstance(cks, list):
            continue
        for ck in _json_sanitize(cks):
            ck = _norm_dict(ck)
            payload = _norm_dict(ck.get("payload"))
            rows.append(
                {
                    "table": tn,
                    "columns": ", ".join([str(x) for x in _norm_list(payload.get("columns"))]),
                    "status": ck.get("status"),
                    "confidence": ck.get("confidence"),
                    "reasons": ", ".join([str(x) for x in _norm_list(ck.get("reason_codes"))]),
                }
            )
    return pd.DataFrame(rows)


def _compact_dt_rows(blocks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        tn = b.get("table_name")
        dts = _section_json(b, is_engine=True, section="dt")
        if not isinstance(dts, list):
            continue
        for dt in _json_sanitize(dts):
            dt = _norm_dict(dt)
            payload = _norm_dict(dt.get("payload"))
            rows.append(
                {
                    "table": tn,
                    "column": payload.get("column"),
                    "db_type": payload.get("db_type"),
                    "status": dt.get("status"),
                    "confidence": dt.get("confidence"),
                    "reasons": ", ".join([str(x) for x in _norm_list(dt.get("reason_codes"))]),
                    "blockers": ", ".join(
                        [str(x.get("code")) for x in _norm_list(dt.get("blockers")) if isinstance(x, dict)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _compact_idx_rows(blocks: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec_rows = []
    avoid_rows = []
    for b in blocks:
        tn = b.get("table_name")
        idx = _section_json(b, is_engine=True, section="idx")
        rec_list = _norm_list(_norm_dict(idx).get("indexes_recommended"))
        avoid_list = _norm_list(_norm_dict(idx).get("indexes_avoid"))

        for r in _json_sanitize(rec_list):
            r = _norm_dict(r)
            payload = _norm_dict(r.get("payload"))
            rec_rows.append(
                {
                    "table": tn,
                    "columns": ", ".join([str(x) for x in _norm_list(payload.get("columns"))]),
                    "priority": payload.get("priority"),
                    "name_suggestion": payload.get("name_suggestion"),
                    "confidence": r.get("confidence"),
                    "reasons": ", ".join([str(x) for x in _norm_list(r.get("reason_codes"))]),
                }
            )

        for r in _json_sanitize(avoid_list):
            r = _norm_dict(r)
            payload = _norm_dict(r.get("payload"))
            avoid_rows.append(
                {
                    "table": tn,
                    "columns": ", ".join([str(x) for x in _norm_list(payload.get("columns"))]),
                    "reason": payload.get("reason"),
                    "confidence": r.get("confidence"),
                    "reasons": ", ".join([str(x) for x in _norm_list(r.get("reason_codes"))]),
                }
            )

    return pd.DataFrame(rec_rows), pd.DataFrame(avoid_rows)


def _section_panel(
    *,
    title: str,
    caption: str,
    selected_df: pd.DataFrame,
    selected_raw: Any,
    selected_download_base: str,
    compact_df: pd.DataFrame,
    compact_download_base: str,
) -> None:
    st.subheader(title)
    st.caption(caption)

    st.markdown("**Selected table**")
    if selected_df is None or selected_df.empty:
        st.info("No rows for this section.")
    else:
        st.dataframe(selected_df, width="stretch", height=360, hide_index=True)
        _download_df(selected_df, selected_download_base)

    with st.expander("Raw JSON for selected table", expanded=False):
        st.json(_json_sanitize(selected_raw))
        _download_json(_json_sanitize(selected_raw), f"{selected_download_base}_raw")

    st.divider()
    st.markdown("**All tables (compact view)**")
    if compact_df is None or compact_df.empty:
        st.info("No rows in compact view.")
    else:
        st.dataframe(compact_df, width="stretch", height=320, hide_index=True)
        _download_df(compact_df, compact_download_base)


def render_pk_tab(blocks: list[dict[str, Any]], out_obj: dict[str, Any], table_name: str) -> None:
    if not blocks:
        st.info("No output available yet. Click Run.")
        return
    if not table_name:
        st.info("Pick a table to view.")
        return

    block = _pick_block(blocks, table_name)
    if block is None:
        st.info("Pick a table to view.")
        return

    pk = _json_sanitize(_section_json(block, is_engine=True, section="pk"))
    selected_df = _df_from_records([pk]) if isinstance(pk, dict) and pk else pd.DataFrame()
    compact_df = _compact_pk_rows(blocks)

    _section_panel(
        title=f"{table_name}.primary_key",
        caption="Primary key recommendation for the selected table.",
        selected_df=selected_df,
        selected_raw=pk,
        selected_download_base=f"pk_{table_name}",
        compact_df=compact_df,
        compact_download_base="pk_all_tables",
    )


def render_fk_tab(blocks: list[dict[str, Any]], out_obj: dict[str, Any], table_name: str) -> None:
    if not blocks:
        st.info("No output available yet. Click Run.")
        return
    if not table_name:
        st.info("Pick a table to view.")
        return

    block = _pick_block(blocks, table_name)
    if block is None:
        st.info("Pick a table to view.")
        return

    fks = _json_sanitize(_section_json(block, is_engine=True, section="fk"))
    selected_df = _df_from_records(fks)
    compact_df = _compact_fk_rows(blocks)

    _section_panel(
        title=f"{table_name}.foreign_keys",
        caption="Foreign key recommendations for the selected table.",
        selected_df=selected_df,
        selected_raw=fks,
        selected_download_base=f"fk_{table_name}",
        compact_df=compact_df,
        compact_download_base="fk_all_tables",
    )


def render_ck_tab(blocks: list[dict[str, Any]], out_obj: dict[str, Any], table_name: str) -> None:
    if not blocks:
        st.info("No output available yet. Click Run.")
        return
    if not table_name:
        st.info("Pick a table to view.")
        return

    block = _pick_block(blocks, table_name)
    if block is None:
        st.info("Pick a table to view.")
        return

    cks = _json_sanitize(_section_json(block, is_engine=True, section="ck"))
    selected_df = _df_from_records(cks)
    compact_df = _compact_ck_rows(blocks)

    _section_panel(
        title=f"{table_name}.composite_keys",
        caption="Composite key recommendations for the selected table.",
        selected_df=selected_df,
        selected_raw=cks,
        selected_download_base=f"ck_{table_name}",
        compact_df=compact_df,
        compact_download_base="ck_all_tables",
    )


def render_dt_tab(blocks: list[dict[str, Any]], out_obj: dict[str, Any], table_name: str) -> None:
    if not blocks:
        st.info("No output available yet. Click Run.")
        return
    if not table_name:
        st.info("Pick a table to view.")
        return

    block = _pick_block(blocks, table_name)
    if block is None:
        st.info("Pick a table to view.")
        return

    dts = _json_sanitize(_section_json(block, is_engine=True, section="dt"))
    selected_df = _df_from_records(dts)
    compact_df = _compact_dt_rows(blocks)

    _section_panel(
        title=f"{table_name}.datatypes",
        caption="Datatype recommendations for the selected table.",
        selected_df=selected_df,
        selected_raw=dts,
        selected_download_base=f"datatypes_{table_name}",
        compact_df=compact_df,
        compact_download_base="datatypes_all_tables",
    )


def render_idx_tab(blocks: list[dict[str, Any]], out_obj: dict[str, Any], table_name: str) -> None:
    if not blocks:
        st.info("No output available yet. Click Run.")
        return
    if not table_name:
        st.info("Pick a table to view.")
        return

    block = _pick_block(blocks, table_name)
    if block is None:
        st.info("Pick a table to view.")
        return

    idx = _json_sanitize(_section_json(block, is_engine=True, section="idx"))
    rec = _norm_list(_norm_dict(idx).get("indexes_recommended"))
    avoid = _norm_list(_norm_dict(idx).get("indexes_avoid"))

    selected_df_rec = _df_from_records(rec)
    selected_df_avoid = _df_from_records(avoid)

    compact_rec, compact_avoid = _compact_idx_rows(blocks)

    st.subheader(f"{table_name}.indexes")
    st.caption("Index recommendations and indexes to avoid for the selected table.")

    st.markdown("**Selected table -> indexes recommended**")
    if selected_df_rec.empty:
        st.info("No index recommendations for this table.")
    else:
        st.dataframe(selected_df_rec, width="stretch", height=280, hide_index=True)
        _download_df(selected_df_rec, f"idx_recommended_{table_name}")

    with st.expander("Raw JSON -> indexes recommended", expanded=False):
        st.json(_json_sanitize(rec))
        _download_json(_json_sanitize(rec), f"idx_recommended_{table_name}_raw")

    st.divider()
    st.markdown("**Selected table -> indexes to avoid**")
    if selected_df_avoid.empty:
        st.info("No indexes to avoid for this table.")
    else:
        st.dataframe(selected_df_avoid, width="stretch", height=280, hide_index=True)
        _download_df(selected_df_avoid, f"idx_avoid_{table_name}")

    with st.expander("Raw JSON -> indexes to avoid", expanded=False):
        st.json(_json_sanitize(avoid))
        _download_json(_json_sanitize(avoid), f"idx_avoid_{table_name}_raw")

    st.divider()
    st.markdown("**All tables (compact view) -> indexes recommended**")
    if compact_rec.empty:
        st.info("No rows in compact view.")
    else:
        st.dataframe(compact_rec, width="stretch", height=300, hide_index=True)
        _download_df(compact_rec, "idx_recommended_all_tables")

    st.divider()
    st.markdown("**All tables (compact view) -> indexes to avoid**")
    if compact_avoid.empty:
        st.info("No rows in compact view.")
    else:
        st.dataframe(compact_avoid, width="stretch", height=300, hide_index=True)
        _download_df(compact_avoid, "idx_avoid_all_tables")


def render_all_tab(out_obj: dict[str, Any]) -> None:
    st.caption("Raw recommendations JSON. Useful for debugging and export.")
    with st.expander("Raw JSON (full output)", expanded=True):
        st.json(_json_sanitize(out_obj))
        _download_json(_json_sanitize(out_obj), "schema_recommendations")


# =============================================================================
# Page
# =============================================================================
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Schema Recommendations")
    st.caption("Runs Perfomance, Structural, Constraint, Datatype recommendation modules.")
    st.caption("💡Heuristic suggestions, not constraints. Validate before enforcing in schema.")
    # -----------------------------
    # Data
    # -----------------------------
    with st.container(border=True):
        st.subheader("Data")

        use_existing = st.toggle(
            "Use tables already uploaded in Data Schematics page (recommended)",
            value=False,
            key="reco_use_existing",
        )

        dfs: dict[str, pd.DataFrame] | None = None
        if use_existing and isinstance(st.session_state.get("dfs_uploaded"), dict) and st.session_state["dfs_uploaded"]:
            dfs = st.session_state["dfs_uploaded"]
            st.success(f"Using {len(dfs)} table(s) from session.")
        else:
            files = st.file_uploader("Drop CSV files here", type=["csv"], accept_multiple_files=True, key="reco_files")
            if not files:
                st.info("Upload CSVs to continue.")
                st.stop()
            dfs = _load_dfs(files)
            st.session_state["dfs_uploaded"] = dfs
            st.success(f"Loaded {len(dfs)} table(s).")

        preview_table = st.selectbox("Preview table", options=_table_picklist(dfs), key="reco_preview_table")
        st.dataframe(dfs[preview_table].head(10), width="stretch", hide_index=True)

    data_sig = _df_signature(dfs)
    st.divider()

    # -----------------------------
    # Inputs (profiles/ucc/edges etc.)
    # -----------------------------
    with st.container(border=True):
        st.subheader("Inputs")

        use_cached = st.toggle(
            "Reuse discovery outputs from other pages (if available)",
            value=False,
            help="Looks for edges_df, profiles_df, ucc_df, composite_ucc_df, profile_extras, duplicates in session.",
            key="reco_use_cached",
        )

        edges_df: pd.DataFrame | None = None
        profiles_df: pd.DataFrame | None = None
        ucc_df: pd.DataFrame | None = None
        composite_ucc_df: pd.DataFrame | None = None
        profile_extras: dict | None = None
        duplicates_out: dict | None = None

        if use_cached:
            e, p, u, cu, px, du, source = _get_discovery_from_session()
            if isinstance(p, pd.DataFrame) and not p.empty:
                edges_df = e
                profiles_df = p
                ucc_df = u
                composite_ucc_df = cu
                profile_extras = px
                duplicates_out = du
                st.success(f"Reusing discovery outputs from: {source}")
            else:
                st.warning("No cached discovery outputs found. Run discovery here.")
                edges_df = profiles_df = ucc_df = composite_ucc_df = None
                profile_extras = {}
                duplicates_out = {}

        cfg = None
        if profiles_df is None or profiles_df.empty:
            with st.expander("Schema discovery config (used only if needed)", expanded=False):
                st.caption("Runs the pipeline to compute profiles, UCCs, composite UCCs, edges, and extras.")
                cA, cB, cC = st.columns(3)
                with cA:
                    profile_mode = st.selectbox("Profile mode", options=["basic", "enhanced"], index=1, key="reco_profile_mode")
                    min_edge_score = st.slider("min_edge_score", 0.50, 1.00, 0.90, 0.01, key="reco_min_edge_score")
                with cB:
                    ind_min_non_null_rows = st.slider("IND min_non_null_rows", 1, 500, 20, 1, key="reco_min_non_null")
                    ind_min_distinct_coverage = st.slider("IND min_distinct_coverage", 0.50, 1.00, 0.90, 0.01, key="reco_min_cov")
                with cC:
                    dtype_selected = st.multiselect(
                        "IND dtype families",
                        options=["int", "string", "float", "datetime", "bool", "other"],
                        default=["int", "string"],
                        key="reco_dtype_sel",
                    )
                    max_rows_per_stage = st.slider("max_rows_per_stage", 100, 5000, 500, 50, key="reco_max_rows")

                cfg = _make_pipeline_config(
                    profile_mode=str(profile_mode),
                    ind_min_non_null_rows=int(ind_min_non_null_rows),
                    ind_min_distinct_coverage=float(ind_min_distinct_coverage),
                    ind_dtype_families=tuple(dtype_selected),
                    min_edge_score=float(min_edge_score),
                    max_rows_per_stage=int(max_rows_per_stage),
                    rel_missing_enabled=False,
                    duplicates_enabled=False,
                    default_fk_enabled=False,
                )

            run_discovery = st.button("Run schema discovery to get inputs", type="primary", use_container_width=True, key="reco_run_discovery")

            if run_discovery:
                with st.spinner("Running schema discovery..."):
                    out = _run_discovery_minimal(data_sig=data_sig, dfs=dfs, cfg=cfg)
                    st.session_state["reco_discovery_out"] = out

            out = st.session_state.get("reco_discovery_out")
            if out is None:
                st.info("Run schema discovery to compute inputs, or enable reuse from other pages.")
                st.stop()

            edges_df = _safe_df(out.get("edges_df"))
            profiles_df = _safe_df(out.get("profiles_df"))
            ucc_df = _safe_df(out.get("ucc_df"))
            composite_ucc_df = _safe_df(out.get("composite_ucc_df"))
            profile_extras = out.get("profile_extras") if isinstance(out.get("profile_extras"), dict) else {}
            duplicates_out = out.get("duplicates") if isinstance(out.get("duplicates"), dict) else {}

        if profiles_df is None or profiles_df.empty:
            st.warning("profiles_df is required to run recommendations.")
            st.stop()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Profiles rows", int(len(profiles_df)))
        with m2:
            st.metric("UCC rows", int(len(ucc_df)) if isinstance(ucc_df, pd.DataFrame) else 0)
        with m3:
            st.metric("Composite UCC rows", int(len(composite_ucc_df)) if isinstance(composite_ucc_df, pd.DataFrame) else 0)
        with m4:
            st.metric("Edges", int(len(edges_df)) if isinstance(edges_df, pd.DataFrame) else 0)

    st.divider()

    # -----------------------------
    # Run recommendations (ENGINE ONLY)
    # -----------------------------
    st.subheader("Run recommendations")

    with st.container(border=True):
        c1, c2 = st.columns([1.3, 1.3])
        with c1:
            include_json = st.toggle("Store full output in session", value=True, key="reco_store_output")
        with c2:
            st.caption("Engine mode only. Output is displayed per tab and exported as JSON/CSV.")

        run_clicked = st.button("Run", type="primary", use_container_width=True, key="reco_run_btn")
        if run_clicked:
            st.session_state.pop("reco_out", None)

        if st.session_state.get("reco_out") is None:
            with st.spinner("Building recommendations..."):
                out_payload = {
                    "profiles_df": profiles_df,
                    "ucc_df": ucc_df if isinstance(ucc_df, pd.DataFrame) else pd.DataFrame(),
                    "composite_ucc_df": composite_ucc_df if isinstance(composite_ucc_df, pd.DataFrame) else pd.DataFrame(),
                    "edges_df": edges_df if isinstance(edges_df, pd.DataFrame) else pd.DataFrame(),
                    "profile_extras": profile_extras or {},
                    "duplicates": duplicates_out or {},
                    # optional if/when you compute them
                    "rel_missing_df": pd.DataFrame(),
                    "default_fk_df": pd.DataFrame(),
                    "surrogate_keys": {},
                }

                schema_rec = build_schema_recommendations(
                    dfs=dfs,
                    out=out_payload,
                    top_n_alternatives=3,
                    normalisation_policy=None,
                )
                out_obj = _json_sanitize(_to_plain(schema_rec))

                st.session_state["reco_out"] = out_obj if include_json else {"note": "output not stored"}
        else:
            out_obj = st.session_state["reco_out"]

    st.divider()

    # -----------------------------
    # Shared table selector + tabs
    # -----------------------------
    blocks = _get_table_blocks(out_obj if isinstance(out_obj, dict) else {})
    safe_out = out_obj if isinstance(out_obj, dict) else {}

    names = [b.get("table_name", "table") for b in blocks]
    if not names:
        st.info("No tables available in output yet.")
        st.stop()

    selected_table = st.selectbox("Table", options=names, index=0, key="reco_table_select")

    tab_pk, tab_fk, tab_ck, tab_dt, tab_idx, tab_all = st.tabs(
        ["Primary keys", "Foreign keys", "Composite keys", "Datatypes", "Indexes", "Full output"]
    )

    with tab_pk:
        render_pk_tab(blocks, safe_out, selected_table)

    with tab_fk:
        render_fk_tab(blocks, safe_out, selected_table)

    with tab_ck:
        render_ck_tab(blocks, safe_out, selected_table)

    with tab_dt:
        render_dt_tab(blocks, safe_out, selected_table)

    with tab_idx:
        render_idx_tab(blocks, safe_out, selected_table)

    with tab_all:
        render_all_tab(safe_out)

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        if st.button("Clear outputs", type="primary", use_container_width=True, key="reco_clear_outputs"):
            st.session_state.pop("reco_out", None)
            st.session_state.pop("reco_discovery_out", None)
            st.toast("Cleared recommendation outputs.", icon="🧹")
            st.rerun()
    with cR:
        if st.button("Clear Streamlit cache and rerun", type="primary", use_container_width=True, key="reco_clear_cache"):
            st.cache_data.clear()
            st.session_state.pop("reco_out", None)
            st.session_state.pop("reco_discovery_out", None)
            st.toast("Cleared Streamlit cache.", icon="🧼")
            st.rerun()
