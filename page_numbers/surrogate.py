from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from schema_discovery.pipeline.run import run_schema_discovery, PipelineConfig
from schema_discovery.annotation.surrogate_keys import (
    SurrogateKeyConfig,
    annotate_surrogate_keys,
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


def _get_discovery_from_session() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """
    Returns:
      edges_df, profiles_df, ucc_df, source
    """
    eo = st.session_state.get("edges_out")
    if isinstance(eo, dict):
        e = eo.get("edges_df")
        p = eo.get("profiles_df")
        u = eo.get("ucc_df")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                eo.get("source") or "Edges cache",
            )

    out = st.session_state.get("schema_out")
    if isinstance(out, dict):
        e = out.get("edges_df")
        p = out.get("profiles_df")
        u = out.get("ucc_df")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                "Full Pipeline (Table Relation page)",
            )

    prof_out = st.session_state.get("prof_out")
    if isinstance(prof_out, dict):
        e = prof_out.get("edges_df")
        p = prof_out.get("profiles_df")
        u = prof_out.get("ucc_df")
        if isinstance(p, pd.DataFrame) and not p.empty:
            return (
                (e if isinstance(e, pd.DataFrame) else pd.DataFrame()),
                p,
                (u if isinstance(u, pd.DataFrame) else pd.DataFrame()),
                "Extras (Components page)",
            )

    return None, None, None, None


@st.cache_data(show_spinner=False)
def _load_dfs(files) -> dict[str, pd.DataFrame]:
    return {f.name.rsplit(".", 1)[0]: pd.read_csv(f) for f in files}


@st.cache_data(show_spinner=False)
def _run_discovery(
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


@st.cache_data(show_spinner=False)
def _run_surrogate(
    data_sig: str,
    dfs: dict[str, pd.DataFrame],
    profiles_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ucc_df: pd.DataFrame,
    cfg: SurrogateKeyConfig,
) -> dict[str, Any]:
    return annotate_surrogate_keys(
        dfs=dfs,
        profiles_df=profiles_df,
        edges_df=edges_df,
        ucc_df=ucc_df if isinstance(ucc_df, pd.DataFrame) else None,
        cfg=cfg,
    )


def _status_dot(is_surrogate: bool, confidence: float) -> str:
    """
    Red -> surrogate with high confidence
    Yellow -> surrogate (lower) or borderline
    Green -> not surrogate
    """
    if is_surrogate and confidence >= 0.85:
        return "🔴"
    if is_surrogate and confidence >= 0.75:
        return "🟠"
    # borderline not-surrogate (close to threshold)
    if (not is_surrogate) and confidence >= 0.70:
        return "🟡"
    return "🟢"


def _flatten_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    if "signals" in df.columns:
        sig = pd.json_normalize(df["signals"]).add_prefix("sig.")
        df = pd.concat([df.drop(columns=["signals"]), sig], axis=1)
    return df


# -----------------------------
# Page
# -----------------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Surrogate Key Detection")
    st.caption(
        "Detects key-like columns and estimates whether they are database generated surrogates "
        "(UUIDs, auto-increment IDs) while avoiding natural identifiers (email, SKU, barcode)."
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
    # Inputs
    # -----------------------------
    with st.container(border=True):
        st.subheader("Inputs")
        st.caption("This page needs profiles_df. edges_df and ucc_df are optional but improve context.")

        use_cached = st.toggle(
            "Reuse discovery outputs from Table Relation page (if available)",
            value=False,
        )

        edges_df: pd.DataFrame | None = None
        profiles_df: pd.DataFrame | None = None
        ucc_df: pd.DataFrame | None = None

        if use_cached:
            e, p, u, source = _get_discovery_from_session()
            if isinstance(p, pd.DataFrame) and not p.empty:
                edges_df, profiles_df, ucc_df = e, p, u
                st.success(f"Reusing discovery outputs from: {source}")
            else:
                st.warning("No cached discovery outputs found. Falling back to running discovery here.")
                edges_df = profiles_df = ucc_df = None

        if profiles_df is None or profiles_df.empty:
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
                )

            if st.button("Run schema discovery to get inputs", type="primary", use_container_width=True):
                with st.spinner("Running schema discovery..."):
                    edges_df, profiles_df, ucc_df = _run_discovery(data_sig, dfs, cfg)
                    st.session_state["sk_edges_df"] = edges_df
                    st.session_state["sk_profiles_df"] = profiles_df
                    st.session_state["sk_ucc_df"] = ucc_df

            edges_df = st.session_state.get("sk_edges_df") if edges_df is None else edges_df
            profiles_df = st.session_state.get("sk_profiles_df") if profiles_df is None else profiles_df
            ucc_df = st.session_state.get("sk_ucc_df") if ucc_df is None else ucc_df

            if not isinstance(edges_df, pd.DataFrame):
                edges_df = pd.DataFrame()
            if not isinstance(ucc_df, pd.DataFrame):
                ucc_df = pd.DataFrame()
            if not isinstance(profiles_df, pd.DataFrame):
                profiles_df = None

        if profiles_df is None or profiles_df.empty:
            st.warning("profiles_df is required to run surrogate detection.")
            st.stop()

        a, b, c = st.columns(3)
        with a:
            st.metric("Profiles rows", int(len(profiles_df)))
        with b:
            st.metric("UCC candidates", int(len(ucc_df)) if isinstance(ucc_df, pd.DataFrame) else 0)
        with c:
            st.metric("Relationships", int(len(edges_df)) if isinstance(edges_df, pd.DataFrame) else 0)

    st.divider()

    # -----------------------------
    # Config + Run
    # -----------------------------
    st.subheader("Run detection")

    with st.expander("Detector config", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            unique_ratio_min = st.slider("unique_ratio_min", 0.90, 1.00, 0.999, 0.001)
            null_ratio_max = st.slider("null_ratio_max", 0.0, 0.20, 0.01, 0.01)
        with c2:
            continuity_good = st.slider("continuity_good", 0.0, 1.0, 0.80, 0.01)
            referenced_by_edges_good = st.slider("referenced_by_edges_good", 0, 10, 2, 1)
        with c3:
            natural_id_penalty = st.slider("natural_id_penalty", 0.0, 0.50, 0.20, 0.01)
            name_hint = st.toggle("Use name_hint", value=True)

    cfg_sk = SurrogateKeyConfig(
        unique_ratio_min=float(unique_ratio_min),
        null_ratio_max=float(null_ratio_max),
        continuity_good=float(continuity_good),
        referenced_by_edges_good=int(referenced_by_edges_good),
        natural_id_penalty=float(natural_id_penalty),
        name_hint=bool(name_hint),
    )

    if st.button("Run detection", type="primary", use_container_width=True):
        st.session_state.pop("sk_out", None)

    if st.session_state.get("sk_out") is None:
        with st.spinner("Detecting surrogate keys..."):
            out_sk = _run_surrogate(
                data_sig=data_sig,
                dfs=dfs,
                profiles_df=profiles_df,
                edges_df=edges_df if isinstance(edges_df, pd.DataFrame) else pd.DataFrame(),
                ucc_df=ucc_df if isinstance(ucc_df, pd.DataFrame) else pd.DataFrame(),
                cfg=cfg_sk,
            )
            st.session_state["sk_out"] = out_sk
    else:
        out_sk = st.session_state["sk_out"]

    items = (out_sk or {}).get("items", []) or []
    show_df = _flatten_items(items)

    st.subheader("Results")

    if show_df.empty:
        st.info("No key-like columns passed the hard gates.")
        st.stop()

    # Summary metrics
    n_total = int(len(show_df))
    n_flag = int((show_df["is_surrogate"] == True).sum())  # noqa: E712
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Candidates", n_total)
    with m2:
        st.metric("Flagged surrogate", n_flag)
    with m3:
        st.metric("Not surrogate", int(n_total - n_flag))

    st.divider()

    # -----------------------------
    # Cleaner UI: Tabs instead of lots of expanders
    # -----------------------------
    tab1, tab2, tab3 = st.tabs(["Table", "Inspect one column", "Raw JSON"])

    with tab1:
        f1, f2, f3, f4 = st.columns([1.2, 1.6, 1.6, 1.4])
        with f1:
            only_flagged = st.toggle("Only flagged", value=False)
        with f2:
            role_filter = st.selectbox("key_role", options=["all", "pk_candidate", "key_candidate"], index=0)
        with f3:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.01)
        with f4:
            top_n = st.number_input("Show top N", min_value=10, max_value=5000, value=200, step=10)

        filtered = show_df.copy()
        if only_flagged:
            filtered = filtered[filtered["is_surrogate"] == True]  # noqa: E712
        if role_filter != "all":
            filtered = filtered[filtered["key_role"] == role_filter]
        filtered = filtered[filtered["confidence"] >= float(min_conf)]

        # Add a compact status column
        filtered = filtered.copy()
        filtered.insert(
            0,
            "status",
            [
                _status_dot(bool(r["is_surrogate"]), float(r["confidence"]))
                for _, r in filtered.iterrows()
            ],
        )

        preferred = [
            "status",
            "table",
            "column",
            "key_role",
            "is_surrogate",
            "confidence",
            "sig.uuid_like",
            "sig.continuity",
            "sig.referenced_by_edges",
            "sig.alt_natural_key_ucc_exists",
            "sig.name_hint",
            "sig.natural_id_hint",
        ]
        cols = [c for c in preferred if c in filtered.columns] + [c for c in filtered.columns if c not in preferred]
        filtered = filtered[cols].head(int(top_n))

        st.dataframe(filtered, width="stretch", height=360)
        _download_df(filtered, base="surrogate_keys")

        st.caption("Legend -> 🔴 high surrogate, 🟠 surrogate, 🟡 borderline, 🟢 not surrogate")

    with tab2:
        st.caption("Pick a single column and view a clean, structured breakdown.")
        show_df2 = show_df.copy()
        show_df2["label"] = show_df2["table"].astype(str) + "." + show_df2["column"].astype(str)

        options = show_df2["label"].tolist()
        picked = st.selectbox("Column", options=options, index=0)

        row = show_df2[show_df2["label"] == picked].iloc[0].to_dict()
        dot = _status_dot(bool(row["is_surrogate"]), float(row["confidence"]))

        h1, h2, h3 = st.columns([2.2, 1.2, 1.6])
        with h1:
            st.markdown("**Column**")
            st.code(picked, language="text")
        with h2:
            st.markdown("**Status**")
            st.write(f"{dot}  confidence={float(row['confidence']):.3f}")
            st.write({"key_role": row.get("key_role"), "is_surrogate": bool(row.get("is_surrogate"))})
        with h3:
            st.markdown("**Core signals**")
            st.write(
                {
                    "uuid_like": row.get("sig.uuid_like"),
                    "continuity": row.get("sig.continuity"),
                    "referenced_by_edges": row.get("sig.referenced_by_edges"),
                }
            )

        st.divider()

        tA, tB = st.columns([1.2, 1.8])
        with tA:
            st.markdown("**Name and natural-id hints**")
            st.write(
                {
                    "name_hint": row.get("sig.name_hint"),
                    "natural_id_hint": row.get("sig.natural_id_hint"),
                    "alt_natural_key_ucc_exists": row.get("sig.alt_natural_key_ucc_exists"),
                }
            )
        with tB:
            st.markdown("**How to interpret**")
            notes = []
            if row.get("sig.uuid_like") is True:
                notes.append("UUID pattern detected -> strong surrogate signal")
            if row.get("sig.continuity") is not None and float(row.get("sig.continuity")) >= float(cfg_sk.continuity_good):
                notes.append("High integer continuity -> likely auto-increment key")
            if int(row.get("sig.referenced_by_edges", 0) or 0) >= int(cfg_sk.referenced_by_edges_good):
                notes.append("Referenced by multiple edges -> likely primary key used in joins")
            if row.get("sig.alt_natural_key_ucc_exists") is True:
                notes.append("Alternative natural unique key exists -> surrogate more likely")
            if float(row.get("sig.natural_id_hint", 0.0) or 0.0) > 0:
                notes.append("Name looks like external identifier (barcode, sku, etc.) -> penalised")
            if not notes:
                notes = ["No strong signals beyond key-like gates (unique + low nulls)."]
            st.write(notes)

    with tab3:
        st.caption("Raw output for debugging or saving.")
        st.json(out_sk)

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        if st.button("Clear outputs", type="primary", use_container_width=True):
            st.session_state.pop("sk_out", None)
            st.toast("Cleared surrogate outputs.", icon="🧹")
    with cR:
        if st.button("Clear Streamlit cache and rerun", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("sk_out", None)
            st.toast("Cleared Streamlit cache.", icon="🧼")
            st.rerun()
