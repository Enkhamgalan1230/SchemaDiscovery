from __future__ import annotations

import hashlib
import io
from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from schema_discovery.profiling.profiler import profile_all_tables
from schema_discovery.candidates.ucc_unary import discover_ucc_unary
from schema_discovery.candidates.ind_unary import discover_ind_unary
from schema_discovery.quality.key_representations import KeyRepConfig
from schema_discovery.candidates.ind_unary_soft import (
    discover_ind_unary_soft_all_pairs,
    SoftIndConfig,
)
from schema_discovery.scoring.fk_score import score_ind_candidates
from schema_discovery.selection.select_edges import select_best_edges
from schema_discovery.viz.erd_graphviz import edges_to_dot


# ==========================================================
# Helpers
# ==========================================================
def _file_signature(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.name):
        data = f.getvalue()
        h.update(f.name.encode("utf-8"))
        h.update(str(len(data)).encode("utf-8"))
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _load_dfs_from_upload_by_sig(
    sig: str,
    file_bytes: list[tuple[str, bytes]],
) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for fname, data in file_bytes:
        name = fname.rsplit(".", 1)[0]
        dfs[name] = pd.read_csv(
            io.BytesIO(data),
            dtype="string",
            keep_default_na=False,
            na_values=["", "NULL", "null", "None", "none", "N/A", "n/a", "NA", "na"],
            low_memory=False,
        )
    return dfs


@st.cache_data(show_spinner=False)
def _load_dfs_from_upload(files) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for f in files:
        name = f.name.rsplit(".", 1)[0]
        dfs[name] = pd.read_csv(
            f,
            dtype="string",
            keep_default_na=False,
            na_values=["", "NULL", "null", "None", "none", "N/A", "n/a", "NA", "na"],
            low_memory=False,
        )
    return dfs


@st.cache_data(show_spinner=False)
def _run_step_profile(dfs: dict[str, pd.DataFrame], sample_k: int = 5) -> pd.DataFrame:
    profiles_df, _extras = profile_all_tables(dfs, sample_k=sample_k, mode="basic")
    return profiles_df


@st.cache_data(show_spinner=False)
def _run_step_ucc(
    profiles_df: pd.DataFrame,
    unique_ratio_min: float,
    null_ratio_max: float,
    use_flag_if_present: bool,
) -> pd.DataFrame:
    return discover_ucc_unary(
        profiles_df=profiles_df,
        unique_ratio_min=unique_ratio_min,
        null_ratio_max=null_ratio_max,
        use_flag_if_present=use_flag_if_present,
    )


# ---------------------------
# STRICT IND
# ---------------------------
@st.cache_data(show_spinner=False)
def _run_step_ind_strict(
    dfs: dict[str, pd.DataFrame],
    ucc_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    min_distinct_child: int,
    min_non_null_rows: int,
    min_distinct_coverage: float,
    dtype_families: tuple[str, ...],
) -> pd.DataFrame:
    return discover_ind_unary(
        dfs=dfs,
        ucc_df=ucc_df,
        profiles_df=profiles_df,
        min_distinct_child=min_distinct_child,
        min_non_null_rows=min_non_null_rows,
        min_distinct_coverage=min_distinct_coverage,
        dtype_families=dtype_families,
    )


# ---------------------------
# SOFT IND (now also returns reject log text)
# ---------------------------
@st.cache_data(show_spinner=False)
def _run_step_ind_soft(
    dfs: dict[str, pd.DataFrame],
    profiles_df: pd.DataFrame,
    ucc_df: pd.DataFrame,
    soft_cfg_dict: dict[str, Any],
    capture_rejects: bool,
) -> tuple[pd.DataFrame, str]:
    """
    Returns:
      (ind_df, reject_csv_text)

    reject_csv_text is "" if capture_rejects=False or nothing was rejected.
    """
    d = dict(soft_cfg_dict)

    rep_cfg_raw = d.get("rep_cfg", None)
    if isinstance(rep_cfg_raw, dict):
        d["rep_cfg"] = KeyRepConfig(**rep_cfg_raw)
    elif rep_cfg_raw is None:
        d["rep_cfg"] = KeyRepConfig()

    soft_cfg = SoftIndConfig(**d)

    reject_buf = io.StringIO() if capture_rejects else None

    ind_df = discover_ind_unary_soft_all_pairs(
        dfs=dfs,
        profiles_df=profiles_df,
        ucc_df=ucc_df,
        cfg=soft_cfg,
        reject_log_path=reject_buf,  # file-like buffer
    )

    reject_csv = reject_buf.getvalue() if reject_buf is not None else ""
    return ind_df, reject_csv


@st.cache_data(show_spinner=False)
def _run_step_score(
    ind_df: pd.DataFrame,
    w_distinct: float,
    w_row: float,
    w_name: float,
    w_range_penalty: float,
    fk_distinct_small_max: int,
) -> pd.DataFrame:
    return score_ind_candidates(
        ind_df=ind_df,
        w_distinct=w_distinct,
        w_row=w_row,
        w_name=w_name,
        w_range_penalty=w_range_penalty,
        fk_distinct_small_max=fk_distinct_small_max,
    )


@st.cache_data(show_spinner=False)
def _run_step_select(
    scored_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    min_score: float,
    optional_null_ratio_min: float,
    one_to_one_unique_ratio_min: float,
) -> pd.DataFrame:
    return select_best_edges(
        scored_df=scored_df,
        profiles_df=profiles_df,
        min_score=min_score,
        optional_null_ratio_min=optional_null_ratio_min,
        one_to_one_unique_ratio_min=one_to_one_unique_ratio_min,
    )


def _df_download_buttons(df: pd.DataFrame, base_name: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label=f"Download {base_name}.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            label=f"Download {base_name}.json",
            data=df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"{base_name}.json",
            mime="application/json",
            use_container_width=True,
        )


def _soft_cfg_to_cache_dict(cfg: SoftIndConfig) -> dict[str, Any]:
    d = asdict(cfg)
    if not isinstance(d.get("rep_cfg"), dict):
        d["rep_cfg"] = asdict(KeyRepConfig())
    return d


# ==========================================================
# Page
# ==========================================================
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Data schematics core components")
    st.caption("Inspect each pipeline core step using tabs, with previews")

    # ----------------------------------------------------------
    # Data source
    # ----------------------------------------------------------
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

            sig = _file_signature(files)
            if st.session_state.get("prof_data_sig") != sig:
                st.session_state["prof_data_sig"] = sig
                st.session_state.pop("prof_out", None)

            file_bytes = [(f.name, f.getvalue()) for f in files]
            dfs = _load_dfs_from_upload_by_sig(sig, file_bytes)
            st.session_state["dfs_uploaded"] = dfs
            st.success(f"Loaded {len(dfs)} table(s).")

        preview_table = st.selectbox("Preview table", options=list(dfs.keys()))
        st.dataframe(dfs[preview_table].head(10), use_container_width=True)

    st.divider()

    # ----------------------------------------------------------
    # Controls
    # ----------------------------------------------------------
    with st.expander("Components config", expanded=False):
        st.markdown("### IND mode")
        ind_mode = st.radio(
            "Discovery mode",
            options=["strict", "soft"],
            index=0,
            horizontal=True,
            help="strict -> exact matching. soft -> representation matching (padding, numeric tokens, etc).",
        )

        cA, cB, cC = st.columns(3)

        with cA:
            sample_k = st.slider(
                "Profile sample_k", 0, 20, 5, 1,
                help="How many example values to store per column for inspection. Does not affect discovery results."
            )
            ucc_unique_ratio_min = st.slider(
                "UCC unique_ratio_min", 0.90, 1.00, 1.00, 0.001,
                help="Minimum uniqueness to treat a column as a key candidate. 1.0 means all rows unique."
            )
            ucc_null_ratio_max = st.slider(
                "UCC null_ratio_max", 0.00, 0.50, 0.00, 0.01,
                help="Maximum null ratio allowed for key candidates. 0.0 means no missing values allowed."
            )

        with cB:
            ind_min_non_null_rows = st.slider(
                "IND min_non_null_rows (shared)", 1, 500, 50, 1,
                help="Minimum non null rows required for a column to be considered."
            )

            ind_min_distinct_child = st.slider(
                "STRICT: min_distinct_child", 1, 50, 10, 1,
                help="Strict only. Minimum distinct values in a child column before testing it as a potential foreign key."
            )
            ind_min_distinct_coverage = st.slider(
                "STRICT: min_distinct_coverage", 0.50, 1.00, 0.90, 0.01,
                help="Strict only. Distinct coverage threshold."
            )

        with cC:
            dtype_opts = ["int", "string", "float", "datetime", "bool", "other"]
            dtype_selected = st.multiselect(
                "STRICT: dtype_families",
                options=dtype_opts,
                default=["int", "string"],
                help="Strict only. Only compare columns from these dtype families. Typically keep to int and string for IDs."
            )
            min_edge_score = st.slider(
                "Select min_score", 0.50, 1.00, 0.80, 0.01,
                help="Minimum final score to accept an edge after scoring."
            )
            optional_null_ratio_min = st.slider(
                "Optional null_ratio >= ", 0.00, 0.50, 0.01, 0.01,
                help="If FK column null ratio is at least this, mark the relationship as optional."
            )

        # Default for non-soft
        capture_rejects = False

        soft_cfg = SoftIndConfig()
        if ind_mode == "soft":
            st.markdown("### Soft IND thresholds")

            s1, s2, s3 = st.columns(3)
            with s1:
                soft_min_distinct_coverage = st.slider(
                    "SOFT: min_distinct_coverage", 0.50, 1.00, float(soft_cfg.min_distinct_coverage), 0.01,
                    help="Soft only. Minimum distinct coverage to accept a candidate."
                )
                soft_min_row_coverage = st.slider(
                    "SOFT: min_row_coverage", 0.50, 1.00, float(soft_cfg.min_row_coverage), 0.01,
                    help="Soft only. Minimum row coverage to accept a candidate."
                )
            with s2:
                soft_min_name_sim_for_routing = st.slider(
                    "SOFT: min_name_sim_for_routing", 0.0, 1.0, float(soft_cfg.min_name_sim_for_routing), 0.01,
                    help="Soft only. Routing threshold to limit FK->PK comparisons."
                )
                soft_max_parents_per_fk = st.slider(
                    "SOFT: max_parents_per_fk", 1, 200, int(soft_cfg.max_parents_per_fk), 1,
                    help="Soft only. Cap number of parent candidates compared per FK column."
                )
            with s3:
                soft_small_domain_fk_distinct_max = st.slider(
                    "SOFT: small_domain_fk_distinct_max", 1, 500, int(soft_cfg.small_domain_fk_distinct_max), 1,
                    help="Soft only. FK columns with <= this many distinct values are considered small-domain."
                )
                soft_small_domain_min_name_sim = st.slider(
                    "SOFT: small_domain_min_name_sim", 0.0, 1.0, float(soft_cfg.small_domain_min_name_sim), 0.01,
                    help="Soft only. Extra name similarity required for small-domain FK columns."
                )

            capture_rejects = st.toggle(
                "SOFT: capture reject log",
                value=True,
                help="Stores rejected routed FK->PK pairs with reasons, and shows them in a new tab.",
            )

            soft_cfg = SoftIndConfig(
                parent_source="ucc_plus_profile",
                min_distinct_coverage=float(soft_min_distinct_coverage),
                min_row_coverage=float(soft_min_row_coverage),
                min_non_null=int(ind_min_non_null_rows),
                min_name_sim_for_routing=float(soft_min_name_sim_for_routing),
                max_parents_per_fk=int(soft_max_parents_per_fk),
                small_domain_fk_distinct_max=int(soft_small_domain_fk_distinct_max),
                small_domain_min_name_sim=float(soft_small_domain_min_name_sim),
                allow_ucc_children=True,
            )

        st.markdown("### Scoring weights")
        w1, w2, w3, w4, w5 = st.columns(5)

        with w1:
            w_distinct = st.slider("w_distinct", 0.0, 1.0, 0.75, 0.01)
        with w2:
            w_row = st.slider("w_row", 0.0, 1.0, 0.20, 0.01)
        with w3:
            w_name = st.slider("w_name", 0.0, 1.0, 0.05, 0.01)
        with w4:
            w_range_penalty = st.slider("w_range_penalty", 0.0, 1.0, 0.10, 0.01)
        with w5:
            fk_distinct_small_max = st.number_input("fk_distinct_small_max", min_value=1, value=2000, step=100)

        one_to_one_unique_ratio_min = st.slider(
            "one_to_one_unique_ratio_min", 0.90, 1.00, 0.999999, 0.000001,
            help="If FK is almost unique (among non nulls), classify as one to one. Otherwise many to one."
        )

        use_flag_if_present = st.toggle(
            "Use is_unary_ucc flag if present", value=True,
            help="Use strict profiling flag (unique and non null) when available as an extra UCC signal."
        )

    # ----------------------------------------------------------
    # Run steps
    # ----------------------------------------------------------
    if st.button("Compute steps", type="primary", use_container_width=True):
        st.session_state["prof_out"] = {}

    prof_out: dict[str, Any] = st.session_state.get("prof_out", {})

    def get_profiles() -> pd.DataFrame:
        if "profiles_df" not in prof_out:
            with st.spinner("Profiling tables..."):
                prof_out["profiles_df"] = _run_step_profile(dfs, sample_k=sample_k)
                st.session_state["prof_out"] = prof_out
        return prof_out["profiles_df"]

    def get_ucc(profiles_df: pd.DataFrame) -> pd.DataFrame:
        if "ucc_df" not in prof_out:
            with st.spinner("Discovering UCC candidates..."):
                prof_out["ucc_df"] = _run_step_ucc(
                    profiles_df,
                    unique_ratio_min=ucc_unique_ratio_min,
                    null_ratio_max=ucc_null_ratio_max,
                    use_flag_if_present=use_flag_if_present,
                )
                st.session_state["prof_out"] = prof_out
        return prof_out["ucc_df"]

    def get_ind(profiles_df: pd.DataFrame, ucc_df: pd.DataFrame) -> pd.DataFrame:
        cache_key = "ind_df_soft" if ind_mode == "soft" else "ind_df_strict"
        reject_key = "rejects_csv_soft"

        if cache_key not in prof_out:
            with st.spinner(f"Discovering IND candidates ({ind_mode})..."):
                if ind_mode == "soft":
                    soft_cfg_dict = _soft_cfg_to_cache_dict(soft_cfg)
                    ind_df, rejects_csv = _run_step_ind_soft(
                        dfs=dfs,
                        profiles_df=profiles_df,
                        ucc_df=ucc_df,
                        soft_cfg_dict=soft_cfg_dict,
                        capture_rejects=bool(capture_rejects),
                    )
                    prof_out[cache_key] = ind_df
                    prof_out[reject_key] = rejects_csv
                else:
                    prof_out[cache_key] = _run_step_ind_strict(
                        dfs=dfs,
                        ucc_df=ucc_df,
                        profiles_df=profiles_df,
                        min_distinct_child=int(ind_min_distinct_child),
                        min_non_null_rows=int(ind_min_non_null_rows),
                        min_distinct_coverage=float(ind_min_distinct_coverage),
                        dtype_families=tuple(dtype_selected),
                    )
                    prof_out[reject_key] = ""

                st.session_state["prof_out"] = prof_out

        return prof_out[cache_key]

    def get_scored(ind_df: pd.DataFrame) -> pd.DataFrame:
        scored_key = "scored_df_soft" if ind_mode == "soft" else "scored_df_strict"
        if scored_key not in prof_out:
            with st.spinner("Scoring candidates..."):
                prof_out[scored_key] = _run_step_score(
                    ind_df=ind_df,
                    w_distinct=float(w_distinct),
                    w_row=float(w_row),
                    w_name=float(w_name),
                    w_range_penalty=float(w_range_penalty),
                    fk_distinct_small_max=int(fk_distinct_small_max),
                )
                st.session_state["prof_out"] = prof_out
        return prof_out[scored_key]

    def get_edges(scored_df: pd.DataFrame, profiles_df: pd.DataFrame) -> pd.DataFrame:
        edges_key = "edges_df_soft" if ind_mode == "soft" else "edges_df_strict"
        if edges_key not in prof_out:
            with st.spinner("Selecting best edges..."):
                prof_out[edges_key] = _run_step_select(
                    scored_df=scored_df,
                    profiles_df=profiles_df,
                    min_score=float(min_edge_score),
                    optional_null_ratio_min=float(optional_null_ratio_min),
                    one_to_one_unique_ratio_min=float(one_to_one_unique_ratio_min),
                )
                st.session_state["prof_out"] = prof_out

                st.session_state["edges_out"] = {
                    "edges_df": prof_out[edges_key],
                    "profiles_df": prof_out.get("profiles_df"),
                    "source": f"Extras (Components page, ind_mode={ind_mode})",
                }
        return prof_out[edges_key]

    # ----------------------------------------------------------
    # Tabs
    # ----------------------------------------------------------
    (
        tab_overview,
        tab_profile,
        tab_ucc,
        tab_ind,
        tab_rejects,
        tab_score,
        tab_select,
        tab_erd,
    ) = st.tabs(
        [
            "Overview",
            "1) Profiling",
            "2) UCC",
            "3) IND",
            "3b) Reject log",
            "4) Scoring",
            "5) Selection",
            "ERD",
        ]
    )

    with tab_overview:
        st.subheader("What you can inspect here")
        st.write(
            """
            This page runs each step of the schema discovery pipeline and shows the intermediate outputs.
            Use it to debug why relationships appear or do not appear.
            """
        )

        if st.button("Build summary now", use_container_width=True):
            profiles_df = get_profiles()
            ucc_df = get_ucc(profiles_df)
            ind_df = get_ind(profiles_df, ucc_df)
            scored_df = get_scored(ind_df)
            edges_df = get_edges(scored_df, profiles_df)

        profiles_df = prof_out.get("profiles_df")
        ucc_df = prof_out.get("ucc_df")

        ind_key = "ind_df_soft" if ind_mode == "soft" else "ind_df_strict"
        scored_key = "scored_df_soft" if ind_mode == "soft" else "scored_df_strict"
        edges_key = "edges_df_soft" if ind_mode == "soft" else "edges_df_strict"

        ind_df = prof_out.get(ind_key)
        scored_df = prof_out.get(scored_key)
        edges_df = prof_out.get(edges_key)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tables", len(dfs))
        c2.metric("Profiled columns", 0 if profiles_df is None else len(profiles_df))
        c3.metric("UCC candidates", 0 if ucc_df is None else len(ucc_df))
        c4.metric("IND candidates", 0 if ind_df is None else len(ind_df))
        c5.metric("Selected edges", 0 if edges_df is None else len(edges_df))

        with st.expander("Current parameters"):
            base = dict(
                profiling=dict(sample_k=sample_k),
                ucc=dict(unique_ratio_min=ucc_unique_ratio_min, null_ratio_max=ucc_null_ratio_max),
                scoring=dict(
                    w_distinct=w_distinct,
                    w_row=w_row,
                    w_name=w_name,
                    w_range_penalty=w_range_penalty,
                    fk_distinct_small_max=fk_distinct_small_max,
                ),
                selection=dict(
                    min_score=min_edge_score,
                    optional_null_ratio_min=optional_null_ratio_min,
                    one_to_one_unique_ratio_min=one_to_one_unique_ratio_min,
                ),
                ind_mode=ind_mode,
            )
            if ind_mode == "soft":
                base["ind"] = dict(**_soft_cfg_to_cache_dict(soft_cfg), capture_rejects=bool(capture_rejects))
            else:
                base["ind"] = dict(
                    dtype_families=dtype_selected,
                    min_distinct_child=ind_min_distinct_child,
                    min_non_null_rows=ind_min_non_null_rows,
                    min_distinct_coverage=ind_min_distinct_coverage,
                )
            st.json(base)

    with tab_profile:
        profiles_df = get_profiles()
        st.subheader("Profiles")
        st.caption("One row per (table, column). Use this to understand dtypes, nulls, uniqueness.")
        st.dataframe(profiles_df, use_container_width=True, height=520)
        _df_download_buttons(profiles_df, "profiles")

    with tab_ucc:
        profiles_df = get_profiles()
        ucc_df = get_ucc(profiles_df)

        st.subheader("Unary UCC candidates")
        st.caption("Columns that are unique and (optionally) non null. Used as potential referenced keys.")
        if ucc_df.empty:
            st.warning("No UCC candidates found with current thresholds.")
        else:
            st.dataframe(ucc_df, use_container_width=True, height=520)
            _df_download_buttons(ucc_df, "ucc_unary")

    with tab_ind:
        profiles_df = get_profiles()
        ucc_df = get_ucc(profiles_df)
        ind_df = get_ind(profiles_df, ucc_df)

        st.subheader(f"Unary IND candidates ({ind_mode})")
        if ind_mode == "soft":
            st.caption(
                "Soft mode uses key representations (numeric tokens, zero padding, trimming) and routing to avoid blow up. "
                "Extra evidence fields include fk_rep and pk_rep."
            )
        else:
            st.caption("Strict mode tests child columns against parent UCC domains with exact token matching.")

        if ind_df.empty:
            st.warning("No IND candidates found with current thresholds.")
        else:
            st.dataframe(ind_df, use_container_width=True, height=520)
            _df_download_buttons(ind_df, f"ind_unary_{ind_mode}")

    with tab_rejects:
        st.subheader("Soft IND reject log")

        if ind_mode != "soft":
            st.info("Reject log is only available in soft mode.")
        else:
            rejects_csv = prof_out.get("rejects_csv_soft", "")

            if not rejects_csv.strip():
                st.warning("No reject log captured. Enable 'SOFT: capture reject log' and run Compute steps.")
            else:
                rejects_df = pd.read_csv(io.StringIO(rejects_csv), dtype="string", keep_default_na=False)
                st.caption("Routed FK->PK pairs that were rejected, with a reason.")
                st.dataframe(rejects_df, use_container_width=True, height=520)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download rejects.csv",
                        data=rejects_csv.encode("utf-8"),
                        file_name="soft_rejects.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary",
                    )
                with c2:
                    st.download_button(
                        "Download rejects.json",
                        data=rejects_df.to_json(orient="records", indent=2).encode("utf-8"),
                        file_name="soft_rejects.json",
                        mime="application/json",
                        use_container_width=True,
                    )

    with tab_score:
        profiles_df = get_profiles()
        ucc_df = get_ucc(profiles_df)
        ind_df = get_ind(profiles_df, ucc_df)
        scored_df = get_scored(ind_df)

        st.subheader("Scored candidates")
        st.caption("Coverage dominates. Name similarity is a weak tie breaker. Range penalty reduces accidental inclusions.")
        if scored_df.empty:
            st.warning("No scored candidates available.")
        else:
            st.dataframe(scored_df, use_container_width=True, height=520)
            _df_download_buttons(scored_df, f"scored_candidates_{ind_mode}")

    with tab_select:
        profiles_df = get_profiles()
        ucc_df = get_ucc(profiles_df)
        ind_df = get_ind(profiles_df, ucc_df)
        scored_df = get_scored(ind_df)
        edges_df = get_edges(scored_df, profiles_df)

        st.subheader("Selected edges")
        st.caption("Highest scoring parent is chosen per (fk_table, fk_column). Optional and cardinality are inferred.")
        if edges_df.empty:
            st.warning("No edges selected with current min_score.")
        else:
            st.dataframe(edges_df, use_container_width=True, height=520)
            _df_download_buttons(edges_df, f"edges_selected_{ind_mode}")

    with tab_erd:
        profiles_df = get_profiles()
        ucc_df = get_ucc(profiles_df)
        ind_df = get_ind(profiles_df, ucc_df)
        scored_df = get_scored(ind_df)
        edges_df = get_edges(scored_df, profiles_df)

        st.subheader(f"Entity Relationship Diagram ({ind_mode})")

        if edges_df.empty:
            st.warning("No edges selected, ERD is empty.")
        else:
            dot = edges_to_dot(edges_df)

            st.graphviz_chart(dot, use_container_width=True)

            with st.expander("DOT source"):
                st.code(dot, language="dot")
                st.download_button(
                    "Download DOT",
                    data=dot,
                    file_name=f"erd_{ind_mode}.dot",
                    mime="text/vnd.graphviz",
                    use_container_width=True,
                    type="primary",
                )

    # ----------------------------------------------------------
    # Convenience actions
    # ----------------------------------------------------------
    st.divider()
    cL, cR = st.columns([1, 1])
    with cL:
        if st.button("Clear step outputs", use_container_width=True):
            st.session_state.pop("prof_out", None)
            st.toast("Cleared profiler outputs.", icon="🧹")
    with cR:
        if st.button("Clear cached data and rerun", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("prof_out", None)
            st.toast("Cleared Streamlit cache.", icon="🧼")