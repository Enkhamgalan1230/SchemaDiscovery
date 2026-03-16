from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from schema_discovery.profiling.profiler import profile_all_tables
from schema_discovery.profiling.atomic_model import InferenceConfig

# Bump this whenever you change profiler output structure to avoid stale cache
PROFILER_CACHE_BUSTER = "2026-02-03-health-tab-v1"


# -----------------------------
# Helpers
# -----------------------------
def _df_signature(dfs: dict[str, pd.DataFrame]) -> str:
    """
    Cheap, stable signature for cache keys.
    Avoid hashing full CSV bytes (too big).
    """
    h = hashlib.sha256()
    for name in sorted(dfs.keys()):
        df = dfs[name]
        h.update(name.encode("utf-8"))
        h.update(str(df.shape[0]).encode("utf-8"))
        h.update(str(df.shape[1]).encode("utf-8"))

        # Hash only a small sample for stability
        sample = df.head(30).to_csv(index=False).encode("utf-8", errors="ignore")
        h.update(hashlib.sha256(sample).digest())
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _load_dfs_from_upload(files) -> dict[str, pd.DataFrame]:
    return {f.name.rsplit(".", 1)[0]: pd.read_csv(f) for f in files}


@st.cache_data(show_spinner=False)
def _run_enhanced_profiles(
    profiler_cache_buster: str,
    data_sig: str,
    dfs: dict[str, pd.DataFrame],
    sample_k: int,
    atomic_profile: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Cache enhanced profiling. We include:
    - profiler_cache_buster: manual knob to bust cache when code/output changes
    - data_sig: data fingerprint to bust cache when input changes
    """
    _ = profiler_cache_buster  # ensures cache key includes it
    atomic_cfg = InferenceConfig(profile=atomic_profile)
    profiles_df, extras = profile_all_tables(
        dfs,
        sample_k=sample_k,
        mode="enhanced",
        atomic_cfg=atomic_cfg,
    )
    return profiles_df, extras


def _download_df_buttons(df: pd.DataFrame, base_name: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"Download {base_name}.csv",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            f"Download {base_name}.json",
            df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"{base_name}.json",
            mime="application/json",
            use_container_width=True,
        )


def _get_table_health(extras: dict[str, Any], table_name: str) -> dict[str, Any] | None:
    """
    Extract per-table health summary from extras.
    Supports:
    - extras["tables"][t]["health"]
    - extras["health"] (fallback)
    """
    if not isinstance(extras, dict):
        return None

    tables = extras.get("tables")
    if isinstance(tables, dict):
        t_extras = tables.get(str(table_name))
        if isinstance(t_extras, dict) and isinstance(t_extras.get("health"), dict):
            return t_extras["health"]

    if isinstance(extras.get("health"), dict):
        return extras["health"]

    return None


def _health_rollup_df(extras: dict[str, Any], profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a small per-table health rollup dataframe for the Health tab.
    """
    rows: list[dict[str, Any]] = []
    tables = extras.get("tables", {}) if isinstance(extras, dict) else {}

    table_names = []
    try:
        table_names = sorted(profiles_df["table_name"].unique().tolist())
    except Exception:
        table_names = sorted(list(tables.keys()))

    for tname in table_names:
        h = _get_table_health(extras, tname) or {}
        counts = h.get("counts") or {}
        rows.append(
            {
                "table_name": tname,
                "avg_health_score": h.get("avg_score"),
                "ok_cols": int(counts.get("ok", 0) or 0),
                "warn_cols": int(counts.get("warn", 0) or 0),
                "fail_cols": int(counts.get("fail", 0) or 0),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("table_name").reset_index(drop=True)
    return out


# -----------------------------
# Page
# -----------------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Enhanced Profiler")
    st.caption(
        "Upload tables and view enhanced per-column profiling, using the atomic dtype model, quality metrics, and health checks."
    )

    # -------- Data source --------
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
            dfs = _load_dfs_from_upload(files)
            st.session_state["dfs_uploaded"] = dfs
            st.success(f"Loaded {len(dfs)} table(s).")

        preview_table = st.selectbox("Preview table", options=list(dfs.keys()))
        st.dataframe(dfs[preview_table].head(10), use_container_width=True)

    st.divider()

    # -------- Controls --------
    with st.expander("Profiler config", expanded=False):
        cA, cB = st.columns(2)
        with cA:
            sample_k = st.slider(
                "Sample values per column (sample_k)",
                min_value=0,
                max_value=20,
                value=5,
                step=1,
                help="Stores up to k example distinct values per column for inspection.",
            )
        with cB:
            atomic_profile = st.toggle(
                "Atomic model profiling mode",
                value=False,
                help="If enabled, atomic inference may compute extra diagnostics and be slower.",
            )

    # -------- Run enhanced profiler --------
    data_sig = _df_signature(dfs)

    if st.button("Run enhanced profiling", type="primary", use_container_width=True):
        st.session_state["enhanced_out_sig"] = data_sig
        st.session_state.pop("enhanced_out", None)

    # Auto-run if nothing exists for current data signature
    out_sig = st.session_state.get("enhanced_out_sig")
    if st.session_state.get("enhanced_out") is None or out_sig != data_sig:
        with st.spinner("Profiling tables (enhanced)..."):
            profiles_df, extras = _run_enhanced_profiles(
                profiler_cache_buster=PROFILER_CACHE_BUSTER,
                data_sig=data_sig,
                dfs=dfs,
                sample_k=int(sample_k),
                atomic_profile=bool(atomic_profile),
            )
            st.session_state["enhanced_out"] = (profiles_df, extras)
            st.session_state["enhanced_out_sig"] = data_sig
    else:
        profiles_df, extras = st.session_state["enhanced_out"]

    # -------- Tabs --------
    tab_explorer, tab_health, tab_raw = st.tabs(["Column Explorer", "Health", "Raw Outputs"])

    with tab_explorer:
        st.subheader("Column Explorer")
        st.caption("Pick a column to see enhanced metrics and what they mean.")

        t = st.selectbox("Table", options=sorted(profiles_df["table_name"].unique().tolist()))
        cols = profiles_df.loc[profiles_df["table_name"] == t, "column_name"].tolist()
        c = st.selectbox("Column", options=cols)

        row = profiles_df[(profiles_df["table_name"] == t) & (profiles_df["column_name"] == c)].iloc[0].to_dict()

        # Show core info
        top_left, top_right = st.columns([2, 3])
        with top_left:
            st.markdown("### Core")
            st.write(
                {
                    "dtype_family": row.get("dtype_family"),
                    "dtype": row.get("dtype"),
                    "n_rows": row.get("n_rows"),
                    "n_non_null": row.get("n_non_null"),
                    "null_ratio": row.get("null_ratio"),
                    "n_unique": row.get("n_unique"),
                    "unique_ratio": row.get("unique_ratio"),
                    "is_unary_ucc": row.get("is_unary_ucc"),
                }
            )

            # Health block (per-column)
            if "health_status" in row:
                st.markdown("### Column health")
                st.write(
                    {
                        "health_status": row.get("health_status"),
                        "health_severity": row.get("health_severity"),
                        "health_score": row.get("health_score"),
                        "health_reasons": row.get("health_reasons"),
                    }
                )

            st.markdown("### Samples")
            st.write(row.get("sample_values", []))

        with top_right:
            st.markdown("### Enhanced")
            enhanced_fields = [
                "atomic_type",
                "atomic_prob",
                "atomic_convertible_ratio",
                "atomic_reason",
                "completeness",
                "consistency_score",
                "overall_quality",
                "pk_continuity",
                "num_mean",
                "num_median",
                "num_min",
                "num_max",
                "num_std_dev",
                "num_iqr",
                "num_skewness",
                "date_min",
                "date_max",
                "date_range_days",
                "freshness_score",
            ]

            shown = {k: row.get(k) for k in enhanced_fields if k in row and row.get(k) is not None}
            if not shown:
                st.info("No enhanced fields available for this column (or values are empty).")
            else:
                st.write(shown)

        st.divider()
        with st.expander("### What these fields mean"):
            st.markdown(
                """
                1. atomic_type: Inferred atomic type from the atomic model (often more informative than pandas dtype).
                2. atomic_prob: Confidence for atomic_type (0.0 to 1.0).
                3. atomic_convertible_ratio: Share of non-null values that can be converted to atomic_type (0.0 to 1.0).
                4. atomic_reason: Short explanation of why the atomic model chose atomic_type.
                5. completeness: Percent of rows that are non-null (0 to 100).
                6. consistency_score: How consistent values look within the inferred dtype family (0 to 100).
                7. overall_quality: Combined score from completeness, uniqueness, and consistency (0 to 100).
                8. pk_continuity: For integer-like key candidates -> how continuous the integer sequence is (0 to 100).
                9. num_mean: Mean.
                10. num_median: Median.
                11. num_min: Minimum.
                12. num_max: Maximum.
                13. num_std_dev: Standard deviation.
                14. num_iqr: Interquartile range (q75 - q25).
                15. num_skewness: Skewness of the numeric distribution.
                16. date_min: Earliest detected date.
                17. date_max: Latest detected date.
                18. date_range_days: Days between date_min and date_max.
                19. freshness_score: How recent date_max is (0 to 100).
                """
            )

    
    with tab_health:
        st.subheader("Health")
        st.caption("Table level health rollups plus quick column status distributions.")

        def _fmt1(x: Any) -> str:
            try:
                return f"{float(x):.1f}"
            except Exception:
                return "n/a"

        def _pct1(x: Any) -> str:
            try:
                return f"{float(x):.1f}%"
            except Exception:
                return "n/a"

        def _status_badge(status: str) -> str:
            s = (status or "").lower()
            if s == "ok":
                return "🟢 ok"
            if s == "warn":
                return "🟡 warn"
            if s == "fail":
                return "🔴 fail"
            if s == "null":
                return "⚪ null"
            return f"⚪ {s or 'unknown'}"

        # -----------------------------
        # Global health
        # -----------------------------
        global_health = extras.get("health", {}) if isinstance(extras, dict) else {}
        g_avg = global_health.get("avg_score")

        g1, g2, g3 = st.columns([2, 1, 1])
        with g1:
            st.markdown("### Overall")
            if g_avg is None:
                st.info("Global average health score not available.")
            else:
                st.metric("Global average health score", _fmt1(g_avg))
        with g2:
            # Optional: show how many tables we have
            n_tables = len(extras.get("tables", {})) if isinstance(extras, dict) and isinstance(extras.get("tables"), dict) else 0
            st.metric("Tables", n_tables)
        with g3:
            # Optional: total columns
            try:
                st.metric("Columns", int(len(profiles_df)))
            except Exception:
                st.metric("Columns", "n/a")

        st.divider()

        # -----------------------------
        # Per table rollup (prettier)
        # -----------------------------
        rollup = _health_rollup_df(extras, profiles_df)

        if rollup.empty:
            st.warning("No table health rollup found. Clear cache and rerun.")
        else:
            roll = rollup.copy()
            # Add percentages + nice labels
            total_cols = (roll["ok_cols"] + roll["warn_cols"] + roll["fail_cols"]).replace(0, pd.NA)
            roll["ok_pct"] = ((roll["ok_cols"] / total_cols) * 100.0).round(1)
            roll["warn_pct"] = ((roll["warn_cols"] / total_cols) * 100.0).round(1)
            roll["fail_pct"] = ((roll["fail_cols"] / total_cols) * 100.0).round(1)

            roll["avg_health_score"] = roll["avg_health_score"].astype(float).round(1)

            # Emoji columns
            roll["🟢 ok"] = roll["ok_cols"].astype("Int64").astype(str) + " (" + roll["ok_pct"].astype("Float64").map(lambda v: _pct1(v) if pd.notna(v) else "n/a") + ")"
            roll["🟡 warn"] = roll["warn_cols"].astype("Int64").astype(str) + " (" + roll["warn_pct"].astype("Float64").map(lambda v: _pct1(v) if pd.notna(v) else "n/a") + ")"
            roll["🔴 fail"] = roll["fail_cols"].astype("Int64").astype(str) + " (" + roll["fail_pct"].astype("Float64").map(lambda v: _pct1(v) if pd.notna(v) else "n/a") + ")"

            show_roll = roll[["table_name", "avg_health_score", "🟢 ok", "🟡 warn", "🔴 fail"]].rename(
                columns={"avg_health_score": "avg score"}
            )

            with st.container(border=True):
                st.markdown("### Table rollup")
                st.dataframe(show_roll, use_container_width=True, height=280)

        st.divider()

        # -----------------------------
        # Drilldown per table
        # -----------------------------
        t2 = st.selectbox(
            "Inspect table health",
            options=sorted(profiles_df["table_name"].unique().tolist()),
            key="health_table_pick",
        )

        h2 = _get_table_health(extras, t2)

        sub = profiles_df[profiles_df["table_name"] == t2].copy()
        n_cols_tbl = int(len(sub))

        if not h2:
            st.warning("No health data found for this table. Cache might be stale.")
        else:
            counts = h2.get("counts") or {}
            ok_n = int(counts.get("ok", 0) or 0)
            warn_n = int(counts.get("warn", 0) or 0)
            fail_n = int(counts.get("fail", 0) or 0)

            denom = max(1, ok_n + warn_n + fail_n)
            ok_p = (ok_n / denom) * 100.0
            warn_p = (warn_n / denom) * 100.0
            fail_p = (fail_n / denom) * 100.0

            avg2 = h2.get("avg_score")

            with st.container(border=True):
                st.markdown(f"### {t2}")

                m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
                with m1:
                    st.metric("Avg health score", _fmt1(avg2))
                with m2:
                    st.metric("Columns", n_cols_tbl)
                with m3:
                    st.metric("🟢 ok", f"{ok_n} ({ok_p:.1f}%)")
                with m4:
                    st.metric("🟡 warn", f"{warn_n} ({warn_p:.1f}%)")

                # second row for fail (keeps it clean)
                m5, _ = st.columns([1, 3])
                with m5:
                    st.metric("🔴 fail", f"{fail_n} ({fail_p:.1f}%)")

            st.divider()


            # -----------------------------
            # Highlight problem columns
            # -----------------------------
            st.markdown("### Attention needed")
            st.caption("Fail columns and warn columns with medium severity.")

            if "health_status" in sub.columns:
                filt = (sub["health_status"] == "fail") | (
                    (sub["health_status"] == "warn") & (sub.get("health_severity") == "medium")
                )

                cols = ["column_name", "health_status", "health_severity", "health_score", "health_reasons"]
                cols = [c for c in cols if c in sub.columns]

                bad = sub.loc[filt, cols].copy()

                if bad.empty:
                    st.success("No columns flagged as fail or warn medium in this table.")
                else:
                    # Add emoji status + score formatting
                    bad["health_status"] = bad["health_status"].astype(str).map(_status_badge)
                    if "health_score" in bad.columns:
                        bad["health_score"] = pd.to_numeric(bad["health_score"], errors="coerce").round(1)

                    # Keep the table readable
                    bad = bad.rename(
                        columns={
                            "column_name": "column",
                            "health_status": "status",
                            "health_severity": "severity",
                            "health_score": "score",
                            "health_reasons": "reasons",
                        }
                    )
                    st.dataframe(bad, use_container_width=True, height=320)

    with tab_raw:
        st.subheader("Raw outputs")
        st.caption("Download the full enhanced profiles, or inspect the nested extras JSON.")

        st.dataframe(profiles_df, use_container_width=True, height=520)
        with st.expander("Extras JSON"):
            st.json(extras)

        _download_df_buttons(profiles_df, "profiles_enhanced")

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        if st.button("Clear enhanced outputs", use_container_width=True):
            st.session_state.pop("enhanced_out", None)
            st.session_state.pop("enhanced_out_sig", None)
            st.toast("Cleared enhanced profiler outputs.", icon="🧹")
    with cR:
        if st.button("Clear Streamlit cache and rerun", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("enhanced_out", None)
            st.session_state.pop("enhanced_out_sig", None)
            st.toast("Cleared Streamlit cache.", icon="🧼")
            st.rerun()
