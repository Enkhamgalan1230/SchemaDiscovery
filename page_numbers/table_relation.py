import streamlit as st
import pandas as pd

from schema_discovery.pipeline.run import run_schema_discovery, PipelineConfig
from schema_discovery.viz.erd_graphviz import edges_to_dot


@st.cache_data(show_spinner=False)
def load_dfs(files):
    # Keep as-is, but you may want dtype="string" for stable key handling.
    return {f.name.rsplit(".", 1)[0]: pd.read_csv(f) for f in files}


col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Data Schematics App")
    st.caption("Drop CSV files to load tables and discover relationships.")

    files = st.file_uploader("Drop CSV files here", type=["csv"], accept_multiple_files=True)
    if not files:
        st.info("Upload one or more CSV files to continue.")
        st.stop()

    # Clear old outputs if user uploads a different set of files
    upload_key = tuple(sorted((f.name, f.size) for f in files))
    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("schema_out", None)

    dfs = load_dfs(files)
    st.session_state["dfs_uploaded"] = dfs

    st.subheader("Preview")
    selected_table = st.selectbox("Select table", options=list(dfs.keys()))
    st.dataframe(dfs[selected_table].head(5), use_container_width=True)

    st.divider()
    st.subheader("Schema Discovery")

    # -------------------------
    # Mode toggle
    # -------------------------
    ind_mode = st.radio(
        "IND discovery mode",
        options=["strict", "soft"],
        index=0,
        horizontal=True,
        help=(
            "strict -> exact value matching (fast, but brittle on messy exports)\n"
            "soft -> representation matching (handles leading zeros, float -> int-like, etc)"
        ),
    )

    # Optional: keep a small config UI. This avoids you hardcoding.
    with st.expander("Advanced thresholds", expanded=False):
        min_edge_score = st.slider("Min edge score", 0.0, 1.0, 0.80, 0.01)

        if ind_mode == "strict":
            ind_min_distinct_coverage = st.slider("Strict min distinct coverage", 0.0, 1.0, 0.90, 0.01)
            ind_min_non_null_rows = st.number_input("Min non null rows", min_value=0, value=20, step=1)
            ind_min_distinct_child = st.number_input("Min distinct child values", min_value=0, value=2, step=1)

        else:
            soft_min_distinct_coverage = st.slider("Soft min distinct coverage", 0.0, 1.0, 0.80, 0.01)
            soft_min_row_coverage = st.slider("Soft min row coverage", 0.0, 1.0, 0.80, 0.01)
            soft_min_name_sim_for_routing = st.slider("Soft routing min name similarity", 0.0, 1.0, 0.30, 0.01)
            soft_max_parents_per_fk = st.number_input("Soft max parents per FK", min_value=1, value=20, step=1)
            soft_small_domain_fk_distinct_max = st.number_input(
                "Soft small-domain FK distinct max", min_value=0, value=30, step=1
            )
            soft_small_domain_min_name_sim = st.slider("Soft small-domain min name similarity", 0.0, 1.0, 0.65, 0.01)

            ind_min_non_null_rows = st.number_input("Min non null rows", min_value=0, value=20, step=1)
            # still relevant for child gating in your pipeline, if you use it for both modes
            ind_min_distinct_child = st.number_input("Min distinct child values", min_value=0, value=2, step=1)

    # -------------------------
    # Build PipelineConfig
    # -------------------------
    if ind_mode == "strict":
        cfg = PipelineConfig(
            ind_mode="strict",
            ind_dtype_families=("int", "string"),
            ind_min_distinct_child=int(ind_min_distinct_child),
            ind_min_non_null_rows=int(ind_min_non_null_rows),
            ind_min_distinct_coverage=float(ind_min_distinct_coverage),
            min_edge_score=float(min_edge_score),
        )
    else:
        cfg = PipelineConfig(
            ind_mode="soft",
            # keep these if your pipeline uses them for basic candidate gating
            ind_min_distinct_child=int(ind_min_distinct_child),
            ind_min_non_null_rows=int(ind_min_non_null_rows),
            min_edge_score=float(min_edge_score),

            # soft knobs (must exist on PipelineConfig + run.py wiring)
            soft_ind_min_distinct_coverage=float(soft_min_distinct_coverage),
            soft_ind_min_row_coverage=float(soft_min_row_coverage),
            soft_ind_min_name_sim_for_routing=float(soft_min_name_sim_for_routing),
            soft_ind_max_parents_per_fk=int(soft_max_parents_per_fk),
            soft_ind_small_domain_fk_distinct_max=int(soft_small_domain_fk_distinct_max),
            soft_ind_small_domain_min_name_sim=float(soft_small_domain_min_name_sim),
        )

    if st.button("Run schema discovery", type="primary"):
        with st.spinner(f"Running schema discovery (mode -> {ind_mode})..."):
            st.session_state["schema_out"] = run_schema_discovery(dfs, config=cfg)

    out = st.session_state.get("schema_out")
    if not out:
        st.info("Click Run schema discovery to generate relationships.")
        st.stop()

    edges_df = out.get("edges_df")
    if edges_df is None or edges_df.empty:
        st.warning("No relationships were discovered.")
        st.stop()

    st.success(f"Discovered {len(edges_df)} relationship(s).")

    # Nice small metadata so you can sanity check what was run
    with st.expander("Run details"):
        st.write(
            {
                "ind_mode": ind_mode,
                "min_edge_score": float(cfg.min_edge_score),
                "ind_min_non_null_rows": int(cfg.ind_min_non_null_rows),
                "ind_min_distinct_child": int(cfg.ind_min_distinct_child),
                # strict only
                "ind_min_distinct_coverage": float(getattr(cfg, "ind_min_distinct_coverage", 0.0)),
                # soft only
                "soft_ind_min_distinct_coverage": float(getattr(cfg, "soft_ind_min_distinct_coverage", 0.0)),
                "soft_ind_min_row_coverage": float(getattr(cfg, "soft_ind_min_row_coverage", 0.0)),
                "soft_ind_min_name_sim_for_routing": float(getattr(cfg, "soft_ind_min_name_sim_for_routing", 0.0)),
                "soft_ind_max_parents_per_fk": int(getattr(cfg, "soft_ind_max_parents_per_fk", 0)),
                "soft_ind_small_domain_fk_distinct_max": int(getattr(cfg, "soft_ind_small_domain_fk_distinct_max", 0)),
                "soft_ind_small_domain_min_name_sim": float(getattr(cfg, "soft_ind_small_domain_min_name_sim", 0.0)),
            }
        )

    st.subheader("Discovered Relationships")
    st.dataframe(edges_df, use_container_width=True)

    st.subheader("Entity Relationship Diagram")
    dot = edges_to_dot(edges_df)
    st.graphviz_chart(dot, use_container_width=True)

    with st.expander("Show DOT source"):
        st.code(dot, language="dot")
        st.download_button(
            label="Download DOT",
            data=dot,
            file_name="erd.dot",
            mime="text/vnd.graphviz",
            use_container_width=True,
            type="primary",
        )
