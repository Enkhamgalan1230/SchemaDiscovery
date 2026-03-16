from __future__ import annotations

import pandas as pd


def edges_to_dot(edges_df: pd.DataFrame, show_labels: bool = True) -> str:
    """
    Convert edges_df (fk_table,fk_column,pk_table,pk_column,optional,cardinality,score,mode)
    into a Graphviz DOT string usable by st.graphviz_chart().

    Node colouring:
      - Parent only (appears only as pk_table) -> green-ish
      - Child only  (appears only as fk_table) -> blue-ish
      - Both (appears as pk_table and fk_table) -> light grey

    Edge styling:
      - optional=True -> dashed
      - mode=="soft"  -> dashed if optional is missing, otherwise keeps optional dash
      - score         -> thicker penwidth
    """
    required = {"fk_table", "pk_table"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges_df missing required columns: {sorted(missing)}")

    # --- role sets
    pk_tables = set(edges_df["pk_table"].astype(str))
    fk_tables = set(edges_df["fk_table"].astype(str))
    tables = sorted(pk_tables | fk_tables)

    # --- colour palette (tweak freely)
    COLOR_PARENT = "#57B5F3"  # light green
    COLOR_CHILD = "#4DF347"   # light blue
    COLOR_BOTH = "#DA6DFC"    # light grey
    COLOR_BORDER = "#d3bbbb"
    COLOR_EDGE = "#aab6cc"

    def _node_fill(t: str) -> str:
        is_parent = t in pk_tables
        is_child = t in fk_tables
        if is_parent and not is_child:
            return COLOR_PARENT
        if is_child and not is_parent:
            return COLOR_CHILD
        if is_parent and is_child:
            return COLOR_BOTH
        return "#ffffff"

    # --- DOT header
    lines = [
        "digraph ERD {",
        '  rankdir="LR";',
        '  graph [bgcolor="#ffffff", pad="0.2", nodesep="0.4", ranksep="0.6"];',
        '  node  [shape=box, style="rounded,filled", color="%s", fontname="Helvetica"];' % COLOR_BORDER,
        '  edge  [color="%s", fontname="Helvetica", fontsize=10];' % COLOR_EDGE,
        "",
    ]

    # --- nodes with per-table colour
    for t in tables:
        fill = _node_fill(t)
        # quote names to handle weird characters safely
        lines.append(f'  "{t}" [fillcolor="{fill}"];')

    lines.append("")

    # --- edges
    has_optional = "optional" in edges_df.columns
    has_mode = "mode" in edges_df.columns
    has_score = "score" in edges_df.columns
    has_pk_col = "pk_column" in edges_df.columns
    has_fk_col = "fk_column" in edges_df.columns

    for r in edges_df.itertuples(index=False):
        pk_t = str(getattr(r, "pk_table"))
        fk_t = str(getattr(r, "fk_table"))

        pk_c = str(getattr(r, "pk_column")) if has_pk_col else ""
        fk_c = str(getattr(r, "fk_column")) if has_fk_col else ""

        label = ""
        if show_labels and (pk_c or fk_c):
            label = f"{pk_c} -> {fk_c}".strip(" ->")

        # Optional relationship -> dashed edge
        optional = bool(getattr(r, "optional", False)) if has_optional else False

        # Soft edges (if you output mode) -> also dashed (but do not override optional)
        mode = str(getattr(r, "mode", "")).lower() if has_mode else ""
        is_soft = mode == "soft"

        # Style logic:
        # - if optional True -> dashed
        # - else if soft -> dashed (so you can visually spot soft links)
        edge_style = ""
        if optional or is_soft:
            edge_style = 'style="dashed"'

        # arrow
        arrow = 'arrowhead="normal" arrowtail="none"'

        # Score -> thicker line (clipped range)
        if has_score:
            try:
                score = float(getattr(r, "score"))
                score_clipped = max(0.0, min(1.0, score))
                penwidth = 1.0 + 4.0 * score_clipped
            except Exception:
                penwidth = 1.5
        else:
            penwidth = 1.5

        attrs = [arrow, f'penwidth="{penwidth:.2f}"']
        if edge_style:
            attrs.append(edge_style)
        if label:
            safe_label = label.replace('"', r"\"")
            attrs.append(f'label="{safe_label}"')

        lines.append(f'  "{pk_t}" -> "{fk_t}" [{", ".join(attrs)}];')

    # --- (Optional) tiny legend as nodes (kept simple)
    lines += [
        "",
        '  subgraph cluster_legend {',
        '    label="Legend";',
        '    fontsize=10;',
        '    color="#e5e7eb";',
        '    style="rounded";',
        '    "Parent only" [shape=box, style="rounded,filled", fillcolor="%s"];' % COLOR_PARENT,
        '    "Child only"  [shape=box, style="rounded,filled", fillcolor="%s"];' % COLOR_CHILD,
        '    "Both"        [shape=box, style="rounded,filled", fillcolor="%s"];' % COLOR_BOTH,
        '  }',
    ]

    lines.append("}")
    return "\n".join(lines)